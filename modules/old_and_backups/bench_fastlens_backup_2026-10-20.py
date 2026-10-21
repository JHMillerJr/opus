#> name: bench_fastlens.py
#> author: John Miller Jr
#> descrp: times the current opus quad generation vs the vectorized newton version (generate.genQuadNewt) (prints only; saves nothing)

""" #> IMPORTS =======================
================================== """

#> standard imports
import io
import time
import contextlib
import numpy as np

#> modules (run from the opus directory)
import params
import generate
import deflectionCPU2 as deflection


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> declarations
    numGals       = 100                                     # galaxies
    numSource_gal = 100                                    # sources per galaxy
    nph           = 50                                     # grid half-width [pix]
    observables   = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    rng           = np.random.default_rng(42)

    #> galaxies (no priors; fixed redshifts; pixel scale grows w/ nph --> fixed field of view)
    galProf     = generate.galProfiles(nfw=True, hern=True, mult=[1,3,4], ex=False)
    paramRanges = params.toggleParams(galProf)
    pix_arc     = np.full(numGals, 60 * nph/50)            # pix/arcsec
    bprofiles   = params.bprofiles(numGals, ranges=paramRanges, pix_arc=pix_arc, redshifts=np.tile((0.5, 1.0), (numGals, 1)))
    numSrc      = numGals*numSource_gal

    #>>># NEW #<<<#
    srt  = time.time()
    outs = [generate.genQuadNewt(bp, numSource_gal, nph=nph, rng=rng, method='deflect', observables=observables) for bp in bprofiles]
    t_new = time.time() - srt
    srcs  = np.vstack([o['srcs'] for o in outs])          # arcsec

    #>>># OLD (random sources: geopandas + shapely, mags + observables) #<<<#
    X, Y = generate.grid(nph=nph)
    with contextlib.redirect_stdout(io.StringIO()):
        srt = time.time()
        gx, gy, lamt = deflection.deflection(X, Y, bprofiles)
        old_r = generate.genQuadPop((X, Y, gx, gy, lamt), bprofiles, method='deflect', jims=5, numSource_gal=numSource_gal,
                                    observables=observables, mags=True, seed=1, warnings=False)
        t_old = time.time() - srt

    #>>># OLD (same sources as new, for comparing results) #<<<#
    with contextlib.redirect_stdout(io.StringIO()):
        old_s = generate.genQuadPop((X, Y, gx, gy, lamt), bprofiles, method='deflect', jims=5, numSource_gal=numSource_gal,
                                    observables=observables, mags=True, source=srcs, seed=1, warnings=False)

    #> comparing image positions
    newI = np.vstack([o['images'] for o in outs])                         # arcsec
    oldI = np.array([np.asarray(q, dtype=float) for q in old_s[0]])        # arcsec
    ok   = ~np.isnan(oldI).any(axis=(1,2)) & ~np.isnan(newI).any(axis=(1,2))
    d    = np.linalg.norm(oldI[ok] - newI[ok], axis=2).max(axis=1)         # arcsec
    
    #> sources the old code did not return as complete quads
    miss = np.flatnonzero(~ok)
    print(f'> old incomplete: {len(miss)} sources')
    for i in miss: print(f'> source {i} at {srcs[i]} [arcsec], old finite images: {np.isfinite(oldI[i,:,0]).sum()}, new mags: {np.round(np.vstack([o["mu"] for o in outs])[i], 1)}')

    
    #> smallest separation between any two images for each missed source
    for i in miss:
        p    = newI[i][np.isfinite(newI[i,:,0])]                               # arcsec
        dmat = np.linalg.norm(p[:,None,:] - p[None,:,:], axis=-1)              # arcsec
        dmin = dmat[np.triu_indices(len(p), k=1)].min()                        # arcsec
        print(f'> source {i}: closest pair = {dmin*1e3:.1f} mas ({dmin*60:.2f} pix)')
    
    #> separation of the high-mag (fold) pair for each missed source
    newM = np.vstack([o['mu'] for o in outs])                              # signed mags
    for i in miss:
        j    = np.argsort(-np.abs(newM[i]))[:2]                            # two largest |mu|
        dsep = np.linalg.norm(newI[i,j[0]] - newI[i,j[1]])                 # arcsec
        print(f'> source {i}: fold pair separation = {dsep*1e3:.1f} mas ({dsep*60:.2f} pix)')
        
        #> raw number of images the old finder returns for each missed source
    import lensing
    from modules.units import u; u=u()
    for i in miss:
        g      = i // numSource_gal                                      # galaxy index
        xs, ys = srcs[i] / u.arc_rad                                     # arcsec --> rad
        ims    = lensing.lfims(X / pix_arc[g] / u.arc_rad,               # rad
                               Y / pix_arc[g] / u.arc_rad,               # rad
                               gx[g], gy[g], xs, ys, pix_arc[g], nph, ordered=False)
        print(f'> source {i}: old raw images = {len(ims)}')
        
            #> where are the old images relative to the new ones, and do they solve the lens equation?
        import profiles
        newp   = newI[i][np.isfinite(newI[i,:,0])]                                       # arcsec
        dnn    = np.linalg.norm(ims[:,None,:] - newp[None,:,:], axis=-1).min(axis=1)     # arcsec
        ax, ay = profiles.alpha_points(ims[:,0], ims[:,1], bprofiles[g])                 # arcsec
        res    = np.hypot(ims[:,0] - ax - srcs[i,0], ims[:,1] - ay - srcs[i,1])          # arcsec
        for p, dn, r in zip(ims, dnn, res): print(f'>   old image {np.round(p, 4)} [arcsec]: nearest new = {dn*1e3:.1f} mas, residual = {r*1e3:.2f} mas')
    
        print(f'> source {i}: old images > 10 mas from any new image = {(dnn > 0.010).sum()}')
        
                #> newton-refining each old image w/ analytic deflections (where does it really converge?)
        xr, yr = ims[:,0].copy(), ims[:,1].copy()                                       # arcsec
        for _ in range(50):
            ax, ay, axx, axy, ayx, ayy = profiles.alpha_jac_points(xr, yr, bprofiles[g])
            rx, ry = xr - ax - srcs[i,0], yr - ay - srcs[i,1]                           # arcsec
            a, b, c, d = 1-axx, -axy, -ayx, 1-ayy
            det = a*d - b*c
            sx, sy = (d*rx - b*ry)/det, (-c*rx + a*ry)/det                              # arcsec
            lam = np.minimum(1.0, 0.005/np.maximum(np.hypot(sx, sy), 1e-300))           # step capped at 5 mas
            xr -= lam*sx                                                                # arcsec
            yr -= lam*sy                                                                # arcsec
        dnr = np.linalg.norm(np.column_stack((xr, yr))[:,None,:] - newp[None,:,:], axis=-1).min(axis=1)  # arcsec
        nr  = len(np.unique(np.round(np.column_stack((xr, yr)), 6), axis=0))
        print(f'> source {i}: refined old images not on a new image = {(dnr > 1e-5).sum()}, distinct roots = {nr}')

    #> print
    print(f'> OLD: {t_old:.2f} s for {numSrc} sources --> {t_old/numSrc*1e3:.3f} ms/quad')
    print(f'> NEW: {t_new:.2f} s for {numSrc} sources --> {t_new/numSrc*1e3:.3f} ms/quad')
    print(f'> speed-up: {t_old/t_new:.1f}x')
    print(f'> old complete quads (same sources): {ok.sum()}/{numSrc}')
    print(f'> old vs new image positions: median {np.median(d)*1e3:.2f} mas, max {d.max()*1e3:.1f} mas')
    
    #> order-independent comparison (nearest new image for each old image)
    dmat  = np.linalg.norm(oldI[ok][:,:,None,:] - newI[ok][:,None,:,:], axis=-1)  # (Q, 5, 5) arcsec
    d_set = dmat.min(axis=2).max(axis=1)                                           # arcsec
    print(f'> order-independent: median {np.median(d_set)*1e3:.2f} mas, max {d_set.max()*1e3:.1f} mas')
    
    #> worst quad (in the original comparison)
    w = np.argmax(d)
    print(f'> worst quad: source {np.flatnonzero(ok)[w]} at {srcs[ok][w]} [arcsec]')
    print('> old images [arcsec]:\n', oldI[ok][w])
    print('> new images [arcsec]:\n', newI[ok][w])

    # end
# thank
