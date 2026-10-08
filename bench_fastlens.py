#> name: bench_fastlens.py
#> author: John Miller Jr (benchmark drafted with Claude)
#> descrp: times the current opus quad generation vs the vectorized fastlens prototype (prints only; saves nothing)

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
import fastlens as fl


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> declarations
    numGals       = 10                                     # galaxies
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
    outs = [fl.lens_galaxy(bp, numSource_gal, nph=nph, rng=rng, method='deflect', observables=observables) for bp in bprofiles]
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
