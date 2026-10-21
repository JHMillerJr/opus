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
import lensing
import profiles
from modules.units import u; u=u()


""" #> DIAGNOSTICS ===================
================================== """

#> newton-refines points onto the nearest root of the lens equation (analytic deflections)
def refine(x, y, src, bprof, nit=100, maxstep=0.005):

    #> declarations
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)       # arcsec

    #> newton loop (step capped at maxstep)
    for _ in range(nit):
        ax, ay, axx, axy, ayx, ayy = profiles.alpha_jac_points(x, y, bprof)
        rx, ry = x - ax - src[0], y - ay - src[1]                     # arcsec
        ja, jb, jc, jd = 1-axx, -axy, -ayx, 1-ayy                     # lensing jacobian
        det    = ja*jd - jb*jc
        sx, sy = (jd*rx - jb*ry)/det, (-jc*rx + ja*ry)/det            # arcsec
        lam    = np.minimum(1.0, maxstep/np.maximum(np.hypot(sx, sy), 1e-300))
        x, y   = x - lam*sx, y - lam*sy                               # arcsec

    #> final lens-equation residual
    ax, ay = profiles.alpha_points(x, y, bprof)                       # arcsec
    res    = np.hypot(x - ax - src[0], y - ay - src[1])               # arcsec

    return x, y, res # arcsec, arcsec, arcsec


#> groups points closer than tol (returns the unique points)
def unique_pts(p, tol):
    uniq = []
    for q in p:
        if len(uniq) == 0 or np.min(np.hypot(*(np.array(uniq) - q).T)) >= tol: uniq.append(q)
    return np.array(uniq).reshape(-1, 2)


#> number of points in a that have no point of b within tol
def unmatched(a, b, tol):
    if len(a) == 0: return 0
    if len(b) == 0: return len(a)
    return int((np.linalg.norm(a[:,None,:] - b[None,:,:], axis=-1).min(axis=1) > tol).sum())


#> diagnoses why old and new disagree for one source
def diagnose(src, bprof, old_raw, new_ims, new_mu, pix_arc, **kwargs):

    #> kwargs
    tol_res   = kwargs.get('tol_res', 1e-7)       # residual of a true root [arcsec]
    tol_same  = kwargs.get('tol_same', 1e-5)      # two roots closer than this are the same [arcsec]
    mu_hi     = kwargs.get('mu_hi', 50)           # |mu| above this counts as a merging (fold/cusp) image

    #> refining the old raw images + checking the new images solve the lens equation
    xr, yr, res_old = refine(old_raw[:,0], old_raw[:,1], src, bprof)  # arcsec
    _, _, res_new   = refine(new_ims[:,0], new_ims[:,1], src, bprof, nit=0)
    conv            = res_old < tol_res                               # old points that converged to a root
    old_roots       = unique_pts(np.column_stack((xr, yr))[conv], tol_same) # arcsec
    true_roots      = unique_pts(np.vstack((old_roots, new_ims)), tol_same) # arcsec

    #> counts
    d = {'n_old':      len(old_raw),                                  # old raw crossings
         'n_new':      len(new_ims),                                  # new images
         'n_true':     len(true_roots),                               # distinct roots found by either
         'old_dups':   int(conv.sum()) - len(old_roots),              # extra old crossings on the same root
         'old_noconv': int((~conv).sum()),                            # old crossings that never reach a root
         'old_missed': unmatched(new_ims, old_roots, tol_same),       # true images the old code did not find
         'new_missed': unmatched(old_roots, new_ims, tol_same),       # true images the new code did not find
         'new_bad':    int((res_new > tol_res).sum())}                # new images that do not solve the lens eq

    #> old raw distances to the nearest true root
    d['old_off'] = np.linalg.norm(old_raw[:,None,:] - true_roots[None,:,:], axis=-1).min(axis=1).max() # arcsec

    #> image geometry: merging images and closest pair
    n_hi      = int((np.abs(new_mu) > mu_hi).sum())
    d['type'] = 'cusp' if n_hi >= 3 else 'fold' if n_hi == 2 else '-'
    sep       = np.linalg.norm(true_roots[:,None,:] - true_roots[None,:,:], axis=-1)
    d['sep']  = sep[np.triu_indices(len(true_roots), k=1)].min() if len(true_roots) > 1 else np.nan # arcsec
    d['sep_pix'] = d['sep'] * pix_arc                                 # pix

    #> classification (worst first)
    if   d['new_bad']    > 0: d['class'] = 'NEW IMAGE DOES NOT SOLVE LENS EQ'
    elif d['new_missed'] > 0: d['class'] = 'NEW MISSED IMAGE(S)'
    elif d['old_missed'] > 0 and d['old_dups'] > 0: d['class'] = 'old missed + duplicate crossings'
    elif d['old_missed'] > 0: d['class'] = 'old missed image(s)'
    elif d['old_dups']   > 0: d['class'] = 'old duplicate crossings'
    elif d['old_noconv'] > 0: d['class'] = 'old spurious crossings (no root)'
    else:                     d['class'] = 'same roots (rejected for another reason)'

    return d



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

    #>>># DIAGNOSING DISAGREEMENTS #<<<#

    #> sources where old and new do not both return a complete quad
    newM = np.vstack([o['mu'] for o in outs])                             # signed mags
    miss = np.flatnonzero(~ok)
    print(f'> sources w/o a complete quad in both: {len(miss)}')

    #> diagnosing each one
    diags = []
    for i in miss:

        #> galaxy index + raw old crossings (no ordering, no jims cut)
        g      = i // numSource_gal                                       # galaxy index
        xs, ys = srcs[i] / u.arc_rad                                      # arcsec --> rad
        with contextlib.redirect_stdout(io.StringIO()):
            old_raw = lensing.lfims(X / pix_arc[g] / u.arc_rad,           # rad
                                    Y / pix_arc[g] / u.arc_rad,           # rad
                                    gx[g], gy[g], xs, ys, pix_arc[g], nph, ordered=False)
        old_raw = np.asarray(old_raw, dtype=float).reshape(-1, 2)         # arcsec

        #> new images + mags (finite only)
        fin = np.isfinite(newI[i,:,0])
        dg  = diagnose(srcs[i], bprofiles[g], old_raw, newI[i][fin], newM[i][fin], pix_arc[g])
        diags.append(dg)

        #> print
        print(f'> source {i:5d} [{dg["type"]:4s}, closest pair {dg["sep"]*1e3:6.1f} mas = {dg["sep_pix"]:5.2f} pix]: '
              f'old {dg["n_old"]}, new {dg["n_new"]}, true {dg["n_true"]} | old dups {dg["old_dups"]}, old missed {dg["old_missed"]}, '
              f'old no-root {dg["old_noconv"]}, new missed {dg["new_missed"]}, new bad {dg["new_bad"]}, '
              f'old max offset {dg["old_off"]*1e3:5.1f} mas --> {dg["class"]}')

    #> summary of classes
    if len(diags):
        classes, counts = np.unique([dg['class'] for dg in diags], return_counts=True)
        print('> summary:')
        for c, n in zip(classes, counts): print(f'>   {n:4d}  {c}')

    #> print
    print(f'> OLD: {t_old:.2f} s for {numSrc} sources --> {t_old/numSrc*1e3:.3f} ms/quad')
    print(f'> NEW: {t_new:.2f} s for {numSrc} sources --> {t_new/numSrc*1e3:.3f} ms/quad')
    print(f'> speed-up: {t_old/t_new:.1f}x')
    print(f'> old complete quads (same sources): {ok.sum()}/{numSrc}')
    print(f'> old vs new image positions: median {np.median(d)*1e3:.2f} mas, max {d.max()*1e3:.1f} mas')
    
    import sys
    sys.exit()
    
    #> order-independent comparison (nearest new image for each old image)
    dmat  = np.linalg.norm(oldI[ok][:,:,None,:] - newI[ok][:,None,:,:], axis=-1)  # (Q, 5, 5) arcsec
    d_set = dmat.min(axis=2).max(axis=1)                                           # arcsec
    print(f'> order-independent: median {np.median(d_set)*1e3:.2f} mas, max {d_set.max()*1e3:.1f} mas')

    #> same positions, different order (order-dependent > 1 pix, order-independent < 1 pix)
    swap = (d > 1/pix_arc[0]) & (d_set < 1/pix_arc[0])
    print(f'> order mismatches (positions agree, order differs): {swap.sum()}/{ok.sum()}')
    
    #> worst quad (in the original comparison)
    w = np.argmax(d)
    print(f'> worst quad: source {np.flatnonzero(ok)[w]} at {srcs[ok][w]} [arcsec]')
    print('> old images [arcsec]:\n', oldI[ok][w])
    print('> new images [arcsec]:\n', newI[ok][w])

    # end
# thank
