#> name: check_newtlens.py
#> author: John Miller Jr
#> descrp: stress test of the newton image finder (newtlens.find_images): finds where and why it misses images (prints only; saves nothing)

""" #> IMPORTS =======================
================================== """

#> standard imports
import io
import time
import contextlib
import numpy as np
from matplotlib.path import Path

#> modules (run from the opus directory)
import params
import generate
import profiles
import newtlens


""" #> GRID ==========================
================================== """

#> lens-plane grid, beta map + tangential eigenvalue (same as generate.genQuadNewt)
def make_grid(bprof, nph, pix_arc):

    #> lens-plane grid (theta)
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc     # arcsec
    xg, yg = np.meshgrid(w, w)                             # arcsec

    #> deflection angles + beta
    ax, ay   = profiles.alpha_points(xg, yg, bprof)        # arcsec
    bxg, byg = xg - ax, yg - ay                            # arcsec

    #> tangential eigenvalue
    axy_, axx_ = np.gradient(ax, w, w)
    ayy_, ayx_ = np.gradient(ay, w, w)
    kappa  = 0.5*(axx_ + ayy_)
    gamma1 = 0.5*(axx_ - ayy_)
    gamma2 = 0.5*(axy_ + ayx_)
    lamt   = 1 - kappa - np.hypot(gamma1, gamma2)

    return xg, yg, bxg, byg, lamt # arcsec, arcsec, arcsec, arcsec, --


""" #> SOURCES =======================
================================== """

#> sources just inside the caustic, at a log-uniform distance delta from it [arcsec]
def near_caustic_sources(caus, nsrc, dmin, dmax, rng):

    #> declarations
    path = Path(caus)
    seg  = np.diff(caus, axis=0)                           # caustic segments [arcsec]
    slen = np.hypot(seg[:,0], seg[:,1])                    # arcsec
    keep = slen > 0
    p0, seg, slen = caus[:-1][keep], seg[keep], slen[keep]

    #> drawing batches until enough sources are found
    out, dout = np.empty((0, 2)), np.empty(0)
    while len(out) < nsrc:

        #> random points along the caustic (weighted by segment length)
        k  = 2*(nsrc - len(out))
        i  = rng.choice(len(seg), size=k, p=slen/slen.sum())
        t  = rng.uniform(0, 1, k)[:,None]
        pt = p0[i] + t*seg[i]                              # arcsec

        #> stepping a distance delta along the normal, onto the inside
        nrm   = np.column_stack((-seg[i,1], seg[i,0])) / slen[i,None]
        delta = 10**rng.uniform(np.log10(dmin), np.log10(dmax), k)  # arcsec
        pin   = pt + delta[:,None]*nrm
        flip  = ~path.contains_points(pin)
        pin[flip] = pt[flip] - delta[flip,None]*nrm[flip]
        ok    = path.contains_points(pin)

        #> saving
        out  = np.vstack([out, pin[ok]])
        dout = np.concatenate([dout, delta[ok]])

    return out[:nsrc], dout[:nsrc] # arcsec, arcsec


""" #> IMAGE FINDING =================
================================== """

#> runs find_images in chunks of sources (quietly; > jmax messages are counted instead)
def find(xg, yg, bxg, byg, srcs, bprof, chunk=250):

    #> declarations
    ims, mus, nims = [], [], []

    #> chunks
    for k in range(0, len(srcs), chunk):
        with contextlib.redirect_stdout(io.StringIO()):
            r = newtlens.find_images(xg, yg, bxg, byg, srcs[k:k+chunk], bprof)
        ims  += [r['images'][j][np.isfinite(r['images'][j][:,0])] for j in range(len(r['nims']))]
        mus  += [r['mu'][j][np.isfinite(r['mu'][j])] for j in range(len(r['nims']))]
        nims += list(r['nims'])

    return ims, mus, np.array(nims) # list of (n,2) arcsec, list of (n,) signed mags, (S,)


#> unique points (closer than tol --> same point), keeping the first
def unique_pts(p, tol):
    uniq = []
    for q in p:
        if len(uniq) == 0 or np.min(np.hypot(*(np.array(uniq) - q).T)) >= tol: uniq.append(q)
    return np.array(uniq).reshape(-1, 2)


#> boolean mask of points in a that have no point of b within tol
def unmatched(a, b, tol):
    if len(a) == 0: return np.zeros(0, dtype=bool)
    if len(b) == 0: return np.ones(len(a), dtype=bool)
    return np.linalg.norm(a[:,None,:] - b[None,:,:], axis=-1).min(axis=1) > tol


""" #> CLASSIFYING ===================
================================== """

#> why did the test grid miss a true image? (checks the seeding stage directly)
def mechanism(theta, roots, src, xg, yg, bxg, byg):

    #> declarations
    Ny, Nx  = xg.shape
    arc_pix = xg[0,1] - xg[0,0]                            # arcsec/pix

    #> 1. outside the grid (no cell can seed it)
    fx = (theta[0] - xg[0,0]) / arc_pix                    # pix
    fy = (theta[1] - yg[0,0]) / arc_pix                    # pix
    if not (0 <= fx < Nx-1 and 0 <= fy < Ny-1): return 'outside grid'
    ix, iy = int(fx), int(fy)

    #> 2. the cell's corner box in the source plane does not contain the source (no seed)
    cx, cy = bxg[iy:iy+2, ix:ix+2], byg[iy:iy+2, ix:ix+2]  # arcsec
    if not (cx.min() <= src[0] <= cx.max() and cy.min() <= src[1] <= cy.max()): return 'corner box missed source'

    #> 3. another true image in the same cell (one seed for both)
    other = roots[np.hypot(*(roots - theta).T) > 0]
    ox = ((other[:,0] - xg[0,0]) / arc_pix).astype(int)
    oy = ((other[:,1] - yg[0,0]) / arc_pix).astype(int)
    if ((ox == ix) & (oy == iy)).any(): return 'two images in one cell'

    #> 4. it was seeded, but newton lost it (merged onto a neighbour or stalled)
    return 'seeded, lost in newton'


#> compares the test run to the truth for one source
def classify(src, low, low_mu, ref, ref_mu, grid, tol=1e-5):

    #> truth: every distinct root found by either run
    truth = unique_pts(np.vstack((ref, low)), tol)         # arcsec
    miss  = unmatched(truth, low, tol)                     # truth roots the test run missed
    rmiss = unmatched(truth, ref, tol)                     # truth roots the reference missed

    #> magnification of each truth root (from whichever run found it)
    pts = np.vstack((ref, low)); mus = np.concatenate((ref_mu, low_mu))
    mu  = mus[np.linalg.norm(truth[:,None,:] - pts[None,:,:], axis=-1).argmin(axis=1)]

    #> closest pair + largest |mu|
    sep = np.linalg.norm(truth[:,None,:] - truth[None,:,:], axis=-1)
    sep = sep[np.triu_indices(len(truth), k=1)].min() if len(truth) > 1 else np.nan # arcsec

    #> mechanisms for each missed root
    mech = [mechanism(t, truth, src, *grid) + (' (central)' if abs(m) < 0.1 else '') for t, m in zip(truth[miss], mu[miss])]

    return {'n_low': len(low), 'n_ref': len(ref), 'n_true': len(truth), 'n_miss': int(miss.sum()), 'n_rmiss': int(rmiss.sum()),
            'sep': sep, 'mu_max': np.abs(mu).max() if len(mu) else np.nan, 'mech': mech}


""" #> PRINTING ======================
================================== """

#> failure rate in bins of a quantity
def rate_table(name, vals, fail, edges, fmt):
    print(f'> failure rate vs {name}:')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi)
        if m.sum() == 0: continue
        print(f'>   {fmt(lo):>8s} - {fmt(hi):<8s}: {fail[m].sum():5d} / {m.sum():6d} = {100*fail[m].mean():6.2f}%')


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> declarations
    numGals   = 100                                         # galaxies
    numUni    = 500                                        # uniform sources inside the caustic per galaxy
    numNear   = 500                                        # near-caustic sources per galaxy
    nph       = 50                                         # test grid half-width [pix]
    pix_arc   = 60.0                                       # test grid [pix/arcsec]
    factor    = 4                                          # reference grid is factor x finer (same field of view)
    dmin      = 1e-5                                       # closest near-caustic source [arcsec]
    dmax      = 1e-2                                       # farthest near-caustic source [arcsec]
    rng       = np.random.default_rng(7)

    #> galaxies (same setup as bench_fastlens.py)
    galProf     = generate.galProfiles(nfw=True, hern=True, mult=[1,3,4], ex=True)
    paramRanges = params.toggleParams(galProf)
    bprofiles   = params.bprofiles(numGals, ranges=paramRanges, pix_arc=np.full(numGals, pix_arc), redshifts=np.tile((0.5, 1.0), (numGals, 1)))

    #> records
    recs, samp, srcs_all, delt = [], [], [], []
    srt = time.time()

    #>>># LENSING #<<<#
    for g, bp in enumerate(bprofiles):

        #> test grid + caustic
        grid = make_grid(bp, nph, pix_arc)
        caus = newtlens.caustic_points(grid[0], grid[1], bp, grid[4])
        if caus is None: print(f'> galaxy {g}: no caustic, skipped'); continue

        #> sources: uniform inside the caustic + just inside the caustic
        s_uni           = newtlens.sample_sources(caus, numUni, rng=rng)                  # arcsec
        s_near, d_near  = near_caustic_sources(caus, numNear, dmin, dmax, rng)            # arcsec
        srcs            = np.vstack((s_uni, s_near))                                     # arcsec

        #> test run + reference run (finer grid, same field of view)
        low, low_mu, _ = find(*grid[:4], srcs, bp)
        rgrid          = make_grid(bp, nph*factor, pix_arc*factor)
        ref, ref_mu, _ = find(*rgrid[:4], srcs, bp)

        #> classifying every source
        for k in range(len(srcs)):
            recs.append(classify(srcs[k], low[k], low_mu[k], ref[k], ref_mu[k], grid[:4]))
        samp     += ['uniform']*len(s_uni) + ['near caustic']*len(s_near)
        srcs_all += list(srcs)
        delt     += [np.nan]*len(s_uni) + list(d_near)
        print(f'> galaxy {g:3d}: {len(srcs)} sources, {sum(r["n_miss"] > 0 for r in recs[-len(srcs):])} with a missed image')

    dt = time.time() - srt


    #>>># RESULTS #<<<#

    #> arrays
    samp  = np.array(samp)
    delt  = np.array(delt)                                                        # arcsec
    fail  = np.array([r['n_miss'] > 0 for r in recs])
    rfail = np.array([r['n_rmiss'] > 0 for r in recs])
    sep   = np.array([r['sep'] for r in recs]) * pix_arc                          # pix (test grid)
    mumax = np.array([r['mu_max'] for r in recs])
    n_low = np.array([r['n_low'] for r in recs])
    n_tru = np.array([r['n_true'] for r in recs])

    #> overall
    print(f'\n> {len(recs)} sources in {dt:.1f} s')
    for s in ['uniform', 'near caustic']:
        m = samp == s
        print(f'> {s:12s}: test grid missed an image in {fail[m].sum():5d} / {m.sum():6d} = {100*fail[m].mean():6.3f}%   '
              f'(reference missed: {rfail[m].sum()})')

    #> mechanisms
    mechs = [mm for r in recs for mm in r['mech']]
    print(f'\n> why images were missed ({len(mechs)} images):')
    if mechs:
        names, counts = np.unique(mechs, return_counts=True)
        for nme, c in sorted(zip(names, counts), key=lambda z: -z[1]): print(f'>   {c:5d}  {nme}')

    #> parity: smooth, non-singular lenses always have an odd number of images
    print('\n> image counts (test grid vs truth):')
    for n in sorted(set(n_low) | set(n_tru)):
        print(f'>   {n} images: test {np.sum(n_low == n):6d}, truth {np.sum(n_tru == n):6d}')
    print(f'> even counts: test {np.sum(n_low % 2 == 0)}, truth {np.sum(n_tru % 2 == 0)}')

    #> selection bias in genQuadNewt (redraws every source that is not a quad)
    m = samp == 'uniform'
    print(f'\n> uniform sources that are true quads but not found as quads (would be redrawn): '
          f'{np.sum(m & (n_tru == 5) & (n_low != 5))} / {np.sum(m & (n_tru == 5))}')
    print(f'> uniform sources that are truly not quads (e.g. naked cusps, > 5 images): {np.sum(m & (n_tru != 5))} / {m.sum()}')

    #> failure rate vs closest pair, |mu| and distance to the caustic
    print()
    rate_table('closest pair [test-grid pix]', sep, fail, [0, 0.25, 0.5, 1, np.sqrt(2), 2, 3, 5, np.inf], lambda v: f'{v:.2f}')
    rate_table('largest |mu|', mumax, fail, [0, 10, 30, 100, 300, 1e3, 3e3, 1e4, np.inf], lambda v: f'{v:.0f}')
    m = samp == 'near caustic'
    rate_table('distance inside caustic [arcsec]', delt[m], fail[m], 10**np.arange(np.log10(dmin), np.log10(dmax)+0.5, 0.5), lambda v: f'{v:.0e}')

    #> worst examples
    print('\n> examples of missed images:')
    for k in np.flatnonzero(fail)[:10]:
        r = recs[k]
        print(f'>   {samp[k]:12s} src {np.round(srcs_all[k], 5)} [arcsec]: test {r["n_low"]}, truth {r["n_true"]}, '
              f'closest pair {r["sep"]*1e3:.2f} mas, |mu|max {r["mu_max"]:.0f} --> {", ".join(r["mech"])}')

    # end
# thank
