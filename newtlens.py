#> name: newtlens.py
#> author: John Miller Jr
#> descrp: newton-method lensing for opus: random sources, image finding (grid seeding + newton refinement), critical curves

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
from matplotlib.path import Path

#> modules
import profiles
import modules.error as error
from modules.units import u; u=u()

""" #> SOURCES =======================
================================== """

#> uniform random sources inside a polygon (vectorized rejection sampling)
def sample_sources(poly_pts, nsrc, rng=None, oversample=3):

    #> declarations
    if rng is None: rng = np.random
    
    #> deconstructing polygon points
    path = Path(poly_pts)
    (xmin, ymin) = poly_pts.min(axis=0)
    (xmax, ymax) = poly_pts.max(axis=0)
    
    #> drawing batches until enough sources are found
    out = np.empty(shape=(0, 2))
    while len(out) < nsrc:
        
        #> uniform oversampling
        k = max(oversample*(nsrc - len(out)), 16)
        pts = np.column_stack([rng.uniform(xmin, xmax, k), 
                               rng.uniform(ymin, ymax, k)])
        
        #> only add points that are inside paths
        mask = path.contains_points(pts)
        out = np.vstack([out, pts[mask]])
        
    #> only returns number of requested sources
    return out[:nsrc] # arcsec


""" #> IMAGE FINDING =================
================================== """

#> finds images for many sources of one galaxy at once
def find_images(xg, yg, bx_g, by_g, srcs, bprof, **kwargs):
    """
    xg, yg     [np.ndarray]: (Ny, Nx) grid [arcsec]
    bx_g, by_g [np.ndarray]: (Ny, Nx) source-plane map beta = theta - alpha [arcsec]
    srcs       [np.ndarray]: (S, 2) sources [arcsec]
    bprof      [dict]      : galaxy profile
    returns: dict w/ images (S, jw, 2) [arcsec], mu (S, jw) [signed], nims (S,), flag (S,) [nims > jmax]; jw = max(jmax, nims.max())
    """

    #> unpacking kwargs
    nit1      = kwargs.get('nit1', 20)         # stage-1 (grid) newton iterations
    nit       = kwargs.get('nit', 15)          # stage-2 (analytic) max newton iterations
    maxstep   = kwargs.get('maxstep', 3.0)     # max stage-2 newton step [pix]
    tol       = kwargs.get('tol', 1e-9)        # convergence on step size [arcsec]
    tol_res   = kwargs.get('tol_res', 1e-7)    # accepted lens-eq residual [arcsec]
    tol_merge = kwargs.get('tol_merge', 1e-5)  # duplicate roots closer than this are merged [arcsec]
    jmax      = kwargs.get('jmax', 5)          # expected max images per source (min array width)

    #> declarations
    nsrcs   = len(srcs)                        # number of sources (S)
    Ny, Nx  = xg.shape                         # grid size [pix]
    arc_pix = xg[0,1] - xg[0,0]                # arcsec/pix
    x0g     = xg[0,0]                          # grid lower-left corner [arcsec]
    y0g     = yg[0,0]                          # arcsec

    #>>># 0. SEEDING #<<<#
    
    #> steps
    #  1. backprojects and creates beta cells
    #  2. creates source box containing ALL sources in the source plane
    #  3. prune beta cells that do not overlap with the source box (i.e., do not contain a source)
    #  4. seed lens-plane cells that contain a source

    #> creates source-plane box of every grid cell
    def cell_minmax(b):
        c = np.stack([ b[:-1,:-1], b[1:,:-1], b[:-1,1:], b[1:,1:] ], axis=0) #> four corners of each cell
        return c.min(axis=0).ravel(), c.max(axis=0).ravel() #> min and max of each grid cell
    bxmin, bxmax = cell_minmax(bx_g)
    bymin, bymax = cell_minmax(by_g)

    #> box containing ALL sources in the source plane
    (sx0, sy0) = srcs.min(axis=0)
    (sx1, sy1) = srcs.max(axis=0)

    #> only keep beta cells that overlap with the source box
    #> must pass all four requirements
    cand = np.flatnonzero((bxmax >= sx0) & 
                          (bxmin <= sx1) & 
                          (bymax >= sy0) & 
                          (bymin <= sy1))
    bxmin, bxmax = bxmin[cand], bxmax[cand]
    bymin, bymax = bymin[cand], bymax[cand]

    #> source reformat, shape=(S,1)
    xs_ = srcs[:,0,None]
    ys_ = srcs[:,1,None]

    #> flag if a source falls within a beta box
    #> each row is a source
    flag = ((bxmin <= xs_) & 
            (xs_ <= bxmax) & 
            (bymin <= ys_) & 
            (ys_ <= bymax))

    #> one seed per flagged (source, beta cell) pair
    #> k       = beta cell index in candidate
    #> cand[k] = beta cell index in bx_g grid
    #> iy, ix  = beta cell row and column in by_g, bx_g
    sidx, k = np.nonzero(flag)          # source index, candidate index (per seed)
    iy, ix  = np.divmod(cand[k], Nx-1)  # cell row, column

    #> seeds start at cell centers, each paired w/ its source position
    x = x0g + (ix + 0.5)*arc_pix               # arcsec
    y = y0g + (iy + 0.5)*arc_pix               # arcsec
    bx, by = srcs[sidx,0], srcs[sidx,1]        # arcsec


    #>>># 1. NEWTON ON BILINEAR GRID #<<<#
    
    #> steps
    #  1. interpolate each seeds source-plane grid
    #  2. calculate newton step
    #  3. dampen step if needed (max 1 cell length)
    #  4. iterate nit1 times
    #  5. merge seeds of the same source if closer than a quarter of a cell 

    #> beta at the 4 corners of each seed's current cell
    def corners(b): return b[jy,jx], b[jy,jx+1], b[jy+1,jx], b[jy+1,jx+1]

    #> newton loop
    #> iterates through nit1 newtonian iterations for each seed
    for _ in range(nit1):

        #> seed positions
        fx = np.clip((x - x0g) / arc_pix, a_min=0, a_max=(Nx - 1 - 1e-9)) # pix
        fy = np.clip((y - y0g) / arc_pix, a_min=0, a_max=(Ny - 1 - 1e-9)) # pix

        #> lower-left node of each seed-containing cell
        jx = np.minimum(fx.astype(int), Nx-2)  # pix
        jy = np.minimum(fy.astype(int), Ny-2)  # pix
        
        #> seeds fractional position inside cell
        dx = fx - jx                           # [0,1)
        dy = fy - jy                           # [0,1)

        #> beta_x and beta_y at the cell corners
        bx00, bx01, bx10, bx11 = corners(bx_g)     # arcsec
        by00, by01, by10, by11 = corners(by_g)     # arcsec

        #> bilinear interpolation of beta at the seeds
        bxi = bx00*(1-dx)*(1-dy) + bx01*dx*(1-dy) + bx10*(1-dx)*dy + bx11*dx*dy # arcsec
        byi = by00*(1-dx)*(1-dy) + by01*dx*(1-dy) + by10*(1-dx)*dy + by11*dx*dy # arcsec

        #> jacobian of the bilinear map, A = [beta_x=[a, b], beta_y=[c, d]]
        a = ( (bx01-bx00)*(1-dy) + (bx11-bx10)*dy ) / arc_pix # dimensionless
        b = ( (bx10-bx00)*(1-dx) + (bx11-bx01)*dx ) / arc_pix
        c = ( (by01-by00)*(1-dy) + (by11-by10)*dy ) / arc_pix
        d = ( (by10-by00)*(1-dx) + (by11-by01)*dx ) / arc_pix

        #> lens-equation residual, f = beta(theta) - beta_s
        rx = bxi - bx                          # arcsec
        ry = byi - by                          # arcsec

        #> determinant (guarded against singular cells)
        det = a*d - b*c
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)

        #> newton step, s = A^-1 f
        sx = ( d*rx - b*ry) / det              # arcsec
        sy = (-c*rx + a*ry) / det              # arcsec
        
        ##> stopped here

        #> damping: step length capped at 1 pixel
        lam = np.minimum(1.0, arc_pix / np.maximum(np.hypot(sx, sy), 1e-300))

        #> updating seeds, theta <-- theta - lam*s
        x -= lam*sx                            # arcsec
        y -= lam*sy                            # arcsec

    #> merging seeds that landed on the same root (within a quarter pixel)
    sidx, x, y = merge(sidx, x, y, 0.25*arc_pix)
    bx = srcs[sidx,0]                          # arcsec
    by = srcs[sidx,1]                          # arcsec


    #>>># 2. NEWTON ON ANALYTIC DEFLECTIONS #<<<#

    #> steps
    #  1. calculate initial deflect + jacobian of each seed analytically
    #  2. calculate newton step
    #  3. dampen step if needed (max 3 cell lengths)
    #  4. backtrack (i.e., halve step) until decrease found
    #  5. iterate until converged, stalled, or no decrease found

    #> deflections + jacobian at each seed position (x,y)
    ax, ay, axx, axy, ayx, ayy = profiles.alpha_jac_points(x, y, bprof)

    #> initializing residuals (f_x = theta_x - alpha_x - beta_x)
    rx = x - ax - bx 
    ry = y - ay - by 
    rn = np.hypot(rx, ry)

    #> active list
    active = np.ones(len(x), dtype=bool)

    #> newton loop
    #> iterates nit times only 'active' seeds exist
    for _ in range(nit):
        
        #> if converged already
        if not active.any(): break
        ia = np.flatnonzero(active) # indices of active roots

        #> lensing jacobian
        a = 1-axx[ia]            # 1 - da_x/d_x
        b =  -axy[ia]            # 0 - da_x/d_y
        c =  -ayx[ia]            # 0 - da_y/d_x
        d = 1-ayy[ia]            # 1 - da_y/d_y
        det = a*d - b*c

        #> newton step, s = A^-1 f
        sx = ( d*rx[ia] - b*ry[ia]) / det                    # arcsec
        sy = (-c*rx[ia] + a*ry[ia]) / det                    # arcsec
        step = np.hypot(sx, sy)

        #> dampening step size, length capped at maxstep (3) pixels
        lam = np.minimum(1.0, maxstep*arc_pix / np.maximum(step, 1e-300))

        #> backtracking, halving the step until the residual decreases (max 8 halvings)
        todo = np.arange(len(ia))
        for _ in range(8):

            #> trial positions, theta - lam*s
            k = ia[todo]                                     # root indices
            xt = x[k] - lam[todo]*sx[todo]                   # arcsec
            yt = y[k] - lam[todo]*sy[todo]                   # arcsec

            #> trial residuals
            o = profiles.alpha_jac_points(xt, yt, bprof)              # (ax, ay, axx, axy, ayx, ayy)
            rxt = xt - o[0] - bx[k]                          # arcsec
            ryt = yt - o[1] - by[k]                          # arcsec
            rnt = np.hypot(rxt, ryt) 

            #> accepting trials w/ a sufficient decrease in |f|
            ok = rnt < rn[k] * (1 - 1e-4*lam[todo]) + 1e-12
            kk = k[ok]

            #> updating accepted roots (positions, residuals, jacobian)
            x[kk], y[kk], rx[kk], ry[kk], rn[kk] = xt[ok], yt[ok], rxt[ok], ryt[ok], rnt[ok]
            axx[kk], axy[kk], ayx[kk], ayy[kk] = o[2][ok], o[3][ok], o[4][ok], o[5][ok]

            #> updating seeds 
            todo = todo[~ok]                  # removing accepted step
            lam[todo] *= 0.5                  # halving step size to try next
            if len(todo) == 0: break          # if all have been accepted

        #> deactivating seeds
        active[ia[(lam*step < tol)]] = False  # converged
        active[ia[lam < 1e-2]] = False        # stalled (step size is tiny)
        active[ia[todo]] = False              # no decrease found (still trying to half)


    #>>># KEEPING ROOTS #<<<#

    #> keeping roots that solve the lens equation
    good = rn < tol_res
    
    #> saving info
    detA = (1-axx)*(1-ayy) - axy*ayx
    sidx, x, y, detA = sidx[good], x[good], y[good], detA[good]

    #> merging duplicate roots
    sidx, x, y, detA = merge(sidx, x, y, tol_merge, detA)


    #>>># PACKING PER SOURCE (NAN PADDED) #<<<#

    #> number of images per source
    nims = np.bincount(sidx, minlength=nsrcs)

    #> flagging sources w/ more than jmax images (rare, but possible for complicated lenses)
    flag = nims > jmax
    if flag.any(): error.highlight(f'{flag.sum()} source(s) have more than {jmax} images (max = {nims.max()})! sidx = {np.flatnonzero(flag)}')

    #> array width, grown so no image is dropped
    jw = max(jmax, nims.max(initial=0))                      # images per source

    #> grouping roots by source
    order = np.argsort(sidx, kind='stable')
    sidx, x, y, detA = sidx[order], x[order], y[order], detA[order]

    #> slot of each root within its source (0, 1, 2, ...)
    slot = np.arange(len(sidx)) - np.repeat(np.cumsum(nims) - nims, nims)

    #> filling (S, jw) arrays
    ims = np.full((nsrcs, jw, 2), np.nan)                    # arcsec
    mu  = np.full((nsrcs, jw), np.nan)                       # signed mags
    ims[sidx, slot, 0] = x
    ims[sidx, slot, 1] = y
    mu[sidx, slot]     = 1/detA

    return {'images': ims, 'mu': mu, 'nims': nims, 'flag': flag} # arcsec, signed mags, counts, bool


#> pads (S, j, ...) arrays w/ nans along axis=1 to width w
def pad(a, w): return np.concatenate([a, np.full((a.shape[0], w - a.shape[1], *a.shape[2:]), np.nan)], axis=1)


#> merges points of the same source closer than tol
def merge(sidx, x, y, tol, *extra):

    #> sorting by source, then x, then y
    order = np.lexsort((y, x, sidx))
    sidx, x, y = sidx[order], x[order], y[order]
    extra = [e[order] for e in extra] # extra arrays follow the same order

    #> flagging later points within tol of an earlier kept point (up to 15 neighbours back)
    dup = np.zeros(len(x), dtype=bool)
    for lag in range(1, 16):
        
        #> if end is reached
        if lag >= len(x): break

        #> point i vs point i-lag: same source + closer than tol
        same = (sidx[lag:] == sidx[:-lag]) & (np.hypot(x[lag:]-x[:-lag], y[lag:]-y[:-lag]) < tol)

        #> duplicate only if the earlier point is itself kept
        dup[lag:] |= same & ~dup[:-lag]

    #> keeping non-duplicates
    keep = ~dup
    return (sidx[keep], x[keep], y[keep], *[e[keep] for e in extra])


""" #> CRITICAL CURVES ===============
================================== """

#> returns tangential critical curve points
def caustic_points(xg, yg, bprof, lamt):
    
    #> imports
    from skimage import measure
    
    #> declarations
    arc_pix = xg[0,1] - xg[0,0]
    
    #> finding lamt = 0
    cs = measure.find_contours(lamt, 0.0)
    
    #> if no critical curve
    if not cs: return None
    cc = max(cs, key=len)
    
    #> tangential critical curve [arcsec]
    tx = xg[0,0] + (cc[:,1] * arc_pix)
    ty = yg[0,0] + (cc[:,0] * arc_pix)
    
    #> deflection angles at tcc
    ax, ay = profiles.alpha_points(tx, ty, bprof)
    
    #> back-projecting
    return np.column_stack((tx - ax, ty - ay)) # arcsec


