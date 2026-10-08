#> name: fastlens.py
#> author: John Miller Jr
#> descrp: vectorized image finding for opus: grid seeding + newton refinement on analytic deflections

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
from matplotlib.path import Path

#> modules
from modules.units import u; u=u()


""" #> CSE COEFFICIENTS ==============
================================== """

#> NFW CSE coefficients (Ai, Si) (16; lower accuracy); 10^-2 in kappa
NFW_COEFFS = np.array([
    [1.434960e-16, 4.041628e-06], [5.232413e-14, 3.086267e-05],
    [2.666660e-12, 1.298542e-04], [7.961761e-11, 4.131977e-04],
    [2.306895e-09, 1.271373e-03], [6.742968e-08, 3.912641e-03],
    [1.991691e-06, 1.208331e-02], [5.904388e-05, 3.740521e-02],
    [1.693069e-03, 1.153247e-01], [4.039850e-02, 3.472038e-01],
    [5.665072e-01, 1.017550e+00], [3.683242e+00, 3.253031e+00],
    [1.582481e+01, 1.190315e+01], [6.340984e+01, 4.627701e+01],
    [2.576763e+02, 1.842613e+02], [1.422619e+03, 8.206569e+02]
    ], dtype=np.float32).astype(np.float64)

#> hernquist CSE coefficients (Ai, Si) (13; lower accuracy); 10^-2 in kappa
HERN_COEFFS = np.array([
    [7.775712e-16, 4.426947e-06], [3.279878e-13, 3.551219e-05],
    [2.374931e-11, 1.639271e-04], [1.137151e-09, 6.047024e-04],
    [5.314908e-08, 2.180958e-03], [2.466940e-06, 7.843573e-03],
    [1.125917e-04, 2.809420e-02], [4.700637e-03, 9.893403e-02],
    [1.257748e-01, 3.246017e-01], [9.744152e-01, 9.409923e-01],
    [1.434502e+00, 2.929948e+00], [5.548243e-01, 1.545137e+01],
    [6.123431e-01, 1.883671e+03]
    ], dtype=np.float32).astype(np.float64)


""" #> DEFLECTION ANGLES =============
================================== """

#> CSE deflection angles at points (x,y)
def cse_alpha(x, y, prof, coeffs, norm_key, angDist, sigCrit):

    #> unpacking kwargs
    arc_rad = u.arc_rad                             # arcsec/rad
    x0      = prof.get('x0', 0.0)                   # arcsec
    y0      = prof.get('y0', 0.0)                   # arcsec
    q       = prof.get('axisrat', 1.0)              # []
    r0      = prof.get('radius', 1.0)               # kpc
    rho0    = prof.get(norm_key, 0.0)               # solMass/kpc^3
    th      = prof.get('theta', 0.0) * np.pi/180    # deg --> rad
    
    #> declarations
    ct = np.cos(th) 
    st = np.sin(th) 
    q2 = q*q

    #> shifting, rotating, and scaling points (arcsec --> units of r0')
    denom = (r0 / np.sqrt(q) / angDist) * arc_rad    # arcsec
    Xr    = (  (x - x0)*ct + (y - y0)*st ) / denom
    Yr    = ( -(x - x0)*st + (y - y0)*ct ) / denom
    
    #> more declarations (fight me)
    shp = Xr.shape
    X2 = Xr*Xr
    Y2 = Yr*Yr
    
    #> flattening and reshaping
    Xr = Xr.ravel()[None]                 # shape=(1,N)
    Yr = Yr.ravel()[None]                 # shape=(1,N)
    X2 = X2.ravel()[None]                 # shape=(1,N)
    Y2 = Y2.ravel()[None]                 # shape=(1,N)

    #> getting coefficents
    Ai = coeffs[:,0,None]                 # shape=(Nc,1)
    Si = coeffs[:,1,None]                 # shape=(Nc,1)
    
    #>>># DEFLECT #<<<#
    
    #> separating equations
    lowp = np.sqrt(q2*(Si*Si + X2) + Y2)  # eq 18, shape=(Nc,N)
    upp  = (lowp + Si)**2 + (1 - q2)*X2   # eq 17
    w    = Ai*q / (Si * lowp * upp)       # part of eq 19/20
    
    #> calculating deflection angles! (and reshapes)
    gx   = ( w*(lowp + q2*Si)).sum(0).reshape(shp) * Xr.reshape(shp)  # eq 19
    gy   = ( w*(lowp + Si)   ).sum(0).reshape(shp) * Yr.reshape(shp)  # eq 20

    #> scaling (--> arcsec)
    scale = (r0/np.sqrt(q)) * (r0*rho0) / sigCrit / angDist * arc_rad # eq 24
    gx *= scale            # arcsec
    gy *= scale            # arcsec
    
    #> rotating back
    gx_ = gx*ct - gy*st    # arcsec
    gy_ = gx*st + gy*ct    # arcsec

    return gx_, gy_


#> CSE deflection angles + jacobian at points (x,y) (analytic second derivatives)
def cse_alpha_jac(x, y, prof, coeffs, norm_key, angDist, sigCrit):

    #> unpacking kwargs
    arc_rad = u.arc_rad                             # arcsec/rad
    x0      = prof.get('x0', 0.0)                   # arcsec
    y0      = prof.get('y0', 0.0)                   # arcsec
    q       = prof.get('axisrat', 1.0)              # []
    r0      = prof.get('radius', 1.0)               # kpc
    rho0    = prof.get(norm_key, 0.0)               # solMass/kpc^3
    th      = prof.get('theta', 0.0) * np.pi/180    # deg --> rad
    
    #> declarations
    ct = np.cos(th)
    st = np.sin(th)
    q2 = q*q

    #> shifting, rotating, and scaling points (arcsec --> units of r0')
    denom = (r0 / np.sqrt(q) / angDist) * arc_rad    # arcsec
    Xr    = (  (x - x0)*ct + (y - y0)*st ) / denom
    Yr    = ( -(x - x0)*st + (y - y0)*ct ) / denom
    
    #> more declarations
    shp = Xr.shape
    X2  = Xr*Xr
    Y2  = Yr*Yr
    
    #> flattening and reshaping
    Xr = Xr.ravel()[None]                   # shape=(1,N)
    Yr = Yr.ravel()[None]                   # shape=(1,N)
    X2 = X2.ravel()[None]                   # shape=(1,N)
    Y2 = Y2.ravel()[None]                   # shape=(1,N)

    #> getting coefficents
    Ai = coeffs[:,0,None]                   # shape=(Nc,1)
    Si = coeffs[:,1,None]                   # shape=(Nc,1)
    
    #>>># DEFLECT + JACOBIAN #<<<#
    
    #> separating equations
    lowp   = np.sqrt(q2*(Si*Si + X2) + Y2)  # eq 18, shape=(Nc,N)
    lowps  = lowp + Si
    upp    = lowps*lowps + (1 - q2)*X2      # eq 19
    w      = Ai*q / (Si * lowp * upp)       # part of eq 19/20
    
    #> calculating lowp derivatives
    ilowp  = 1/lowp                         # 1/lowp
    lowp_x = q2*Xr*ilowp                    # d(lowp)/dX
    lowp_y = Yr*ilowp                       # d(lowp)/dY
    
    #> calculating upp derivatives
    upp_x  = 2*lowps*lowp_x + 2*(1 - q2)*Xr # d(upp)/dX
    upp_y  = 2*lowps*lowp_y                 # d(upp)/dY
    lnD_x  = (lowp_x*ilowp) + (upp_x/upp)   # d(ln D)/dX, D = lowp*upp
    lnD_y  = (lowp_y*ilowp) + (upp_y/upp)   # d(ln D)/dY
    
    #> numerators (saving for alpha derivatives)
    ux     = lowp + q2*Si                   # x numerator
    uy     = lowps                          # y numerator
    
    #> calculating deflection angles!
    gx = w * Xr * ux                        # shape=(Nc,N)
    gy = w * Yr * uy                        # ...
    
    #> calculating alpha derivatives
    gxx = ( w*(ux + Xr*lowp_x) - gx*lnD_x ).sum(0)  # d(gx)/dX
    gxy = ( w * Xr * lowp_y    - gx*lnD_y ).sum(0)  # d(gx)/dY
    gyy = ( w*(uy + Yr*lowp_y) - gy*lnD_y ).sum(0)  # d(gy)/dY
    
    #> back to original shape
    gx  = gx.sum(0).reshape(shp)
    gy  = gy.sum(0).reshape(shp)
    gxx = gxx.reshape(shp)
    gxy = gxy.reshape(shp)
    gyy = gyy.reshape(shp)
    
    #>>># DEFLECT #<<<#
    
    #> scaling (--> arcsec)
    scale = (r0/np.sqrt(q)) * (r0*rho0) / sigCrit / angDist * arc_rad
    gx *= scale            # arcsec
    gy *= scale            # arcsec
    
    #> rotating deflection angles back
    gx_ = gx*ct - gy*st    # arcsec
    gy_ = gx*st + gy*ct    # arcsec
    
    #>>># JACOBIAN #<<<#
    
    #> scaling jacobian (d/dX --> d/dx)
    jfac = scale / denom
    gxx *= jfac
    gxy *= jfac
    gyy *= jfac
    
    #> rotating jacobian back: R J R^T
    c2   = ct*ct
    s2   = st*st
    cs   = ct*st
    gxx_ = gxx*c2 - 2*gxy*cs + gyy*s2
    gyy_ = gxx*s2 + 2*gxy*cs + gyy*c2
    gxy_ = (gxx - gyy)*cs + gxy*(c2 - s2)
    
    return gx_, gy_, gxx_, gxy_, gxy_, gyy_   # arcsec, []


#> deflection + jacobian at points (analytic for CSE & shear, central differences for multipoles)
def alpha_jac_points(x, y, bprof, h=1e-6):
    
    #> declarations
    angDist, sigCrit = bprof['angDist'], bprof['sigCrit']   # kpc, solMass/kpc^2
    out = [np.zeros_like(x, dtype=np.float64) for _ in range(6)]
    
    #> nfw + hernquist
    for key, coeffs, nkey in [('nfw', NFW_COEFFS, 'rho0'), ('hern', HERN_COEFFS, 'norm')]:
        for prof in bprof.get(key, []):
            for o, v in zip(out, cse_alpha_jac(x, y, prof, coeffs, nkey, angDist, sigCrit)): o += v
    
    #> multipoles (analytic; psi = -(a_m/m) r^slope cos[m(phi - phi_m)], skipping norm=0)
    for prof in bprof.get('mult', []):
        norm = prof.get('norm', 0.0)
        if norm == 0: continue
        m, gam = prof.get('m', 1), prof.get('slope', 2)
        pa = prof.get('theta', 0.0) * np.pi/180                     # rad
        Xr = x - prof.get('x0', 0.0); Yr = y - prof.get('y0', 0.0)  # arcsec
        r2 = Xr*Xr + Yr*Yr; r = np.sqrt(r2)                         # arcsec
        phi = np.arctan2(Yr, Xr); cm, sm = np.cos(m*(phi - pa)), np.sin(m*(phi - pa))
        c, s_ = np.where(r > 0, Xr/np.maximum(r, 1e-300), 1.0), np.where(r > 0, Yr/np.maximum(r, 1e-300), 0.0)
        k = -norm/m                                                 # prefactor
        rg2 = np.where(r > 0, r**(gam - 2), 0.0)                    # r^(slope-2)
        f_r_r = gam*rg2*cm; f_rr = gam*(gam - 1)*rg2*cm             # f_r / r, f_rr
        f_p_r2 = -m*rg2*sm; f_pp_r2 = -m*m*rg2*cm; f_rp_r = -m*gam*rg2*sm # f_phi/r^2, f_phiphi/r^2, f_rphi/r
        out[0] += k*rg2*(gam*Xr*cm + m*Yr*sm)                       # arcsec
        out[1] += k*rg2*(gam*Yr*cm - m*Xr*sm)                       # arcsec
        fxx = f_rr*c*c + (f_r_r + f_pp_r2)*s_*s_ - 2*(f_rp_r - f_p_r2)*s_*c
        fyy = f_rr*s_*s_ + (f_r_r + f_pp_r2)*c*c + 2*(f_rp_r - f_p_r2)*s_*c
        fxy = (f_rr - f_r_r - f_pp_r2)*s_*c + (f_rp_r - f_p_r2)*(c*c - s_*s_)
        out[2] += k*fxx; out[3] += k*fxy; out[4] += k*fxy; out[5] += k*fyy
    
    #> external shear (analytic, constant jacobian)
    for prof in bprof.get('ex', []):
        g = prof.get('norm', 0.0)
        if g == 0: continue
        tg = 2*prof.get('theta', 0.0)*np.pi/180                     # rad
        c2, s2 = np.cos(tg), np.sin(tg)
        out[0] += -g*(x*c2 + y*s2); out[1] += -g*(x*s2 - y*c2)      # arcsec
        out[2] += -g*c2; out[3] += -g*s2; out[4] += -g*s2; out[5] += g*c2
    
    return out # alpha_x, alpha_y [arcsec], axx, axy, ayx, ayy []


#> deflection angles at arbitrary points
def alpha_points(x, y, bprof):
    """
    x, y  [np.ndarray]: positions [arcsec], any (matching) shape
    bprof [dict]      : one galaxy from params.bprofiles
    returns: alpha_x, alpha_y [arcsec]
    """

    #> declarations
    angDist = bprof['angDist']                             # kpc
    sigCrit = bprof['sigCrit']                             # solMass/kpc^2
    ax = np.zeros_like(x, dtype=np.float64)
    ay = np.zeros_like(y, dtype=np.float64)

    #> nfw
    for prof in bprof.get('nfw', []):
        gx, gy = cse_alpha(x, y, prof, NFW_COEFFS, 'rho0', angDist, sigCrit)
        ax += gx; ay += gy

    #> hernquist
    for prof in bprof.get('hern', []):
        gx, gy = cse_alpha(x, y, prof, HERN_COEFFS, 'norm', angDist, sigCrit)
        ax += gx; ay += gy

    #> multipoles (skipping norm=0)
    for prof in bprof.get('mult', []):
        norm = prof.get('norm', 0.0)
        if norm == 0: continue
        m, slope = prof.get('m', 1), prof.get('slope', 2)
        pa = prof.get('theta', 0.0) * np.pi/180            # rad
        Xr = x - prof.get('x0', 0.0); Yr = y - prof.get('y0', 0.0) # arcsec
        r  = np.sqrt(Xr**2 + Yr**2)                        # arcsec
        phi = m*(np.arctan2(Yr, Xr) - pa)
        C  = -norm/m * r**(slope-2)
        ax += C * (Xr*slope*np.cos(phi) + Yr*m*np.sin(phi))  # arcsec
        ay += C * (Yr*slope*np.cos(phi) - Xr*m*np.sin(phi))  # arcsec

    #> external shear (skipping norm=0); cartesian form
    for prof in bprof.get('ex', []):
        g = prof.get('norm', 0.0)
        if g == 0: continue
        tg = 2*prof.get('theta', 0.0)*np.pi/180            # rad
        c2, s2 = np.cos(tg), np.sin(tg)
        ax += -g*(x*c2 + y*s2)                             # arcsec
        ay += -g*(x*s2 - y*c2)                             # arcsec

    return ax, ay # arcsec


""" #> POINT POTENTIAL ===============
================================== """

#> lensing potential at points (same conventions as lpotCPU) [rad^2]
def psi_points(x, y, bprof):

    #> declarations
    arc_rad = u.arc_rad
    angDist, sigCrit = bprof['angDist'], bprof['sigCrit']
    psi = np.zeros_like(x, dtype=np.float64)

    #> nfw + hernquist (CSE)
    for key, coeffs, nkey in [('nfw', NFW_COEFFS, 'rho0'), ('hern', HERN_COEFFS, 'norm')]:
        for prof in bprof.get(key, []):
            q  = prof.get('axisrat', 1.0); q2 = q*q
            r0 = prof.get('radius', 1.0); rho0 = prof.get(nkey, 0.0)
            th = prof.get('theta', 0.0)*np.pi/180; ct, st = np.cos(th), np.sin(th)
            denom = (r0/np.sqrt(q)/angDist) * arc_rad                       # arcsec
            xs_ = x - prof.get('x0', 0.0); ys_ = y - prof.get('y0', 0.0)
            Xr = ( xs_*ct + ys_*st) / denom; Yr = (-xs_*st + ys_*ct) / denom
            X2 = Xr*Xr
            lp = np.zeros_like(Xr)
            for Ai, Si in coeffs:
                sp = np.sqrt(q2*(Si*Si + X2) + Yr*Yr)
                lP = (sp + Si)**2 + X2*(1 - q2)
                lp += Ai * ((q/Si/2)*np.log(lP) - (q/Si)*np.log((1+q)*Si))
            psi += lp * (r0*r0/q) * (r0*rho0) / sigCrit / (angDist*angDist)  # rad^2

    #> multipoles
    for prof in bprof.get('mult', []):
        norm = prof.get('norm', 0.0)
        if norm == 0: continue
        m, slope = prof.get('m', 1), prof.get('slope', 2)
        pa = prof.get('theta', 0.0)*np.pi/180
        Xr = x - prof.get('x0', 0.0); Yr = y - prof.get('y0', 0.0)
        r = np.sqrt(Xr**2 + Yr**2)
        psi += -norm/m * r**slope * np.cos(m*(np.arctan2(Yr, Xr) - pa)) / arc_rad**2

    #> external shear
    for prof in bprof.get('ex', []):
        g = prof.get('norm', 0.0)
        if g == 0: continue
        tg = 2*prof.get('theta', 0.0)*np.pi/180
        xr, yr = x/arc_rad, y/arc_rad
        psi += -g/2 * ((xr*xr - yr*yr)*np.cos(tg) + 2*xr*yr*np.sin(tg))

    return psi # rad^2


""" #> SOURCES =======================
================================== """

#> uniform random sources inside a polygon (vectorized rejection sampling)
def sample_sources(poly_pts, n, rng=None, oversample=3):
    """
    poly_pts [np.ndarray]: (N, 2) polygon vertices [arcsec] (e.g., caustic)
    n        [int]       : number of sources
    returns: (n, 2) sources [arcsec]
    """

    #> declarations
    if rng is None: rng = np.random
    path = Path(poly_pts)
    (xmin, ymin), (xmax, ymax) = poly_pts.min(0), poly_pts.max(0)
    out = np.empty((0, 2))

    #> drawing in batches until enough are inside
    while len(out) < n:
        k = max(oversample*(n - len(out)), 16)
        pts = np.column_stack((rng.uniform(xmin, xmax, k), rng.uniform(ymin, ymax, k))) # arcsec
        out = np.vstack((out, pts[path.contains_points(pts)]))

    return out[:n] # arcsec


""" #> IMAGE FINDING =================
================================== """

#> finds images for many sources of one galaxy at once
def find_images(xg, yg, bx_g, by_g, srcs, bprof, **kwargs):
    """
    xg, yg     [np.ndarray]: (Ny, Nx) grid [arcsec]
    bx_g, by_g [np.ndarray]: (Ny, Nx) source-plane map beta = theta - alpha [arcsec]
    srcs       [np.ndarray]: (S, 2) sources [arcsec]
    bprof      [dict]      : galaxy profile
    returns: dict w/ images (S, jmax, 2) [arcsec], mu (S, jmax) [signed], nims (S,)
    """
    
    #> kwargs
    nit1      = kwargs.get('nit1', 20)         # stage-1 (grid) newton iterations
    nit       = kwargs.get('nit', 15)          # stage-2 (analytic) max newton iterations
    maxstep   = kwargs.get('maxstep', 3.0)     # max stage-2 newton step [pix]
    tol       = kwargs.get('tol', 1e-9)        # convergence on step size [arcsec]
    tol_res   = kwargs.get('tol_res', 1e-7)    # accepted lens-eq residual [arcsec]
    tol_merge = kwargs.get('tol_merge', 1e-5)  # duplicate roots closer than this are merged [arcsec]
    jmax      = kwargs.get('jmax', 5)          # images kept per source
    
    #> declarations
    S      = len(srcs)
    Ny, Nx = xg.shape
    pix    = xg[0,1] - xg[0,0]                               # arcsec/pix
    x0g, y0g = xg[0,0], yg[0,0]                              # arcsec
    
    #>>># SEEDING #<<<#
    
    #> per-cell min/max of the source-plane map (once per galaxy)
    def cell_minmax(b):
        c = np.stack([b[:-1,:-1], b[1:,:-1], b[:-1,1:], b[1:,1:]])
        return c.min(0).ravel(), c.max(0).ravel()
    bxmin, bxmax = cell_minmax(bx_g)
    bymin, bymax = cell_minmax(by_g)
    
    #> keeping only cells whose beta range overlaps the sources' bounding box
    (sx0, sy0), (sx1, sy1) = srcs.min(0), srcs.max(0)        # arcsec
    cand = np.flatnonzero((bxmax >= sx0) & (bxmin <= sx1) & (bymax >= sy0) & (bymin <= sy1))
    bxmin, bxmax, bymin, bymax = bxmin[cand], bxmax[cand], bymin[cand], bymax[cand]
    
    #> a cell can hold an image of source s if s lies inside the cell's beta range
    xs_, ys_ = srcs[:,0,None], srcs[:,1,None]                # arcsec
    flag = (bxmin <= xs_) & (xs_ <= bxmax) & (bymin <= ys_) & (ys_ <= bymax) # (S, cand cells)
    sidx, k = np.nonzero(flag)
    iy, ix = np.divmod(cand[k], Nx-1)
    x = x0g + (ix + 0.5)*pix                                 # arcsec
    y = y0g + (iy + 0.5)*pix                                 # arcsec
    bx, by = srcs[sidx,0], srcs[sidx,1]                      # arcsec
    
    #>>># STAGE 1: NEWTON ON THE BILINEAR GRID MAP (cheap) #<<<#
    
    #> grid map derivatives (per pixel --> per arcsec)
    for _ in range(nit1):
        fx = np.clip((x - x0g)/pix, 0, Nx-1-1e-9); fy = np.clip((y - y0g)/pix, 0, Ny-1-1e-9)
        jx = np.minimum(fx.astype(int), Nx-2); jy = np.minimum(fy.astype(int), Ny-2)
        uu, vv = fx - jx, fy - jy
        def corners(b): return b[jy,jx], b[jy,jx+1], b[jy+1,jx], b[jy+1,jx+1]
        b00, b01, b10, b11 = corners(bx_g); c00, c01, c10, c11 = corners(by_g)
        bxi = b00*(1-uu)*(1-vv) + b01*uu*(1-vv) + b10*(1-uu)*vv + b11*uu*vv
        byi = c00*(1-uu)*(1-vv) + c01*uu*(1-vv) + c10*(1-uu)*vv + c11*uu*vv
        a = ((b01-b00)*(1-vv) + (b11-b10)*vv)/pix; b = ((b10-b00)*(1-uu) + (b11-b01)*uu)/pix
        c = ((c01-c00)*(1-vv) + (c11-c10)*vv)/pix; d = ((c10-c00)*(1-uu) + (c11-c01)*uu)/pix
        rx, ry = bxi - bx, byi - by                          # arcsec
        det = a*d - b*c
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        sx = ( d*rx - b*ry)/det; sy = (-c*rx + a*ry)/det     # arcsec
        lam = np.minimum(1.0, pix/np.maximum(np.hypot(sx, sy), 1e-300))
        x, y = x - lam*sx, y - lam*sy
    
    #> merging stage-1 duplicates within a quarter pixel
    sidx, x, y = _merge(sidx, x, y, 0.25*pix)
    bx, by = srcs[sidx,0], srcs[sidx,1]                      # arcsec
    
    #>>># STAGE 2: NEWTON ON ANALYTIC DEFLECTIONS (w/ backtracking) #<<<#
    
    #> initial residuals
    ax, ay, axx, axy, ayx, ayy = alpha_jac_points(x, y, bprof)
    rx, ry = x - ax - bx, y - ay - by                        # arcsec
    rn = np.hypot(rx, ry)                                    # arcsec
    active = np.ones(len(x), dtype=bool)
    for _ in range(nit):
        if not active.any(): break
        ia = np.flatnonzero(active)
        a, b, c, d = 1-axx[ia], -axy[ia], -ayx[ia], 1-ayy[ia] # lensing jacobian
        det = a*d - b*c
        sx = ( d*rx[ia] - b*ry[ia]) / det                    # arcsec
        sy = (-c*rx[ia] + a*ry[ia]) / det                    # arcsec
        step = np.hypot(sx, sy)                              # arcsec
        lam = np.minimum(1.0, maxstep*pix / np.maximum(step, 1e-300)) # limiting step size
        
        #> backtracking: halving the step until the residual decreases
        todo = np.arange(len(ia))
        for _ in range(8):
            k = ia[todo]
            xt = x[k] - lam[todo]*sx[todo]; yt = y[k] - lam[todo]*sy[todo]
            o = alpha_jac_points(xt, yt, bprof)
            rxt = xt - o[0] - bx[k]; ryt = yt - o[1] - by[k]
            rnt = np.hypot(rxt, ryt)
            ok = rnt < rn[k] * (1 - 1e-4*lam[todo]) + 1e-12
            kk = k[ok]
            x[kk], y[kk], rx[kk], ry[kk], rn[kk] = xt[ok], yt[ok], rxt[ok], ryt[ok], rnt[ok]
            axx[kk], axy[kk], ayx[kk], ayy[kk] = o[2][ok], o[3][ok], o[4][ok], o[5][ok]
            todo = todo[~ok]; lam[todo] *= 0.5
            if len(todo) == 0: break
        
        #> converged (tiny step) or stalled (no/negligible decrease found)
        active[ia[(lam*step < tol)]] = False
        active[ia[lam < 1e-2]] = False
        active[ia[todo]] = False
    
    #> keeping converged roots + magnifications from the jacobian
    detA = (1-axx)*(1-ayy) - axy*ayx
    good = rn < tol_res
    sidx, x, y, detA = sidx[good], x[good], y[good], detA[good]
    sidx, x, y, detA = _merge(sidx, x, y, tol_merge, detA)
    
    #> packing per source (nan padded)
    nims = np.bincount(sidx, minlength=S)
    order = np.argsort(sidx, kind='stable')
    sidx, x, y, detA = sidx[order], x[order], y[order], detA[order]
    slot = np.arange(len(sidx)) - np.repeat(np.cumsum(nims) - nims, nims)
    keep = slot < jmax
    ims = np.full((S, jmax, 2), np.nan); mu = np.full((S, jmax), np.nan)
    ims[sidx[keep], slot[keep], 0] = x[keep]
    ims[sidx[keep], slot[keep], 1] = y[keep]
    mu[sidx[keep], slot[keep]] = 1/detA[keep]
    
    return {'images': ims, 'mu': mu, 'nims': nims} # arcsec, signed mags, counts


#> merges points of the same source closer than tol (vectorized)
def _merge(sidx, x, y, tol, *extra):
    
    #> sorting by source, then x
    order = np.lexsort((y, x, sidx))
    sidx, x, y = sidx[order], x[order], y[order]
    extra = [e[order] for e in extra]
    
    #> flagging later points within tol of an earlier kept point
    dup = np.zeros(len(x), dtype=bool)
    for lag in range(1, 16):
        if lag >= len(x): break
        same = (sidx[lag:] == sidx[:-lag]) & (np.hypot(x[lag:]-x[:-lag], y[lag:]-y[:-lag]) < tol)
        dup[lag:] |= same & ~dup[:-lag]
    
    keep = ~dup
    return (sidx[keep], x[keep], y[keep], *[e[keep] for e in extra])


#> orders images by fermat potential (vectorized over sources)
def fermat_order(ims, mu, srcs, bprof):

    #> fermat potential at every image
    S, J, _ = ims.shape
    th = ims.reshape(-1, 2)                                  # arcsec
    ok = ~np.isnan(th[:,0])
    tau = np.full(S*J, np.inf)
    b = np.repeat(srcs, J, axis=0)                           # arcsec
    psi = psi_points(th[ok,0], th[ok,1], bprof)              # rad^2
    tau[ok] = 0.5*np.sum(((th[ok]-b[ok])/u.arc_rad)**2, axis=1) - psi # rad^2
    tau = tau.reshape(S, J)

    #> sorting each source's images
    o = np.argsort(tau, axis=1)
    take = lambda a: np.take_along_axis(a, o, axis=1)
    ims = np.stack([take(ims[...,0]), take(ims[...,1])], axis=-1)

    return ims, take(mu), take(tau) # arcsec, signed mags, rad^2


""" #> ORDERING + OBSERVABLES ========
================================== """

#> morphological arrival order (vectorized version of lensing.arrivalOrder for quads)
def morph_order(ims, mu=None):
    """
    ims [np.ndarray]: (Q, 5, 2) images [arcsec] (any order; lens at origin)
    returns: (Q, 5, 2) images in morphological arrival order (+ mu reordered if given)
    """
    
    #> declarations
    Q = len(ims); q = np.arange(Q)[:,None]
    
    #> sorting by distance (furthest first)
    r = np.hypot(ims[...,0], ims[...,1])                      # arcsec
    o = np.argsort(r, axis=1)[:, ::-1]
    oims, r = ims[q, o], r[q, o]
    
    #> 1st (or 4th) image: biggest radial gap between ims 12 or 34
    ind = np.where(r[:,0]-r[:,1] >= r[:,2]-r[:,3], 0, 3)      # argmax picks first on ties
    
    #> outer 4 sorted by polar angle, rolled to start at the 1st/4th image
    dum = oims[:, :4]
    ang = np.arctan2(dum[...,1], dum[...,0])
    oa = np.argsort(ang, axis=1)
    pos = np.argmax(oa == ind[:,None], axis=1)                # where image ind sits after sorting
    roll = (oa[q, (np.arange(4)[None] + pos[:,None]) % 4])   # indices into dum, rolled
    d0, d1, d2, d3 = [dum[np.arange(Q), roll[:,k]] for k in range(4)]
    
    #> angles at the origin (law of cosines, as in lensing.lawCos)
    def lawcos(a, b):
        ra, rb = np.hypot(*a.T), np.hypot(*b.T); c = np.hypot(*(a-b).T)
        return np.degrees(np.arccos(np.clip((ra**2 + rb**2 - c**2)/(2*ra*rb), -1, 1)))
    a21, a23 = lawcos(d2, d1), lawcos(d2, d3)
    near1 = (a21 < a23)[:,None]
    
    #> assigning
    first = (ind == 0)[:,None]
    i1 = np.where(first, dum[:,0], np.where(near1, d3, d1))
    i2 = np.where(first, d2,       np.where(near1, d1, d3))
    i3 = np.where(first, np.where(near1, d1, d3), d2)
    i4 = np.where(first, np.where(near1, d3, d1), dum[:,3])
    out = np.stack([i1, i2, i3, i4, oims[:,4]], axis=1)
    
    #> reordering mags to match (by position)
    if mu is not None:
        idx = np.argmin(np.linalg.norm(out[:,:,None,:] - ims[:,None,:,:], axis=-1), axis=2)
        return out, np.take_along_axis(mu, idx, axis=1)
    return out # arcsec


#> quad observables (vectorized version of geometry.lensObs; lens at origin)
def quad_obs(ims, observables):
    
    #> declarations
    p = [ims[:,k] for k in range(4)]                          # (Q, 2) each [arcsec]
    pol = lambda v: np.degrees(np.arctan2(v[:,1], v[:,0])) % 360
    ccw = lambda a, b: (b - a) % 360
    def contain(pa, pb, pc):                                  # angle pa->pb containing pc
        a, b, c = pol(pa), pol(pb), pol(pc)
        ab, ac = ccw(a, b), ccw(a, c)
        return np.where(ac <= ab, ab, 360 - ab)
    def ang(v1, v2):
        cos = np.sum(v1*v2, 1) / (np.hypot(*v1.T) * np.hypot(*v2.T))
        return np.degrees(np.arccos(cos))
    
    #> angles [deg]
    t12 = contain(p[0], p[1], p[2])
    t23 = ang(p[1], p[2])
    t34 = contain(p[2], p[3], p[1])
    
    #> FSQ (Woldesenbet & Williams 2012, eq 18)
    a, b = np.radians(t12), np.radians(t34)
    t23_fsq = ( -5.792 + (1.783 * a) + (0.1648 * a**2) - (0.04591 * a**3)
    - (0.0001486 * a**4) + (1.784 * b) - (0.7275 * b * a)
    + (0.0549 * b * a**2) + (0.01487 * b * a**3) + (0.1643 * b**2)
    + (0.05493 * b**2 * a) - (0.03429 * b**2 * a**2) - (0.04579 * b**3)
    + (0.01487 * b**3 * a) - (0.0001593 * b**4) )
    dt23 = np.degrees(np.radians(t23) - t23_fsq)
    
    #> distances [arcsec]
    d = {f'd0{k+1}': np.hypot(*p[k].T) for k in range(4)}
    d |= {'d13': np.hypot(*(p[0]-p[2]).T), 'd14': np.hypot(*(p[0]-p[3]).T),
          'd23': np.hypot(*(p[1]-p[2]).T), 'd24': np.hypot(*(p[1]-p[3]).T)}
    data = {'t12': t12, 't23': t23, 't34': t34, 'dt23': dt23,
            'd2/d1': d['d02']/d['d01'], 'd3/d1': d['d03']/d['d01'],
            'd4/d1': d['d04']/d['d01'], 'd3/d2': d['d03']/d['d02']} | d
    for k in range(5): data[f'x{k+1}'], data[f'y{k+1}'] = ims[:,k,0], ims[:,k,1]
    
    return np.column_stack([data[o] for o in observables]) # (Q, numObs)


""" #> PIPELINE (ONE GALAXY) =========
================================== """

#> caustic (tangential) from the critical curve on the grid, back-projected w/ analytic deflections
def caustic_points(xg, yg, bprof, lamt):
    
    #> imports
    from skimage import measure
    
    #> critical curve (longest contour of lambda_t = 0)
    pix = xg[0,1] - xg[0,0]                                  # arcsec/pix
    cs = measure.find_contours(lamt, 0.0)
    if not cs: return None
    cc = max(cs, key=len)                                    # (row, col)
    tx = xg[0,0] + cc[:,1]*pix; ty = yg[0,0] + cc[:,0]*pix   # arcsec
    
    #> back-projecting
    ax, ay = alpha_points(tx, ty, bprof)
    
    return np.column_stack((tx - ax, ty - ay)) # arcsec


#> lenses numSource sources behind one galaxy (all vectorized)
def lens_galaxy(bprof, numSource, nph=50, rng=None, **kwargs):
    
    #> kwargs
    method      = kwargs.get('method', 'deflect')   # 'deflect' (morph order) or 'lpot' (fermat order + delays)
    observables = kwargs.get('observables', None)   # list of quad observables
    srcs        = kwargs.get('srcs', None)          # supplied sources [arcsec] (otherwise random in caustic)
    maxRedraw   = kwargs.get('maxRedraw', 10)       # redraw rounds for non-quads
    if rng is None: rng = np.random
    
    #> grid [arcsec] + source-plane map
    pix_arc = bprof['pix_arc']
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc       # arcsec
    xg, yg = np.meshgrid(w, w)
    ax, ay = alpha_points(xg, yg, bprof)                      # arcsec
    bxg, byg = xg - ax, yg - ay                               # arcsec
    
    #> sources
    if srcs is None:
        
        #> tangential eigenvalue from the grid jacobian (finite differences)
        axy_, axx_ = np.gradient(ax, w, w); ayy_, ayx_ = np.gradient(ay, w, w)
        kap = 0.5*(axx_ + ayy_); g1 = 0.5*(axx_ - ayy_); g2 = 0.5*(axy_ + ayx_)
        lamt = 1 - kap - np.hypot(g1, g2)
        caus = caustic_points(xg, yg, bprof, lamt)
        srcs = sample_sources(caus, numSource, rng=rng)       # arcsec
        
        #> lensing + redrawing sources that do not give a quad
        res = find_images(xg, yg, bxg, byg, srcs, bprof)
        for _ in range(maxRedraw):
            bad = np.flatnonzero(res['nims'] != 5)
            if len(bad) == 0: break
            srcs[bad] = sample_sources(caus, len(bad), rng=rng)
            r2 = find_images(xg, yg, bxg, byg, srcs[bad], bprof)
            for k in res: res[k][bad] = r2[k]
    else:
        srcs = np.asarray(srcs, dtype=float)
        res = find_images(xg, yg, bxg, byg, srcs, bprof)
    
    #> ordering (only complete quads)
    ims, mu = res['images'], res['mu']
    quad = res['nims'] == 5
    out = {'images': np.full_like(ims, np.nan), 'mu': np.full_like(mu, np.nan),
           'srcs': srcs, 'nims': res['nims']}
    if method == 'lpot':
        o_ims, o_mu, tau = fermat_order(ims[quad], mu[quad], srcs[quad], bprof)
        D_dt = (1+bprof['zl']) * 4*np.pi*u.G_kpc_solMass * bprof['angDist']**2 * bprof['sigCrit'] / u.c_kpc**2 # kpc
        out['delays'] = np.full(mu.shape, np.nan)
        out['delays'][quad] = D_dt / u.c_kpc * (tau - tau[:,:1]) / 86400 # days
    else:
        o_ims, o_mu = morph_order(ims[quad], mu[quad])
    out['images'][quad], out['mu'][quad] = o_ims, o_mu
    
    #> observables
    if observables is not None:
        out['obs'] = np.full((len(srcs), len(observables)), np.nan)
        out['obs'][quad] = quad_obs(o_ims, observables)
    
    return out


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__),'\n')
    
    #> imports
    import time
    import params
    import generate
    
    #> declarations
    numGals       = 10                                     # galaxies
    numSource_gal = 100                                    # sources per galaxy
    nph           = 50                                     # grid half-width [pix]
    observables   = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    rng           = np.random.default_rng(42)              # random generator
    
    #> galaxies
    galProf     = generate.galProfiles(nfw=True, hern=True, mult=[], ex=False)
    paramRanges = params.toggleParams(galProf)
    bprofiles   = params.bprofiles(numGals, ranges=paramRanges, redshifts=np.tile((0.5, 1.0), (numGals, 1)))
    
    #> lensing!
    srt = time.time()
    outs = [lens_galaxy(bp, numSource_gal, nph=nph, rng=rng, method='lpot', observables=observables) for bp in bprofiles]
    dt = time.time() - srt
    
    #> print
    numQuads = sum(int((o['nims'] == 5).sum()) for o in outs)
    print(f'> {numQuads} quads in {dt:.2f} s ({dt/numQuads*1e3:.3f} ms/quad)')
    print('> first quad images [arcsec]:\n', outs[0]['images'][0])
    print('> first quad delays [days]:\n', outs[0]['delays'][0])
    
    # end
# thank
