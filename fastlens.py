#> name: fastlens.py
#> author: John Miller Jr
#> descrp: vectorized image finding for opus: grid seeding + newton refinement on analytic deflections

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
from matplotlib.path import Path

#> modules
import modules.error as error
from modules.units import u; u=u()

#> declarations
global maxTries
maxTries = 100

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


""" #> CSE PROFILE ===================
================================== """

#> CSE initialization
def cse_frame(x, y, prof, norm_key, angDist, sigCrit):
    
    #> unpacking kwargs
    arc_rad = u.arc_rad                             # arcsec/rad
    x0      = prof.get('x0', 0.0)                   # arcsec
    y0      = prof.get('y0', 0.0)                   # arcsec
    q       = prof.get('axisrat', 1.0)              # []
    r0      = prof.get('radius', 1.0)               # kpc
    rho0    = prof.get(norm_key, 0.0)               # solMass/kpc^3
    th      = prof.get('theta', 0.0) * np.pi/180    # deg --> rad
    
    #> declarations
    ct    = np.cos(th)
    st    = np.sin(th)
    denom = (r0 / np.sqrt(q) / angDist) * arc_rad   # r0' [arcsec]
    bnorm = r0*rho0 / sigCrit                       # b_norm []
    
    #> shifting, rotating, and scaling points (arcsec --> units of r0')
    Xr  = (  (x - x0)*ct + (y - y0)*st ) / denom
    Yr  = ( -(x - x0)*st + (y - y0)*ct ) / denom
    shp = Xr.shape
    
    #> flattening and reshaping
    Xr = Xr.ravel()[None]                           # shape=(1,N)
    Yr = Yr.ravel()[None]                           # shape=(1,N)
    
    return Xr, Yr, shp, q, ct, st, denom, bnorm


#> CSE deflection angles
def cse_alpha(x, y, prof, coeffs, norm_key, angDist, sigCrit):
    
    #> profile frame
    Xr, Yr, shp, q, ct, st, denom, bnorm = cse_frame(x, y, prof, norm_key, angDist, sigCrit)
    
    #> getting coefficents
    Ai = coeffs[:,0,None]                 # shape=(Nc,1)
    Si = coeffs[:,1,None]                 # shape=(Nc,1)
    
    #> declarations
    X2 = Xr*Xr
    Y2 = Yr*Yr
    q2 = q*q
    
    #>>># DEFLECT #<<<#
    
    #> separating equations
    lowp = np.sqrt(q2*(Si*Si + X2) + Y2)  # eq 18, shape=(Nc,N)
    upp  = (lowp + Si)**2 + (1 - q2)*X2   # eq 17
    w    = Ai*q / (Si * lowp * upp)       # part of eq 19/20
    
    #> calculating deflection angles! (and reshapes)
    gx = (w*Xr*(lowp + q2*Si)).sum(axis=0).reshape(shp)  # eq 19
    gy = (w*Yr*(lowp + Si)   ).sum(axis=0).reshape(shp)  # eq 20
    
    #> scaling (--> arcsec)
    scale = bnorm * denom
    gx   *= scale          # arcsec
    gy   *= scale          # arcsec
    
    #> rotating back
    gx_ = gx*ct - gy*st
    gy_ = gx*st + gy*ct
    
    return gx_, gy_ # arcsec, arcsec


#> CSE deflection angles + jacobian (Oguri 2021 eqs 19-23)
def cse_alpha_jac(x, y, prof, coeffs, norm_key, angDist, sigCrit):
    
    #> profile frame
    Xr, Yr, shp, q, ct, st, denom, bnorm = cse_frame(x, y, prof, norm_key, angDist, sigCrit)
    
    #> getting coefficents
    Ai = coeffs[:,0,None]                 # shape=(Nc,1)
    Si = coeffs[:,1,None]                 # shape=(Nc,1)
    
    #> declarations
    X2    = Xr*Xr                         # shape=(1,N)
    Y2    = Yr*Yr                         # shape=(1,N)
    XY    = Xr*Yr                         # shape=(1,N)
    q2    = q*q
    Si2   = Si*Si
    q2Si  = q2*Si                         # q^2 s
    q2Si2 = q2Si*Si                       # q^2 s^2
    Aiq_Si = Ai*q/Si                      # A q/s
    
    #>>># DEFLECT + JACOBIAN #<<<#
    
    #> separating equations
    lowp   = np.sqrt(q2*(Si2 + X2) + Y2)  # eq 18, shape=(Nc,N)
    lowps  = lowp + Si                    # psi + s
    upp    = lowps*lowps + (1 - q2)*X2    # eq 17
    A      = Aiq_Si / upp                 # Ai q/(s Psi)
    ilowp  = 1/lowp                       # 1/psi
    ilowp2 = ilowp*ilowp                  # 1/psi^2
    
    #> numerators
    ax     = Xr*(lowp + q2Si)             # X (psi + q^2 s)
    ay     = Yr*lowps                     # Y (psi + s)
    
    #> calculating deflection angles!
    pil    = A*ilowp                      # A q/(s psi Psi)
    gx     = (pil*ax).sum(axis=0)         # eq 19
    gy     = (pil*ay).sum(axis=0)         # eq 20
    
    #> shared jacobian terms
    B      = q2Si*ilowp2*ilowp            # q^2 s/psi^3
    C      = 2*ilowp2/upp                 # 2/(psi^2 Psi)
    
    #> calculating alpha derivatives
    gxx    = ( A*(1 + B*(q2Si2 + Y2) - ax*ax*C)).sum(axis=0)  # eq 21
    gyy    = ( A*(1 + B*(Si2 + X2)   - ay*ay*C)).sum(axis=0)  # eq 22
    gxy    = (-A*(XY*B + ax*ay*C)).sum(axis=0)                # eq 23
    
    #> back to original shape
    gx  = gx.reshape(shp)
    gy  = gy.reshape(shp)
    gxx = gxx.reshape(shp)
    gxy = gxy.reshape(shp)
    gyy = gyy.reshape(shp)
    
    #>>># DEFLECT #<<<#
    
    #> scaling (--> arcsec)
    scale = bnorm * denom
    gx   *= scale          # arcsec
    gy   *= scale          # arcsec
    
    #> rotating deflection angles back
    gx_ = gx*ct - gy*st
    gy_ = gx*st + gy*ct
    
    #>>># JACOBIAN #<<<#
    
    #> scaling jacobian
    gxx *= bnorm
    gxy *= bnorm
    gyy *= bnorm
    
    #> rotating jacobian back
    c2   = ct*ct
    s2   = st*st
    cs   = ct*st
    gxx_ = gxx*c2 - 2*gxy*cs + gyy*s2
    gyy_ = gxx*s2 + 2*gxy*cs + gyy*c2
    gxy_ = (gxx - gyy)*cs + gxy*(c2 - s2)
    
    return gx_, gy_, gxx_, gxy_, gxy_, gyy_ # arcsec, arcsec, [], [], [], []


#> CSE: lens potential
def cse_lpot(x, y, prof, coeffs, norm_key, angDist, sigCrit):
    
    #> profile frame
    Xr, Yr, shp, q, ct, st, denom, bnorm = cse_frame(x, y, prof, norm_key, angDist, sigCrit)
    
    #> getting coefficents
    Ai = coeffs[:,0,None]                 # shape=(Nc,1)
    Si = coeffs[:,1,None]                 # shape=(Nc,1)
    
    #> declarations
    X2 = Xr*Xr
    q2 = q*q
    q_Si = q/Si
    
    #> separating equations
    lowp = np.sqrt(q2*(Si*Si + X2) + Yr*Yr)
    upp = (lowp + Si)**2 + X2*(1 - q2)
    
    #> calculating lpot!
    lpot = (Ai * ((q_Si/2)*np.log(upp) - (q_Si)*np.log((1+q)*Si))).sum(axis=0)

    #> scaling (--> rad^2)
    r0    = prof.get('radius', 1.0)
    scale = bnorm * (r0 / np.sqrt(q) / angDist)**2     # rad^2
    lpot  = (lpot * scale).reshape(shp)

    return lpot # rad^2


""" #> MULTIPOLES ====================
================================== """

#> MULT initialization
def mult_frame(x, y, prof):
    
    #> unpacking kwargs
    x0  = prof.get('x0', 0.0)                 # arcsec
    y0  = prof.get('y0', 0.0)                 # arcsec
    m   = prof.get('m', 1)                    # []
    am  = prof.get('norm', 0.0)               # a_m []
    gam = prof.get('slope', 2)                # []
    th  = prof.get('theta', 0.0) * np.pi/180  # deg --> rad
    
    #> declarations
    k = -am/m                                 # -a_m/m
    
    #> shifting points
    Xr = x - x0                               # arcsec
    Yr = y - y0                               # arcsec
    r2 = Xr*Xr + Yr*Yr
    
    #> angular terms
    mphi = m*(np.arctan2(Yr, Xr) - th)
    cm   = np.cos(mphi)
    sm   = np.sin(mphi)
    
    return Xr, Yr, r2, cm, sm, m, gam, k


#> MULT deflection angles
def mult_alpha(x, y, prof):
    
    #> multipole frame
    Xr, Yr, r2, cm, sm, m, gam, k = mult_frame(x, y, prof)
    
    #> declarations
    pos = r2 > 0                                 # avoids r=0
    r2s = np.where(pos, r2, 1.0)                 # arcsec^2
    R   = np.where(pos, r2s**(0.5*gam - 1), 0.0) # r^(gam-2)
    
    #>>># DEFLECT #<<<#
    
    #> numerators
    F = gam*Xr*cm + m*Yr*sm                      # gam X cos + m Y sin
    G = gam*Yr*cm - m*Xr*sm                      # gam Y cos - m X sin
    
    #> calculating deflection angles!
    gx = k*R*F
    gy = k*R*G
    
    return gx, gy # arcsec, arcsec


#> MULT deflection angles + jacobian
def mult_alpha_jac(x, y, prof):
    
    #> multipole frame
    Xr, Yr, r2, cm, sm, m, gam, k = mult_frame(x, y, prof)
    
    #> declarations
    pos = r2 > 0                                 # avoids r=0
    r2s = np.where(pos, r2, 1.0)                 # arcsec^2
    R   = np.where(pos, r2s**(0.5*gam - 1), 0.0) # r^(gam-2)
    R4  = R / r2s                                # r^(gam-4)
    X2  = Xr*Xr                                  # arcsec^2
    Y2  = Yr*Yr                                  # arcsec^2
    XY  = Xr*Yr                                  # arcsec^2
    g_2 = gam - 2                                # gam - 2
    m2  = m*m                                    # m^2
    gm  = gam*m                                  # gam m
    
    #>>># DEFLECT + JACOBIAN #<<<#
    
    #> numerators
    F = gam*Xr*cm + m*Yr*sm                      # gam X cos + m Y sin
    G = gam*Yr*cm - m*Xr*sm                      # gam Y cos - m X sin
    
    #> calculating deflection angles!
    gx = k*R*F
    gy = k*R*G
    
    #> calculating derivatives
    gxx = k*R4*(g_2*Xr*F + gam*r2*cm + gm*XY*sm - m2*Y2*cm)
    gxy = k*R4*(g_2*Yr*F + m*r2*sm   - gm*X2*sm + m2*XY*cm)
    gyy = k*R4*(g_2*Yr*G + gam*r2*cm - gm*XY*sm - m2*X2*cm)
    
    return gx, gy, gxx, gxy, gxy, gyy # arcsec, arcsec, [], [], [], []


#> MULT: lens potential
def mult_lpot(x, y, prof):
    
    #> multipole frame
    Xr, Yr, r2, cm, sm, m, gam, k = mult_frame(x, y, prof)
    
    #> calculating lpot
    lpot = k * r2**(0.5*gam) * cm / u.arc_rad**2
    
    return lpot # rad^2


""" #> EXTERNAL SHEAR ================
================================== """

#> EX initialization
def shear_frame(prof):
    
    #> unpacking kwargs
    g  = prof.get('norm',  0.0)              # gamma []
    th = prof.get('theta', 0.0) * np.pi/180  # deg --> rad
    
    #> declarations
    c2 = np.cos(2*th)
    s2 = np.sin(2*th)
    
    return g, th, c2, s2


#> EX: deflection angles
def shear_alpha(x, y, prof):
    
    #> shear frame
    g, th, c2, s2 = shear_frame(prof)
    
    #>>># DEFLECT #<<<#
    
    #> calculating deflection angles!
    gx = -g*(x*c2 + y*s2)
    gy = -g*(x*s2 - y*c2)
    
    return gx, gy # arcsec, arcsec


#> EX: deflection angles + jacobian
def shear_alpha_jac(x, y, prof):
    
    #> shear frame
    g, th, c2, s2 = shear_frame(prof)
    
    #>>># DEFLECT + JACOBIAN #<<<#
    
    #> calculating deflection angles!
    gx = -g*(x*c2 + y*s2)
    gy = -g*(x*s2 - y*c2)
    
    #> calculating alpha derivatives (constant)
    gxx = -g*c2
    gxy = -g*s2
    gyy =  g*c2
    
    return gx, gy, gxx, gxy, gxy, gyy # arcsec, arcsec, [], [], [], []


#> EX: lens potential
def shear_lpot(x, y, prof):

    #> shear frame
    g, th, c2, s2 = shear_frame(prof)
    
    #> calculating lpot
    lpot = -g/2 * ((x*x - y*y)*c2 + 2*x*y*s2) / u.arc_rad**2
    
    return lpot # rad^2


""" #> GALAXY PROFILES ===============
================================== """

#> deflection angles [arcsec]
def alpha_points(x, y, bprof):
    
    #> unpacking kwargs
    angDist = bprof['angDist'] # kpc
    sigCrit = bprof['sigCrit'] # solMass/kpc^2
    
    #> declarations (profile, coeffs, norm key)
    cse_profs = [('nfw',  NFW_COEFFS,  'rho0'),
                 ('hern', HERN_COEFFS, 'norm')]
    
    #> initializing deflection angles (ax, ay)
    #> format: ax, ay
    out = [np.zeros_like(x, dtype=np.float64) for _ in range(2)]
    
    #>>># NFW + HERNQUIST #<<<#
    for key, coeffs, norm_key in cse_profs:
        for prof in bprof.get(key, []): # both nfw and hern
            for o, v in zip(out, cse_alpha(x, y, prof, coeffs, norm_key, angDist, sigCrit)): o += v
    
    #>>># MULTIPOLES #<<<#
    for prof in bprof.get('mult', []):
        if prof.get('norm', 0.0) == 0: continue
        for o, v in zip(out, mult_alpha(x, y, prof)): o += v
    
    #>>># EX #<<<#
    for prof in bprof.get('ex', []):
        if prof.get('norm', 0.0) == 0: continue
        for o, v in zip(out, shear_alpha(x, y, prof)): o += v
    
    return out # summed (ax, ay)


#> deflection angles + jacobian [arcsec, .]
def alpha_jac_points(x, y, bprof):
    
    #> unpacking kwargs
    angDist = bprof['angDist'] # kpc
    sigCrit = bprof['sigCrit'] # solMass/kpc^2
     
    #> declarations (profile, coeffs, norm key)
    cse_profs = [('nfw',  NFW_COEFFS,  'rho0'),
                 ('hern', HERN_COEFFS, 'norm')]  
    
    #> initializing deflection angles (ax, ay) and jacobian (axx, axy, ayx, ayy)
    #> format: ax, ay, axx, axy, ayx, ayy
    out = [np.zeros_like(x, dtype=np.float64) for _ in range(6)] 
    
    #>>># NFW + HERNQUIST #<<<#
    for key, coeffs, norm_key in cse_profs:
        for prof in bprof.get(key, []): # both nfw and hern
            for o, v in zip(out, cse_alpha_jac(x, y, prof, coeffs, norm_key, angDist, sigCrit)): o += v
    
    #>>># MULTIPOLES #<<<#
    for prof in bprof.get('mult', []):
        if prof.get('norm', 0.0) == 0: continue
        for o, v in zip(out, mult_alpha_jac(x, y, prof)): o += v
    
    #>>># EX #<<<#
    for prof in bprof.get('ex', []):
        if prof.get('norm', 0.0) == 0: continue
        for o, v in zip(out, shear_alpha_jac(x, y, prof)): o += v
    
    return out # summed (ax, ay, axx, axy, ayx, ayy)


#> lens potential [rad^2]
def lpot_points(x, y, bprof):

    #> declarations
    angDist = bprof['angDist'] # kpc
    sigCrit = bprof['sigCrit'] # solMass/kpc^2
    
    #> declarations (profile, coeffs, norm key)
    cse_profs = [('nfw',  NFW_COEFFS,  'rho0'),
                 ('hern', HERN_COEFFS, 'norm')]  
    
    #> initlaize lpot
    lpot = np.zeros_like(x, dtype=np.float64)

    #>>># NFW + HERNQUIST #<<<#
    for key, coeffs, norm_key in cse_profs:
        for prof in bprof.get(key, []):
            lpot += cse_lpot(x, y, prof, coeffs, norm_key, angDist, sigCrit)

    #>>># MULTIPOLES #<<<#
    for prof in bprof.get('mult', []):
        if prof.get('norm', 0.0) == 0: continue
        lpot += mult_lpot(x, y, prof)

    #>>># EX #<<<#
    for prof in bprof.get('ex', []):
        if prof.get('norm', 0.0) == 0: continue
        lpot += shear_lpot(x, y, prof)

    return lpot # rad^2


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
    returns: dict w/ images (S, jmax, 2) [arcsec], mu (S, jmax) [signed], nims (S,)
    """

    #> unpacking kwargs
    nit1      = kwargs.get('nit1', 20)         # stage-1 (grid) newton iterations
    nit       = kwargs.get('nit', 15)          # stage-2 (analytic) max newton iterations
    maxstep   = kwargs.get('maxstep', 3.0)     # max stage-2 newton step [pix]
    tol       = kwargs.get('tol', 1e-9)        # convergence on step size [arcsec]
    tol_res   = kwargs.get('tol_res', 1e-7)    # accepted lens-eq residual [arcsec]
    tol_merge = kwargs.get('tol_merge', 1e-5)  # duplicate roots closer than this are merged [arcsec]
    jmax      = kwargs.get('jmax', 5)          # images kept per source

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
    #  4. seed cells that contain a source

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
    #  1. 

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
        a = ( (bx01-bx00)*(1-dy) + (bx11-bx10)*dy ) / arc_pix
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
    sidx, x, y = _merge(sidx, x, y, 0.25*arc_pix)
    bx = srcs[sidx,0]                          # arcsec
    by = srcs[sidx,1]                          # arcsec


    #>>># STAGE 2: NEWTON ON ANALYTIC DEFLECTIONS (w/ backtracking) #<<<#

    #> analytic deflections + jacobian at every root
    ax, ay, axx, axy, ayx, ayy = alpha_jac_points(x, y, bprof)

    #> initial residuals, f = theta - alpha(theta) - beta_s
    rx, ry = x - ax - bx, y - ay - by                        # arcsec
    rn = np.hypot(rx, ry)                                    # |f| [arcsec]

    #> roots still iterating
    active = np.ones(len(x), dtype=bool)

    #> newton iterations (only on active roots)
    for _ in range(nit):
        if not active.any(): break
        ia = np.flatnonzero(active)                          # indices of active roots

        #> lensing jacobian, A = I - d(alpha)/d(theta) = [[a, b], [c, d]]
        a, b, c, d = 1-axx[ia], -axy[ia], -ayx[ia], 1-ayy[ia]
        det = a*d - b*c

        #> newton step, s = A^-1 f
        sx = ( d*rx[ia] - b*ry[ia]) / det                    # arcsec
        sy = (-c*rx[ia] + a*ry[ia]) / det                    # arcsec
        step = np.hypot(sx, sy)                              # arcsec

        #> damping: step length capped at maxstep pixels
        lam = np.minimum(1.0, maxstep*arc_pix / np.maximum(step, 1e-300))

        #> backtracking: halving the step until the residual decreases (max 8 halvings)
        todo = np.arange(len(ia))                            # positions in ia still searching
        for _ in range(8):

            #> trial positions, theta - lam*s
            k = ia[todo]                                     # root indices
            xt = x[k] - lam[todo]*sx[todo]                   # arcsec
            yt = y[k] - lam[todo]*sy[todo]                   # arcsec

            #> trial residuals
            o = alpha_jac_points(xt, yt, bprof)              # (ax, ay, axx, axy, ayx, ayy)
            rxt = xt - o[0] - bx[k]                          # arcsec
            ryt = yt - o[1] - by[k]                          # arcsec
            rnt = np.hypot(rxt, ryt)                         # arcsec

            #> accepting trials w/ a sufficient decrease in |f|
            ok = rnt < rn[k] * (1 - 1e-4*lam[todo]) + 1e-12
            kk = k[ok]

            #> updating accepted roots (positions, residuals, jacobian)
            x[kk], y[kk], rx[kk], ry[kk], rn[kk] = xt[ok], yt[ok], rxt[ok], ryt[ok], rnt[ok]
            axx[kk], axy[kk], ayx[kk], ayy[kk] = o[2][ok], o[3][ok], o[4][ok], o[5][ok]

            #> halving the step for the rejected trials
            todo = todo[~ok]
            lam[todo] *= 0.5
            if len(todo) == 0: break

        #> deactivating: converged (tiny step), stalled (step cut below 1%), or no decrease found
        active[ia[(lam*step < tol)]] = False
        active[ia[lam < 1e-2]] = False
        active[ia[todo]] = False


    #>>># KEEPING ROOTS #<<<#

    #> det(A) at each root (magnification, mu = 1/det(A))
    detA = (1-axx)*(1-ayy) - axy*ayx

    #> keeping roots that solve the lens equation
    good = rn < tol_res
    sidx, x, y, detA = sidx[good], x[good], y[good], detA[good]

    #> merging duplicate roots (det(A) carried along)
    sidx, x, y, detA = _merge(sidx, x, y, tol_merge, detA)


    #>>># PACKING PER SOURCE (NAN PADDED) #<<<#

    #> number of images per source
    nims = np.bincount(sidx, minlength=nsrcs)

    #> grouping roots by source
    order = np.argsort(sidx, kind='stable')
    sidx, x, y, detA = sidx[order], x[order], y[order], detA[order]

    #> slot of each root within its source (0, 1, 2, ...); only the first jmax are kept
    slot = np.arange(len(sidx)) - np.repeat(np.cumsum(nims) - nims, nims)
    keep = slot < jmax

    #> filling (S, jmax) arrays
    ims = np.full((nsrcs, jmax, 2), np.nan)                  # arcsec
    mu  = np.full((nsrcs, jmax), np.nan)                     # signed mags
    ims[sidx[keep], slot[keep], 0] = x[keep]
    ims[sidx[keep], slot[keep], 1] = y[keep]
    mu[sidx[keep], slot[keep]] = 1/detA[keep]

    return {'images': ims, 'mu': mu, 'nims': nims} # arcsec, signed mags, counts


#> merges points of the same source closer than tol (vectorized)
def _merge(sidx, x, y, tol, *extra):

    #> sorting by source, then x, then y
    order = np.lexsort((y, x, sidx))
    sidx, x, y = sidx[order], x[order], y[order]
    extra = [e[order] for e in extra]          # extra arrays follow the same order

    #> flagging later points within tol of an earlier kept point (up to 15 neighbours back)
    dup = np.zeros(len(x), dtype=bool)
    for lag in range(1, 16):
        if lag >= len(x): break

        #> point i vs point i-lag: same source + closer than tol
        same = (sidx[lag:] == sidx[:-lag]) & (np.hypot(x[lag:]-x[:-lag], y[lag:]-y[:-lag]) < tol)

        #> duplicate only if the earlier point is itself kept
        dup[lag:] |= same & ~dup[:-lag]

    #> keeping non-duplicates
    keep = ~dup
    return (sidx[keep], x[keep], y[keep], *[e[keep] for e in extra])


""" #> ORDERING ======================
================================== """

#> orders images by fermat potential
def fermat_order(ims, mu, srcs, bprof):

    #> flatting images (S=# sources, I=# images per source)
    S, I, _ = ims.shape
    th      = ims.reshape(-1, 2) # thetas
    
    #> nan handling
    ok = ~np.isnan(th[:,0])
    
    #> declarations
    tau  = np.full(S*I, np.inf)        # rad^2
    beta = np.repeat(srcs, I, axis=0)  # source positions

    #> lens potential
    psi = lpot_points(th[ok,0], th[ok,1], bprof) # rad^2
    
    #> fermat potentail
    al2 = np.sum( ( (th[ok] - beta[ok]) / u.arc_rad )**2, axis=1)
    tau[ok] = 0.5*al2 - psi  # rad^2
    tau = tau.reshape(S, I) 

    #> fermat order
    order = np.argsort(tau, axis=1)
    
    #> sorting all observables
    take = lambda array: np.take_along_axis(array, order, axis=1)
    ims = np.stack([take(ims[:,:,0]), take(ims[:,:,1])], axis=-1)

    return ims, take(mu), take(tau) # arcsec, signed mags, rad^2


#> morphological arrival order
def morph_order(ims, mu=None):
    
    #> declarations
    numIms = len(ims)
    q = np.arange(numIms)[:,None]
    
    #> sorting by distance (furthest first)
    r = np.hypot(ims[:,:,0], ims[:,:,1]) # arcsec
    o = np.argsort(r, axis=1)[:,::-1]
    oims, r = ims[q, o], r[q, o]
    
    #> 1st (or 4th) image, depending on the biggest radial gap between ims 12 or 34
    ind = np.where(r[:,0]-r[:,1] >= r[:,2]-r[:,3], 0, 3)
    
    #> removing 5th image (for now)
    dummy = oims[:,:4]
    
    #> sorting by angle (ensures switching between m & s)
    ang    = np.arctan2(dummy[:,:,1], dummy[:,:,0]) # rad
    oaindx = np.argsort(ang, axis=1)
    
    #> starts quad arrays at ind, orders in ccw angle
    pos  = np.argmax(oaindx == ind[:,None], axis=1)
    wrap = (np.arange(4)[None] + pos[:,None]) % 4
    roll = oaindx[q, wrap]
    d0, d1, d2, d3 = [ dummy[ np.arange(numIms), roll[:,k] ] for k in range(4) ]
    
    #> d0 is first (fourth), and d2 is second (third)
    
    #> angles at the origin
    def lawcos(a, b):
        
        #> calculating distances
        ra = np.hypot(*a.T)
        rb = np.hypot(*b.T)
        c  = np.hypot(*(a-b).T)
        
        #> calc anlge C
        tc = (ra**2 + rb**2 - c**2) / (2*ra*rb)
        return np.degrees(np.arccos(np.clip(tc, a_min=-1, a_max=1)))
    
    #> finding third (second) by finding smallest t23
    a21   = lawcos(d2, d1)
    a23   = lawcos(d2, d3)
    near1 = (a21 < a23)[:,None]
    
    #> quads where first arriving is found (=True), (fourth=False)
    first = (ind == 0)[:,None]
    
    #> if ind == 0 (i.e., first arriving is found)
    #> d0 = first
    #> d2 = second
    #> if near1=True, d1 is third , else d3 is third
    #> if near1=True, d3 is fourth, else d1 is fourth
    
    #> if ind == 3 (i.e., fourth arriving is found)
    #> d0 = fourth
    #> d2 = third
    #> if near1=True, d1 is second, else d3 is second
    #> if near1=True, d3 is first , else d1 is first
    
    #> assigning images  # if first               # if fourth
    i1 = np.where(first, dummy[:,0]             , np.where(near1, d3, d1))
    i2 = np.where(first, d2                     , np.where(near1, d1, d3))
    i3 = np.where(first, np.where(near1, d1, d3), d2)
    i4 = np.where(first, np.where(near1, d3, d1), dummy[:,3])
    
    #> collecting + adding back fifth arriving
    out = np.stack([i1, i2, i3, i4, oims[:,4]], axis=1)
    
    #> reordering mags to match (by position)
    if mu is not None:
        
        #> distance from every output image to every input image
        diff = out[:,:,None,:] - ims[:,None,:,:]           # (Q,5,5,2)
        dist = np.hypot(diff[:,:,:,0], diff[:,:,:,1])      # (Q,5,5)
        
        #> input slot of each output image (closest match)
        idx = np.argmin(dist, axis=2)                      # (Q,5)
        
        #> reordering mags
        mu = np.take_along_axis(mu, idx, axis=1)
        return out, mu
    
    return out


""" #> OBSERVABLES ===================
================================== """

#> lensing observables
def quad_obs(ims, observables):
    
    #> declarations
    o = (0., 0.) # origin
    
    #> separates into images 1, 2, etc.
    p = [ ims[:,k] for k in range(4) ]
    
    #>>># EQUATIONS #<<<#
    
    #> polar angle of vector [0, 360), +x-axis=0, ccw+
    th   = lambda v: np.degrees(np.arctan2(v[:,1], v[:,0])) % 360
    
    #> ccw angle between two anlges a & b [0, 360)
    ccw  = lambda a, b: (b - a) % 360
    
    #> distance between two vectors v & u
    dist = lambda v, u: np.hypot(*(v - u).T)
    
    #> angle between point a & b, to contain point c
    def contain(pa, pb, pc):
        a, b, c = th(pa), th(pb), th(pc)
        ab, ac  = ccw(a, b), ccw(a, c)
        return np.where(ac <= ab, ab, 360 - ab) # ab if contain, else 360 - ab
    
    #> returns angle [deg] between two vectors
    def ang(v1, v2):
        cos = np.sum(v1*v2, axis=1) / (dist(v1,o) * dist(v2,o))
        return np.degrees(np.arccos(np.clip(cos, a_min=-1, a_max=1)))
    
    
    #>>># CALC OBSERVABLES #<<<#
    
    #> angles [deg]
    t12 = contain(p[0], p[1], p[2]) # contains image 3, [0, 360), [0, ~180]
    t23 = ang(p[1], p[2])           #                   [0, 180], [0, ~90]
    t34 = contain(p[2], p[3], p[1]) # contains image 2  [0, 360), [0, ~180]
    
    #> FSQ (Woldesenbet & Williams 2012, eq 18)
    a = np.radians(t12) 
    b = np.radians(t34)
    t23_fsq = ( -5.792 + (1.783 * a) + (0.1648 * a**2) - (0.04591 * a**3)
              - (0.0001486 * a**4) + (1.784 * b) - (0.7275 * b * a)
              + (0.0549 * b * a**2) + (0.01487 * b * a**3) + (0.1643 * b**2)
              + (0.05493 * b**2 * a) - (0.03429 * b**2 * a**2) - (0.04579 * b**3)
              + (0.01487 * b**3 * a) - (0.0001593 * b**4) )
    dt23 = t23 - np.degrees(t23_fsq)
    
    #> distances [arcsec]
    d = {f'd0{k+1}': dist(p[k], o) for k in range(4)}
    d |= {'d13': dist(p[0], p[2]), 
          'd14': dist(p[0], p[3]),
          'd23': dist(p[1], p[2]), 
          'd24': dist(p[1], p[3])}
    
    #> collecting all observables
    data = {'t12': t12, 't23': t23, 't34': t34, 'dt23': dt23,
            'd2/d1': d['d02']/d['d01'], 'd3/d1': d['d03']/d['d01'],
            'd4/d1': d['d04']/d['d01'], 'd3/d2': d['d03']/d['d02']} | d
    
    #> saving image positions
    for i in range(5): 
        data[f'x{i+1}'] = ims[:,i,0] # xi
        data[f'y{i+1}'] = ims[:,i,1] # yi
    
    #> reshaping and returning
    return np.column_stack([data[x] for x in observables]) # (Q, numObs)


""" #> PIPELINE (ONE GALAXY) =========
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
    ax, ay = alpha_points(tx, ty, bprof)
    
    #> back-projecting
    return np.column_stack((tx - ax, ty - ay)) # arcsec


#> lenses numSource sources behind one galaxy (all vectorized)
def lens_galaxy(bprof, numSource, nph=50, rng=None, **kwargs):
    
    #> unpacking kwargs
    method      = kwargs.get('method', 'deflect')   # 'deflect' (morph order) or 'lpot' (fermat order + delays)
    observables = kwargs.get('observables', None)   # list of lenisng observables
    srcs        = kwargs.get('srcs', None)          # supplied sources [arcsec] (otherwise random in caustic)
    maxRedraw   = kwargs.get('maxRedraw', 10)       # redraw rounds for non-quads
    if rng is None: rng = np.random
    
    #> declarations
    pix_arc = bprof['pix_arc']
    
    
    #>>># CREATING GALAXY #<<<#
    
    #> lens-plane grid (theta)
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc
    xg, yg = np.meshgrid(w, w)
    
    #> deflection angles + beta
    ax, ay   = alpha_points(xg, yg, bprof)    # arcsec
    bxg, byg = xg - ax, yg - ay               # beta_i = theta_i - alpha_i [arcsec]
    
    
    #>>># LENSING #<<<#
    
    #> sources
    if srcs is None:
        
        #> calculating tangential caustic
        axy_, axx_ = np.gradient(ax, w, w)
        ayy_, ayx_ = np.gradient(ay, w, w)
        kappa   = 0.5*(axx_ + ayy_) 
        gamma1  = 0.5*(axx_ - ayy_)
        gamma2  = 0.5*(axy_ + ayx_)
        lamt = 1 - kappa - np.hypot(gamma1, gamma2)
        caus = caustic_points(xg, yg, bprof, lamt)
        
        #> no critical curve
        if caus is None:
            
            #> filling requested data with nans
            out = {'images': np.full((numSource, 5, 2), np.nan), 'mu':   np.full((numSource, 5), np.nan),
                   'srcs':   np.full((numSource, 2), np.nan),    'nims': np.zeros(numSource, dtype=int)}
            out['delays'] = np.full((numSource, 5), np.nan)                     # days
            if observables is not None: out['obs'] = np.full((numSource, len(observables)), np.nan)
            error.highlight(f'Caustic does not exist! bprof={bprof}')
            return out
        
        #> drawing sources inside the caustic
        srcs = sample_sources(caus, numSource, rng=rng)
        
        #> lensing!
        res = find_images(xg, yg, bxg, byg, srcs, bprof)
        
        #> redrawing non-quads
        for _ in range(maxRedraw):
            
            #> finds non-quads
            bad = np.flatnonzero(res['nims'] != 5)
            if len(bad) == 0: break #> all quads
            
            #> redraws bad sources + replaces
            srcs[bad] = sample_sources(caus, len(bad), rng=rng)
            new_res = find_images(xg, yg, bxg, byg, srcs[bad], bprof)
            for i in res: res[i][bad] = new_res[i]
            
    else: #> if sources are provided
        srcs = np.asarray(srcs, dtype=float)
        res = find_images(xg, yg, bxg, byg, srcs, bprof)
        
        
    #>>># LENSING OBSERVABLES #<<<#
    
    #> declarations
    ims  = res['images']     # image positions [arcsec]
    mu   = res['mu']         # image magnifications
    quad = res['nims'] == 5  # quad mask
    
    #> out dictionary (fills with nans if needed)
    out = {'images': np.full_like(ims, np.nan), 
           'mu':     np.full_like(mu, np.nan),
           'srcs':   srcs,                      # sources
           'nims':   res['nims'],               # number of images
           'delays': np.full_like(mu, np.nan)}  # time delays [days] (nan for deflect)
    
    #> if using lens potential (fermat order)
    if method == 'lpot':
        
        #> returns imgaes and mags in fermat order
        o_ims, o_mu, tau = fermat_order(ims[quad], mu[quad], srcs[quad], bprof)
        
        #> gets time delays
        D_dt = (1 + bprof['zl']) * (4 * np.pi * u.G_kpc_solMass) * bprof['angDist']**2 * bprof['sigCrit'] / u.c_kpc**2 # kpc
        out['delays'][quad] = D_dt / u.c_kpc * (tau - tau[:,:1]) / 86400 # days
    
    else: #> if only using deflection angles (morphological order)
        o_ims, o_mu = morph_order(ims[quad], mu[quad])
        
    #> saving images and mags
    out['images'][quad] = o_ims
    out['mu'][quad]     = o_mu
    
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
    galProf     = generate.galProfiles(nfw=True, hern=True, mult=[1,3,4], ex=True)
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
