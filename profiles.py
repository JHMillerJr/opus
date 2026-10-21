#> name: profiles.py
#> author: John Miller Jr
#> descrp: analytic lens profiles for opus (CSE NFW/hernquist, multipoles, external shear): deflections, jacobians, lens potentials

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np

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

