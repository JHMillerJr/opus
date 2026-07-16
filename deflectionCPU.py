#> deflection.py
# calculating deflections w/ GPU

""" #> IMPORTS =======================
================================== """

#> if wanting to supress warnings
# import warnings
# The regex '.*' ensures it matches the full message even if it has dynamic parts.
# warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*", category=UserWarning)
    

""" #> DEFLECTION FN =================
================================== """

#> calculates the deflection fields for a batch of galaxies/ profiles
def deflection(X, Y, batch_profiles, gpu=False):
    
    import numpy as cp
    
    #> NFW CSE coefficients (16; lower accuracy); 10^-2 in kappa
    NFW_COEFFS = cp.array([
        [1.434960e-16, 4.041628e-06], [5.232413e-14, 3.086267e-05],
        [2.666660e-12, 1.298542e-04], [7.961761e-11, 4.131977e-04],
        [2.306895e-09, 1.271373e-03], [6.742968e-08, 3.912641e-03],
        [1.991691e-06, 1.208331e-02], [5.904388e-05, 3.740521e-02],
        [1.693069e-03, 1.153247e-01], [4.039850e-02, 3.472038e-01],
        [5.665072e-01, 1.017550e+00], [3.683242e+00, 3.253031e+00],
        [1.582481e+01, 1.190315e+01], [6.340984e+01, 4.627701e+01],
        [2.576763e+02, 1.842613e+02], [1.422619e+03, 8.206569e+02]
        ], dtype=cp.float32)

    #> hernquist CSE coefficients (13; lower accuracy); 10^-2 in kappa
    HERN_COEFFS = cp.array([
        [7.775712e-16, 4.426947e-06], [3.279878e-13, 3.551219e-05],
        [2.374931e-11, 1.639271e-04], [1.137151e-09, 6.047024e-04],
        [5.314908e-08, 2.180958e-03], [2.466940e-06, 7.843573e-03],
        [1.125917e-04, 2.809420e-02], [4.700637e-03, 9.893403e-02],
        [1.257748e-01, 3.246017e-01], [9.744152e-01, 9.409923e-01],
        [1.434502e+00, 2.929948e+00], [5.548243e-01, 1.545137e+01],
        [6.123431e-01, 1.883671e+03]
        ], dtype=cp.float32)
    
    #> declarations
    arc_rad = 206264.8062471  # arcsec/rad
    B = len(batch_profiles)   # num total galaxies
    
    #> initializing grids to be computed on
    Ny, Nx = X.shape
    X = cp.asarray(X, dtype=cp.float32) # pix
    Y = cp.asarray(Y, dtype=cp.float32) # pix

    #> initializing deflection angle grids
    gradx = cp.zeros((B, Ny, Nx), dtype=cp.float32)
    grady = cp.zeros((B, Ny, Nx), dtype=cp.float32)

    #> cosmological params + pix-->arc conversion (per galaxy!)
    angDist = cp.array( [ p['angDist'] for p in batch_profiles ], dtype=cp.float32)[:, None, None]
    sigCrit = cp.array( [ p['sigCrit'] for p in batch_profiles ], dtype=cp.float32)[:, None, None]
    pix_arc = cp.array( [ p['pix_arc'] for p in batch_profiles ], dtype=cp.float32)[:, None, None]

    #> NFW - - - - - - -
    nfw_max = max(len(p.get('nfw', [])) for p in batch_profiles) # gets max number of nfw profiles for all galaxies
    for idx in range(nfw_max):
        
        #> unpacking profile parameters
        x0 =      cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('x0',0)      for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        y0 =      cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('y0',0)      for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        axisrat = cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('axisrat',1) for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        radius =  cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('radius',1)  for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        rho0 =    cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('rho0',0)    for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        theta =   cp.array( [ p.get('nfw',[{}]*nfw_max)[idx].get('theta',0)*cp.pi/180 for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        
        #> rotating, translating, and converting grid
        denom = ( radius / cp.sqrt(axisrat) / angDist)
        Xr = (( X[None,:,:]*cp.cos(theta) + Y[None,:,:]*cp.sin(theta)) / pix_arc - x0) / arc_rad / denom  # pix --> rad
        Yr = ((-X[None,:,:]*cp.sin(theta) + Y[None,:,:]*cp.cos(theta)) / pix_arc - y0) / arc_rad / denom  # pix --> rad
        
        #> getting CSE coeffs
        Ai = NFW_COEFFS[:,0][:,None,None,None]
        Si = NFW_COEFFS[:,1][:,None,None,None]
        
        #> expanding grids
        Xr_cse = Xr[None,:,:,:]
        Yr_cse = Yr[None,:,:,:]
        
        #> calculating CSE profiles
        axis2 = axisrat**2
        lowp = cp.sqrt( axis2*(Si**2 + Xr_cse**2) + Yr_cse**2 )
        upp = (lowp + Si)**2 + (1 - axis2)*Xr_cse**2
        denom = Si*lowp*upp
        
        #> calculating deflection angles + summing CSE profiles
        scale = (radius / cp.sqrt(axisrat)) * (radius*rho0) / sigCrit / angDist
        gx = cp.sum( Ai*(axisrat*Xr_cse) * (lowp + axis2*Si) / denom, axis=0) * scale
        gy = cp.sum( Ai*(axisrat*Yr_cse) * (lowp + Si) / denom, axis=0) * scale

        #> rotating deflection angles back + adding to total deflection angles
        gradx += gx*cp.cos(theta) - gy*cp.sin(theta)
        grady += gx*cp.sin(theta) + gy*cp.cos(theta)

    #> HERN - - - - - - -
    hern_max = max(len(p.get('hern', [])) for p in batch_profiles) # gets max number of hern profiles for all galaxies
    for idx in range(hern_max):
        
        #> unpacking profile parameters
        x0 =      cp.array( [ p.get('hern',[{}]*hern_max)[idx].get('x0',0)      for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        y0 =      cp.array( [ p.get('hern',[{}]*hern_max)[idx].get('y0',0)      for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        axisrat = cp.array( [ p.get('hern',[{}]*hern_max)[idx].get('axisrat',1) for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        radius =  cp.array( [ p.get('hern',[{}]*hern_max)[idx].get('radius',1)  for p in batch_profiles ], dtype=cp.float32)[:,None,None]
        norm =    cp.array( [ p.get('hern',[{}]*hern_max)[idx].get('norm',0)    for p in batch_profiles ], dtype=cp.float32)[:,None,None]

        #> converting, translating, converting grid
        denom = (radius / cp.sqrt(axisrat) / angDist)
        Xr = ( X[None,:,:]/pix_arc -x0 ) / arc_rad / denom # pix --> rad
        Yr = ( Y[None,:,:]/pix_arc -y0 ) / arc_rad / denom # pix --> rad

        #> getting CSE coeffs
        Ai = HERN_COEFFS[:,0][:,None,None,None]
        Si = HERN_COEFFS[:,1][:,None,None,None]
        
        #> expanding grids
        Xr_cse = Xr[None,:,:,:]
        Yr_cse = Yr[None,:,:,:]
        
        #> calculating CSE profiles
        axis2 = axisrat**2
        lowp = cp.sqrt( axis2*(Si**2 + Xr_cse**2) + Yr_cse**2 )
        upp = (lowp + Si)**2 + (1 - axis2)*Xr_cse**2
        denom = Si*lowp*upp
        
        #> calculating deflection angles, summing CSE profiles, and adding to total def angles
        scale = (radius / cp.sqrt(axisrat)) * (radius*norm) / sigCrit / angDist
        gradx += cp.sum( Ai*(axisrat*Xr_cse) * (lowp + axis2*Si) / denom, axis=0) * scale
        grady += cp.sum( Ai*(axisrat*Yr_cse) * (lowp + Si) / denom, axis=0) * scale

    #> MULT - - - - - - -
    max_mult = max(len(p.get('mult', [])) for p in batch_profiles) # gets max number of mult profiles for all galaxies
    for idx in range(max_mult):
        
        #> unpacking profile params
        x0 =    cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('x0',0)    for p in batch_profiles], dtype=cp.float32)[:,None,None]
        y0 =    cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('y0',0)    for p in batch_profiles], dtype=cp.float32)[:,None,None]
        m =     cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('m',1)     for p in batch_profiles], dtype=cp.float32)[:,None,None]
        norm =  cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('norm',0)  for p in batch_profiles], dtype=cp.float32)[:,None,None]
        slope = cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('slope',2) for p in batch_profiles], dtype=cp.float32)[:,None,None]
        pa =    cp.array( [ p.get('mult',[{}]*max_mult)[idx].get('theta',0)*cp.pi/180 for p in batch_profiles], dtype=cp.float32)[:,None,None]
        
        #> converting and translating grid
        Xr = (X[None,:,:] / pix_arc) - x0 # pix --> arc
        Yr = (Y[None,:,:] / pix_arc) - y0 # pix --> arc
        
        #> calculating polar coords
        r = cp.sqrt(Xr**2 + Yr**2)
        theta_grid = cp.arctan2(Yr, Xr)
        
        #> other calculating shit
        phi = m*(theta_grid - pa)
        C = -norm/m * r**(slope-2)

        #> calculating deflection angles + adding to total
        gradx += C * (Xr*slope*cp.cos(phi) + Yr*m*cp.sin(phi)) / arc_rad
        grady += C * (Yr*slope*cp.cos(phi) - Xr*m*cp.sin(phi)) / arc_rad


    #> EX - - - - - - -
    max_ex = max(len(p.get('ex', [])) for p in batch_profiles) # gets max number of ex profiles for all galaxies
    for idx in range(max_ex):
        
        #> unpacking profile parameters
        norm = cp.array([p.get('ex',[{}]*max_ex)[idx].get('norm',0) for p in batch_profiles], dtype=cp.float32)[:,None,None]
        theta_g = cp.array([p.get('ex',[{}]*max_ex)[idx].get('theta',0)*cp.pi/180 for p in batch_profiles], dtype=cp.float32)[:,None,None]

        #> converting grid
        Xr = X[None,:,:] / pix_arc / arc_rad # pix --> rad
        Yr = Y[None,:,:] / pix_arc / arc_rad # pix --> rad
        
        #> calculating angle shit
        dphi = 2*(cp.arctan2(Yr,Xr)-theta_g)
        c2 = cp.cos(dphi)
        s2 = cp.sin(dphi)
        
        #> calculating deflection angles + adding to total
        gradx += norm*(Xr*c2 + Yr*s2)
        grady += norm*(Yr*c2 - Xr*s2)
        
    
    #> calculating kappa
    gradxy, gradxx = cp.gradient(gradx * pix_arc * arc_rad, axis=(1,2)) # grad^2 x
    gradyy, gradyx = cp.gradient(grady * pix_arc * arc_rad, axis=(1,2)) # grad^2 y
    kap = 0.5 * (gradxx + gradyy)

    #> calculating gamma
    gamma1 = 0.5 * (gradxx - gradyy)
    gamma = cp.sqrt( gamma1**2 + gradxy**2 )
            
    return gradx, grady, (1-kap-gamma) # gradx, grady, lamt