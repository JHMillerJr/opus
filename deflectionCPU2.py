def deflection(X, Y, batch_profiles):

    import numpy as np
    
    #> NFW CSE coefficients (16; lower accuracy); 10^-2 in kappa
    NFW_COEFFS = np.array([
        [1.434960e-16, 4.041628e-06], [5.232413e-14, 3.086267e-05],
        [2.666660e-12, 1.298542e-04], [7.961761e-11, 4.131977e-04],
        [2.306895e-09, 1.271373e-03], [6.742968e-08, 3.912641e-03],
        [1.991691e-06, 1.208331e-02], [5.904388e-05, 3.740521e-02],
        [1.693069e-03, 1.153247e-01], [4.039850e-02, 3.472038e-01],
        [5.665072e-01, 1.017550e+00], [3.683242e+00, 3.253031e+00],
        [1.582481e+01, 1.190315e+01], [6.340984e+01, 4.627701e+01],
        [2.576763e+02, 1.842613e+02], [1.422619e+03, 8.206569e+02]
        ], dtype=np.float32)

    #> hernquist CSE coefficients (13; lower accuracy); 10^-2 in kappa
    HERN_COEFFS = np.array([
        [7.775712e-16, 4.426947e-06], [3.279878e-13, 3.551219e-05],
        [2.374931e-11, 1.639271e-04], [1.137151e-09, 6.047024e-04],
        [5.314908e-08, 2.180958e-03], [2.466940e-06, 7.843573e-03],
        [1.125917e-04, 2.809420e-02], [4.700637e-03, 9.893403e-02],
        [1.257748e-01, 3.246017e-01], [9.744152e-01, 9.409923e-01],
        [1.434502e+00, 2.929948e+00], [5.548243e-01, 1.545137e+01],
        [6.123431e-01, 1.883671e+03]
        ], dtype=np.float32)

    #> declarations
    arc_rad = 206264.8062471  # arcsec/rad
    B = len(batch_profiles)
    Ny, Nx = X.shape

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    gradx = np.zeros((B, Ny, Nx), dtype=np.float32)
    grady = np.zeros((B, Ny, Nx), dtype=np.float32)

    # -------------------------
    # extract common arrays once
    # -------------------------
    angDist = np.array([p['angDist'] for p in batch_profiles], dtype=np.float32)
    sigCrit = np.array([p['sigCrit'] for p in batch_profiles], dtype=np.float32)
    pix_arc = np.array([p['pix_arc'] for p in batch_profiles], dtype=np.float32)

    angDist = angDist[:,None,None]
    sigCrit = sigCrit[:,None,None]
    pix_arc = pix_arc[:,None,None]

    # =========================================================
    # NFW BLOCK (STREAMED, NO 4D TENSORS)
    # =========================================================
    nfw_max = max(len(p.get('nfw', [])) for p in batch_profiles)

    for i in range(nfw_max):

        x0 = np.array([p.get('nfw',[{}]*nfw_max)[i].get('x0',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        y0 = np.array([p.get('nfw',[{}]*nfw_max)[i].get('y0',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        axisrat = np.array([p.get('nfw',[{}]*nfw_max)[i].get('axisrat',1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        radius = np.array([p.get('nfw',[{}]*nfw_max)[i].get('radius',1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        rho0 = np.array([p.get('nfw',[{}]*nfw_max)[i].get('rho0',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        theta = np.array([p.get('nfw',[{}]*nfw_max)[i].get('theta',0)*np.pi/180 for p in batch_profiles], dtype=np.float32)[:,None,None]

        ct = np.cos(theta)
        st = np.sin(theta)
        
        denom = ( radius / np.sqrt(axisrat) / angDist)
        Xr = (( X[None,:,:]*ct + Y[None,:,:]*st) / pix_arc - x0) / arc_rad / denom  # pix --> rad
        Yr = ((-X[None,:,:]*st + Y[None,:,:]*ct) / pix_arc - y0) / arc_rad / denom  # pix --> rad

        axis2 = axisrat**2
        gx = np.zeros_like(Xr, dtype=np.float32)
        gy = np.zeros_like(Yr, dtype=np.float32)
        for Ai, Si in NFW_COEFFS:

            lowp = np.sqrt(axis2*(Si*Si + Xr*Xr) + Yr*Yr)
            upp = (lowp + Si)**2 + (1 - axis2)*Xr*Xr
            denom = Si * lowp * upp

            gx += Ai * (axisrat*Xr) * (lowp + axis2*Si) / denom
            gy += Ai * (axisrat*Yr) * (lowp + Si) / denom

        scale = (radius/np.sqrt(axisrat)) * (radius*rho0) / sigCrit / angDist
        gx *= scale
        gy *= scale

        gradx += gx*ct - gy*st
        grady += gx*st + gy*ct

    # =========================================================
    # HERN BLOCK (same pattern)
    # =========================================================
    hern_max = max(len(p.get('hern', [])) for p in batch_profiles)

    for i in range(hern_max):

        x0 = np.array([p.get('hern',[{}]*hern_max)[i].get('x0',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        y0 = np.array([p.get('hern',[{}]*hern_max)[i].get('y0',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        axisrat = np.array([p.get('hern',[{}]*hern_max)[i].get('axisrat',1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        radius = np.array([p.get('hern',[{}]*hern_max)[i].get('radius',1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        norm = np.array([p.get('hern',[{}]*hern_max)[i].get('norm',0) for p in batch_profiles], dtype=np.float32)[:,None,None]

        #> converting, translating, converting grid
        denom = (radius / np.sqrt(axisrat) / angDist)
        Xr = ( X[None,:,:]/pix_arc -x0 ) / arc_rad / denom # pix --> rad
        Yr = ( Y[None,:,:]/pix_arc -y0 ) / arc_rad / denom # pix --> rad

        axis2 = axisrat**2

        scale = (radius/np.sqrt(axisrat)) * (radius*norm) / sigCrit / angDist

        gx = np.zeros_like(Xr, dtype=np.float32)
        gy = np.zeros_like(Yr, dtype=np.float32)

        for Ai, Si in HERN_COEFFS:

            lowp = np.sqrt(axis2*(Si*Si + Xr*Xr) + Yr*Yr)
            upp = (lowp + Si)**2 + (1-axis2)*Xr*Xr
            denom = Si * lowp * upp + 1e-12

            gx += Ai*(axisrat*Xr)*(lowp + axis2*Si)/denom
            gy += Ai*(axisrat*Yr)*(lowp + Si)/denom

        gradx += gx * scale
        grady += gy * scale
        
    #> MULT - - - - - - -
    max_mult = max(len(p.get('mult', [])) for p in batch_profiles) # gets max number of mult profiles for all galaxies
    for idx in range(max_mult):
        
        #> unpacking profile params
        x0 =    np.array( [ p.get('mult',[{}]*max_mult)[idx].get('x0',0)    for p in batch_profiles], dtype=np.float32)[:,None,None]
        y0 =    np.array( [ p.get('mult',[{}]*max_mult)[idx].get('y0',0)    for p in batch_profiles], dtype=np.float32)[:,None,None]
        m =     np.array( [ p.get('mult',[{}]*max_mult)[idx].get('m',1)     for p in batch_profiles], dtype=np.float32)[:,None,None]
        norm =  np.array( [ p.get('mult',[{}]*max_mult)[idx].get('norm',0)  for p in batch_profiles], dtype=np.float32)[:,None,None]
        slope = np.array( [ p.get('mult',[{}]*max_mult)[idx].get('slope',2) for p in batch_profiles], dtype=np.float32)[:,None,None]
        pa =    np.array( [ p.get('mult',[{}]*max_mult)[idx].get('theta',0)*np.pi/180 for p in batch_profiles], dtype=np.float32)[:,None,None]
        
        #> converting and translating grid
        Xr = (X[None,:,:] / pix_arc) - x0 # pix --> arc
        Yr = (Y[None,:,:] / pix_arc) - y0 # pix --> arc
        
        #> calculating polar coords
        r = np.sqrt(Xr**2 + Yr**2)
        theta_grid = np.arctan2(Yr, Xr)
        
        #> other calculating shit
        phi = m*(theta_grid - pa)
        C = -norm/m * r**(slope-2)

        #> calculating deflection angles + adding to total
        gradx += C * (Xr*slope*np.cos(phi) + Yr*m*np.sin(phi)) / arc_rad
        grady += C * (Yr*slope*np.cos(phi) - Xr*m*np.sin(phi)) / arc_rad


    #> EX - - - - - - -
    max_ex = max(len(p.get('ex', [])) for p in batch_profiles) # gets max number of ex profiles for all galaxies
    for idx in range(max_ex):
        
        #> unpacking profile parameters
        norm = np.array([p.get('ex',[{}]*max_ex)[idx].get('norm',0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        theta_g = np.array([p.get('ex',[{}]*max_ex)[idx].get('theta',0)*np.pi/180 for p in batch_profiles], dtype=np.float32)[:,None,None]

        #> converting grid
        Xr = X[None,:,:] / pix_arc / arc_rad # pix --> rad
        Yr = Y[None,:,:] / pix_arc / arc_rad # pix --> rad
        
        #> calculating angle shit
        dphi = 2*(np.arctan2(Yr,Xr)-theta_g)
        c2 = np.cos(dphi)
        s2 = np.sin(dphi)
        
        #> calculating deflection angles + adding to total
        gradx += norm*(Xr*c2 + Yr*s2)
        grady += norm*(Yr*c2 - Xr*s2)

    #> calculating kappa
    gradxy, gradxx = np.gradient(gradx * pix_arc * arc_rad, axis=(1,2)) # grad^2 x
    gradyy, gradyx = np.gradient(grady * pix_arc * arc_rad, axis=(1,2)) # grad^2 y
    kap = 0.5 * (gradxx + gradyy)

    #> calculating gamma
    gamma1 = 0.5 * (gradxx - gradyy)
    gamma = np.sqrt( gamma1**2 + gradxy**2 )
            
    return gradx, grady, (1-kap-gamma) # gradx, grady, lamt