#> name: lpot.py
#> author: John Miller Jr
#> descrp: calculates the lensing potential for nfw, hernquist, multipoles, and ex for opus

""" #> LENSING POTENTIAL =============
================================== """

#> main function
def potential(X, Y, batch_profiles):

    #> imports
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
    B = len(batch_profiles)   # number of galaxies
    Ny, Nx = X.shape          # shape of lpot map

    #> changing grid dtype
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    #> initializing lpot grid
    lpot = np.zeros((B, Ny, Nx), dtype=np.float32)

    #> getting per galaxy parameters
    angDist = np.array([p['angDist'] for p in batch_profiles], dtype=np.float32)
    sigCrit = np.array([p['sigCrit'] for p in batch_profiles], dtype=np.float32)
    pix_arc = np.array([p['pix_arc'] for p in batch_profiles], dtype=np.float32)

    #> reformatting size for vectorization
    angDist = angDist[:,None,None]
    sigCrit = sigCrit[:,None,None]
    pix_arc = pix_arc[:,None,None]


    ###############
    #>>># NFW #<<<#
    ###############
    
    #> getting max number of NFW profiles
    nfw_max = max(len(p.get('nfw', [])) for p in batch_profiles)

    #> iterating thru each NFW profile
    for i in range(nfw_max):

        #> unpacking each of the NFW params
        x0      = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('x0', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        y0      = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('y0', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        axisrat = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('axisrat', 1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        radius  = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('radius', 1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        rho0    = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('rho0', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        theta   = np.array( [p.get('nfw',[{}]*nfw_max )[i].get('theta', 0)*np.pi/180 for p in batch_profiles], dtype=np.float32)[:,None,None]

        #> declarations
        ct = np.cos(theta)
        st = np.sin(theta)
        axis2 = axisrat**2
        
        #> translating, rotating, and converting grid
        denom = ( radius / np.sqrt(axisrat) / angDist)
        Xr = (( X[None,:,:]*ct + Y[None,:,:]*st) / pix_arc - x0) / arc_rad / denom  # pix --> rad
        Yr = ((-X[None,:,:]*st + Y[None,:,:]*ct) / pix_arc - y0) / arc_rad / denom  # pix --> rad
        
        #> (more) declarations
        Xr2 = Xr*Xr

        #> main nfw lp array
        lp = np.zeros_like(Xr, dtype=np.float32)
        
        #> iterating thru each nfw coefficent
        for Ai, Si in NFW_COEFFS:

            #> small psi
            small_psi = np.sqrt(axis2*(Si*Si + Xr2) + (Yr*Yr))
            
            #> large psi
            large_psi = (small_psi + Si)*(small_psi + Si) + Xr2*(1-axis2)
            
            #> psi_cse
            q_Si = axisrat/Si
            psi = (q_Si/2)*np.log(large_psi) - q_Si*np.log( (1+axisrat)*Si )

            lp += Ai * (psi)

        #> sclaing lensing potential
        r0_prime2 = (radius * radius) / axisrat
        scale = r0_prime2 * (radius*rho0) / sigCrit / angDist # do i need to divide by angDist here??

        #> adding to main lpot object (do i need to rotate back for lpot??)
        lpot += lp * scale
        # lpot += gx*ct - gy*st


    ################
    #>>># HERN #<<<#
    ################
    
    #> getting max number of hernquist profiles
    hern_max = max(len(p.get('hern', [])) for p in batch_profiles)

    #> iterating thru each Hern profile
    for i in range(hern_max):

        #> unpacking each of the Hern params
        x0      = np.array( [p.get('hern',[{}]*hern_max )[i].get('x0', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        y0      = np.array( [p.get('hern',[{}]*hern_max )[i].get('y0', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        axisrat = np.array( [p.get('hern',[{}]*hern_max )[i].get('axisrat', 1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        radius  = np.array( [p.get('hern',[{}]*hern_max )[i].get('radius', 1) for p in batch_profiles], dtype=np.float32)[:,None,None]
        norm    = np.array( [p.get('hern',[{}]*hern_max )[i].get('norm', 0) for p in batch_profiles], dtype=np.float32)[:,None,None]
        theta   = np.array( [p.get('hern',[{}]*hern_max )[i].get('theta', 0)*np.pi/180 for p in batch_profiles], dtype=np.float32)[:,None,None]
        
        #> declarations
        ct = np.cos(theta)
        st = np.sin(theta)
        axis2 = axisrat**2
        
        #> translating, rotating, and converting grid
        denom = ( radius / np.sqrt(axisrat) / angDist)
        Xr = (( X[None,:,:]*ct + Y[None,:,:]*st) / pix_arc - x0) / arc_rad / denom  # pix --> rad
        Yr = ((-X[None,:,:]*st + Y[None,:,:]*ct) / pix_arc - y0) / arc_rad / denom  # pix --> rad

        #> main hern lp array
        lp = np.zeros_like(Xr, dtype=np.float32)

        #> iterating thru each hern coefficient
        for Ai, Si in HERN_COEFFS:
            
            #> small psi
            small_psi = np.sqrt(axis2*(Si*Si + Xr2) + (Yr*Yr))
            
            #> large psi
            large_psi = (small_psi + Si)*(small_psi + Si) + Xr2*(1-axis2)
            
            #> psi_cse
            q_Si = axisrat/Si
            psi = (q_Si/2)*np.log(large_psi) - q_Si*np.log( (1+axisrat)*Si )

            lp += Ai * (psi)

        #> sclaing lensing potential
        r0_prime2 = (radius * radius) / axisrat
        scale = r0_prime2 * (radius*norm) / sigCrit / angDist

        #> adding to main lpot object
        lpot += lp * scale

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
        gradx += -norm*(Xr*c2 + Yr*s2)
        grady += -norm*(Yr*c2 - Xr*s2)
            
    return lpot, scaled_lpot


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__),'\n')
    
    #> imports
    import params
    import generate
    import deflectionCPU2 as deflection
    import numpy as np
    
    #> declarations
    numGals = 1
    
    #> grid
    X, Y = generate.grid(nph=50)
    
    #> preping galaxy params
    nfw = True
    hern = False
    mult = []
    ex = False
    galProf = generate.galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)
    
    #> getting bprofiles
    paramRanges = params.toggleParams(galProf)
    batch_profiles = params.bprofiles(numGals, ranges=paramRanges)
    
    #> running
    lpot, scaled_lpot = potential(X, Y, batch_profiles)
    
    #> calculating first derivatives
    lgrady, lgradx = np.gradient(lpot, axis=(1,2))
    
    #> getting the deflection angles
    gradx, grady, lamt = deflection.deflection(X, Y, batch_profiles)
    
    print(lgradx[0])
    print(gradx[0])
    
    # end
# thank