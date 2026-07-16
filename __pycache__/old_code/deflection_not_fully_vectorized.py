#> deflection.py
# calculating deflections w/ GPU

""" #> IMPORTS =======================
================================== """

import os

#> setting path for CUDA (on home desktop)
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin"
cuda_lib = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\libnvvp"
os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
os.environ['PATH'] = cuda_bin + ";" + cuda_lib + ";" + os.environ['PATH']

import cupy as cp

if cp.is_available():     
    pass
else:
    import numpy as cp

# print('-----------------------------')
# print(cp.__name__)
# print(cp.cuda.runtime.getDeviceCount())


""" #> DECLARATIONS ==================
================================== """

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


""" #> DEFLECTION FN =================
================================== """

#> calculates the deflection fields for a batch of galaxies/ profiles
def deflectionGPU(X, Y, batch_profiles):
    
    #> declarations
    arc_rad = 206264.8062471 # arc/rad
    
    #> initializing grids to be computed on
    Ny, Nx = X.shape
    X_pix = cp.asarray(X, dtype=cp.float32) # pix
    Y_pix = cp.asarray(Y, dtype=cp.float32) # pix
    
    #> initializing all deflection grids
    B = len(batch_profiles)
    gradx = cp.zeros((B, Ny, Nx), dtype=cp.float32)
    grady = cp.zeros((B, Ny, Nx), dtype=cp.float32)
    
    print('-----')
    
    #> iterating over each galaxy in batch
    for b, profiles in enumerate(batch_profiles):
        
        # print(b)
        
        #> getting unit conversion
        pix_arc = profiles['pix_arc'] # pix/arc
        # pix_rad = pix_arc * arc_rad   # pix/rad
        
        #> converting grid pix --> arc
        X = X_pix / pix_arc # arcsec
        Y = Y_pix / pix_arc # arcsec
        
        #> NFW - - - - - - -
        for prof in profiles.get('nfw', []):
            
            # NEEDS MODIFIED GRID 

            #> unpacking parameters
            x0 = prof['x0']                       # x center [arcsec]
            y0 = prof['y0']                       # y center [arcsec]
            axisrat = prof['axisrat']             # axis ratio []
            radius = prof['radius']               # char radius [kpc]
            rho0 = prof['rho0']                   # solMass/kpc^3
            theta = prof['theta'] * cp.pi / 180.0 # position angle [deg --> rad]
            
            #> rotating grid (arcsec --> rad)
            mod_value = radius / cp.sqrt(axisrat) / profiles['angDist']
            Xr = ( ( X*cp.cos(theta) + Y*cp.sin(theta)) - x0) / arc_rad / mod_value # arc --> rad, squeeze?
            Yr = ( (-X*cp.sin(theta) + Y*cp.cos(theta)) - y0) / arc_rad / mod_value # arc --> rad, squeeze?
            
            #> getting larger grid array + coefficient array
            Xr_cse = Xr[None, :, :]
            Yr_cse = Yr[None, :, :]
            Ai = NFW_COEFFS[:,0][:, None, None]
            Si = NFW_COEFFS[:,1][:, None, None]
            
            #> calculating relevant values
            axis2 = axisrat**2
            lowp = ( (axis2) * (Si*Si + Xr_cse*Xr_cse) + Yr_cse*Yr_cse)**(0.5)
            upp  = (lowp + Si)**2 + ((1-axis2)*Xr_cse*Xr_cse)
            denom = Si * lowp * upp
            
            #> calculating deflection angles
            scale = (radius/cp.sqrt(axisrat)) * (radius*rho0) / profiles['sigCrit'] / profiles['angDist']
            gx = cp.sum( Ai*(axisrat*Xr_cse)*(lowp + axis2*Si)/denom, axis=0) * scale # axis=0 ?
            gy = cp.sum( Ai*(axisrat*Yr_cse)*(lowp + Si)/denom, axis=0) * scale                 # axis=0 ?
            
            #> rotating deflection angles back + adding
            gradx[b] += gx*cp.cos(theta) - gy*cp.sin(theta)
            grady[b] += gx*cp.sin(theta) + gy*cp.cos(theta)
        
        
        #> HERN - - - - - - -
        for prof in profiles.get('hern', []):
            
            # NEEDS MODIFIED GRID
            
            #> unpacking parameters
            x0 = prof['x0']                       # x center [arcsec]
            y0 = prof['y0']                       # y center [arcsec]
            axisrat = prof['axisrat']             # axis ratio []
            radius = prof['radius']               # char radius [kpc]
            norm = prof['norm']                   # solMass/kpc^3
            # theta = prof['theta'] * cp.pi / 180.0 # position angle [deg --> rad]
            # for the 7D project, the semi-major axis of the hernquist profile 
            # is defined to be along the x-axis (can be done for any obs)
            
            #> shifting & modifying grid & converting (arcsec --> rad)
            mod_value = radius / cp.sqrt(axisrat) / profiles['angDist']
            Xr = (X - x0) / arc_rad / mod_value # arc --> rad, squeeze?
            Yr = (Y - y0) / arc_rad / mod_value # arc --> rad, squeeze?
            
            #> getting larger grid array + coefficient array
            Xr_cse = Xr[None, :, :]
            Yr_cse = Yr[None, :, :]
            Ai = HERN_COEFFS[:,0][:, None, None]
            Si = HERN_COEFFS[:,1][:, None, None]
            
            #> calculating relevant values
            axis2 = axisrat**2
            lowp = ( (axis2) * (Si*Si + Xr_cse*Xr_cse) + Yr_cse*Yr_cse)**(0.5)
            upp  = (lowp + Si)**2 + ((1-axis2)*Xr_cse*Xr_cse)
            denom = Si * lowp * upp
            
            #> calculating deflection angles
            scale = (radius/cp.sqrt(axisrat)) * (radius*norm) / profiles['sigCrit'] / profiles['angDist']
            gradx[b] += cp.sum( Ai*(axisrat*Xr_cse)*(lowp + axis2*Si)/denom, axis=0 ) * scale # axis=0 ?
            grady[b] += cp.sum( Ai*(axisrat*Yr_cse)*(lowp + Si)/denom, axis=0 ) * scale                 # axis=0 ?
        
        
        #> MULTIPOLES - - - - - - -
        for mp in profiles.get('mult', []):
            
            #> unpacking parameters
            x0 = mp['x0']               # x center [arcsec]
            y0 = mp['y0']               # y center [arcsec]
            m = mp['m']                 # multipole number (1,3,4,...)
            norm = mp['norm']           # normalization []?
            slope = mp['slope']         # slope []
            pa = mp['theta']*cp.pi/180  # position angle [deg --> rad]
            
            #> translating grid
            Xr = (X - x0)
            Yr = (Y - y0)
            
            #> calculating polar coords
            r = cp.sqrt(Xr**2 + Yr**2)
            theta_grid = cp.arctan2(Yr, Xr)
            
            #> calculating other shit
            C = - norm/m * r**(slope-2)
            phi = m*(theta_grid - pa)
            
            #> calculating angles
            cos_term = cp.cos(phi)
            sin_term = cp.sin(phi)
            
            #> calculating deflection angles
            gradx[b] += C * ( Xr * slope * cos_term + Yr * m * sin_term) / arc_rad
            grady[b] += C * ( Yr * slope * cos_term - Xr * m * sin_term) / arc_rad
            
            # DO I HAVE TO ROTATE THEM BACK?... NO I THINK THIS ALREADY DOES?
        
        
        #> EXTERNAL SHEAR - - - - - - -
        for prof in profiles.get('ex', []):
            
            #> unpacking parameters
            norm = prof['norm']               # []
            theta_g = prof['theta']*cp.pi/180 # [deg --> rad]
            
            #> converting grid
            Xr = X / arc_rad # arcsec --> rad
            Yr = Y / arc_rad # arcsec --> rad
            
            #> calculating some more shit
            dphi = 2*(cp.arctan2(Yr, Xr) - theta_g)
            
            #> fucking angles or whatever
            c2 = cp.cos(dphi)
            s2 = cp.sin(dphi)
            
            #> calculating deflection angles
            gradx[b] += norm*(Xr*c2 + Yr*s2)
            grady[b] += norm*(Yr*c2 - Xr*s2)
    
    return gradx, grady
