#> name: cosmology.py
#> author: John Miller Jr
#> descrp: calculates various cosmological params for lensing

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import scipy as sp

#> modules
from modules.units import u; u=u()


""" #> COSMOLOGY =====================
================================== """

#> calculates angular diameter distance (Mpc) between z_1 and z_2
def angDist(z1, z2, cosmo=u.cosmo):
    """
    z1 [float]: first redshift
    
    z2 [float]: second redshift

    cosmo [dict, opt.]: cosmo params
    
    returns: D12 # Mpc
    """

    #> dimensionless friedman eq; 2023 Birrer Eq. 12
    #! technically 1/E
    E = lambda z: 1 / ( cosmo['omega_m_0']*(1 + z)**3 + 
                        cosmo['omega_lam_0'] )**0.5
    
    #> comoving distance; 2023 Birrer Eq. 10 & 11
    X_1 = (u.c_km/cosmo['h0']) * sp.integrate.quad(E, 0, z1)[0]
    X_2 = (u.c_km/cosmo['h0']) * sp.integrate.quad(E, 0, z2)[0]
    
    #> angular diameter distance; 2024 Saha Eq. 19
    D12 = ( X_2 - X_1 ) / ( 1 + z2 )
    
    return D12 # Mpc


#> returns the comoving distance between two redshifts
def comovingDist(z1, z2, cosmo=u.cosmo):
    """
    z1 [float]: first redshift
    
    z2 [float]: second redshift

    cosmo [dict, opt.]: cosmo params
    
    returns: X_1 # Mpc
    """

    #> dimensionless friedman eq; 2023 Birrer Eq. 12
    E = lambda z: 1 / ( cosmo['omega_m_0']*(1 + z)**3 + 
                        cosmo['omega_lam_0'] )**0.5
    
    #> comoving distance; 2023 Birrer Eq. 10 & 11
    X_1 = (u.c_km/cosmo['h0']) * sp.integrate.quad(E, z1, z2)[0]
    
    return X_1 # Mpc


#> returns hubble constant at given redshift
def hubble(z, cosmo=u.cosmo):
    """
    z [float]: redshift

    cosmo [dict, opt.]: cosmo params
    
    returns: H(z) # 1 / s
    """

    #> hubble constant at a given redshift (km/s/Mpc)
    Hz = cosmo['h0'] * ( cosmo['omega_m_0']*(1 + z)**3 + cosmo['omega_lam_0'] )**0.5 # km/s/Mpc

    return Hz / u.km_mpc # 1/s


#> calculates kappa constant
def sigmaCrit(zd, zs, cosmo=u.cosmo):
    """
    zd [float]: lens redshift
    
    zs [float]: source redshift

    cosmo [dict, opt.]: cosmo params
    
    returns: sigmaCrit # solMass / kpc^2
    """
    
    #> angular diameter distances
    Dd  = angDist( 0, zd, cosmo)
    Ds  = angDist( 0, zs, cosmo)
    Dds = angDist(zd, zs, cosmo)
    
    #> sigmaCrit
    kC = ( u.c_cm**2 * Ds ) / ( 4*np.pi*u.G * Dds * Dd * u.cm_mpc ) # g/cm^2
    
    #> converting from g/cm^2 --> solMass/kpc^2
    kC /= u.solMass      # g/cm^2 --> solMass/cm^2
    kC *= (u.cm_kpc**2)  # solMass/cm^2 --> solMass/kpc^2
    
    return kC # solMass/kpc^2


#> calculates the critical density at a given redshift
def critDensity(z, cosmo=u.cosmo):
    """
    z [float]: redshift

    cosmo [dict, opt.]: cosmo params
    
    returns: rhoc # solMass / kpc^3
    """
    
    #> hubble param at redshift
    Hz = hubble(z, cosmo)
    
    #> rho crit
    rhoc = (3 * Hz**2) / (8*np.pi*u.G) # g/cm^3
    
    #> converting
    rhoc /= u.solMass     # g/cm^3 --> solMass/cm^3
    rhoc *= (u.cm_kpc**3) # solMass/cm^3 --> solMass/kpc^3

    return rhoc # solMass/kpc^3


#> calculates the einstein radius for a given lens M, zl, zs
def einRadius(logM, zd, zs, cosmo=u.cosmo):

    #> angular diameter distances
    Dd  = angDist( 0, zd, cosmo) # Mpc
    Ds  = angDist( 0, zs, cosmo) # Mpc
    Dds = angDist(zd, zs, cosmo) # Mpc
    
    #> fractions
    frac1 = ( (4 * u.G_kpc_solMass * (10**logM) ) / u.c_kpc**2 ) / 1e3 # Mpc
    ER = ( (frac1 * Dds) / (Ds * Dd) )**0.5 * u.arc_rad # arcsecond

    return ER # arcsec
        

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    
    
    # end
# thank