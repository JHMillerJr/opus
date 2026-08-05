#> name: priors.py
#> author: John Miller Jr
#> descrp: priors for cosmological population of elliptical galaxies

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import scipy as sp

#> modules
import cosmology
from modules.units import u; u=u()
import modules.error as error


""" #> MAIN PRIOR FN =================
================================== """

#> generating the redshift and halo-mass population
def sample_lenses(zobs, nlens, **kwargs):
    
    #> kwargs
    log10Mmin = kwargs.get('log10Mmin', 11.5) # min log10(M/solM)
    log10Mmax = kwargs.get('log10Mmax', 14.5) # max log10(M/solM)
    mdef      = kwargs.get('mdef', 'vir')     # halo mass definition (200c, 200m, etc.)
    ngrid     = kwargs.get('ngrid', 4096)     # number grid
    seed      = kwargs.get('seed', None)     # random seed
    
    #> random-number generator
    rng = np.random.default_rng(seed)
    
    #> sampling redshifts and halo masses
    zlens = sample_redshifts(zobs, nlens, rng)
    halom = halo_mass_fn(zlens,               # array of lens redshifts
                         log10Mmin=log10Mmin, # lower bound of mass range (solM)
                         log10Mmax=log10Mmax, # upper bound of mass range (solM)
                         rng=rng,             # rng seed
                         ngrid=ngrid,         # ln(M/h) grid size
                         mdef=mdef)           # halo mass definition (200c, 200m, etc.)
    
    return zlens, halom


""" #> REDSHIFT ======================
================================== """

#> returns random redshifts
def ranRedshifts(numGals, **kwargs):
    
    ### THESE SHOULD BE INFORMED FROM OBSERVATIONS/ THEORY
    ### I CAN DO A FIRST PASS LOOK W/ EINSTEIN RADIUS...? 
    ### OR LENSING PROBABILITY
    
    ### should be a joint probability distribution
    
    #> imports
    from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN    
    
    #> correlation matrix
    cross_corl = 0.4
    R = ( np.identity(n=2) * (1-cross_corl) ) + cross_corl
    
    #> deflector: lb, ub, mu, sigma
    ub_zl    = kwargs.get('ub_zl', 1.5)
    lb_zl    = kwargs.get('lb_zl', 0.1)
    mu_zl    = kwargs.get('mu_zl', 0.5)
    sigma_zl = kwargs.get('mu_zl', 0.1)
    
    #> source: lb, ub, mu, sigma
    ub_zs    = kwargs.get('ub_zs', 3.0)
    lb_zs    = kwargs.get('lb_zs', 0.5)
    mu_zs    = kwargs.get('mu_zs', 1.0)
    sigma_zs = kwargs.get('mu_zs', 0.1)
    
    #> constructing params
    lb = np.array([lb_zl, lb_zs])
    ub = np.array([ub_zl, ub_zs])
    mu = np.array([mu_zl, mu_zs])
    sigma = np.array([sigma_zl, sigma_zs])
    cov = np.diag(sigma) @ R @ np.diag(sigma) # converting correlation & std vectors to covariance matrix
    
    #> drawing samples
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    samples = tmvn.sample(numGals)
    samples = np.array(samples)
    
    return samples.T


#> sampling lens redshifts from the observed sample
def sample_redshifts(zobs, nlens, rng):
    
    #> cleaning redshift array
    zobs = np.asarray(zobs)
    zobs = zobs[np.isfinite(zobs)]
    
    if len(zobs) == 0: error.phrase('No valid lens redshifts supplied!')
    if any(zobs < 0): error.phrase('Lens redshifts must be non-negative!')
    
    #> bootstrap sampling
    return rng.choice(zobs, nlens, replace=True)


""" #> DENSITY CONTRAST ==============
================================== """

#> returns the density contrast for a virialized halo at redshift z
#> from eq 6 of 1997 Bryan & Norman, and eq 1 of 2014 van den Bosch 
def densityContrast(z, cosmo=u.cosmo):
    
    #> dimensionless friedman eq; 2023 Birrer Eq. 12
    E = lambda z: ( cosmo['omega_m']*(1 + z)**3 + cosmo['omega_lam'] )**0.5
    
    #> omega_m(z); 1997 Bryan & Norman
    omega_m_z = ( cosmo['omega_m'] * (1+z)**3 ) / E(z)**2
    x = omega_m_z - 1
    
    #> density contrast; 1997 Bryan & Norman Eq. 6
    delta_vir_c = 18*(np.pi**2) + 82*x - 39*(x**2)
    
    return delta_vir_c


""" #> HALO MASS FN ==================
================================== """

#> colossus halo mass function
def halo_mass_fn(zlens, cosmo=u.cosmo, **kwargs):
    
    #> imports
    from colossus.cosmology import cosmology as ccos
    
    #> declarations
    ccosmo = ccos.setCosmology('myCosmo', 
                               params=ccos.cosmologies['planck18'], # initial base cosmology (keeps Ob0, sigma8, and ns)
                               H0 =cosmo['h0'],            # H0
                               Om0=cosmo['omega_m_0'],     # omega_m_0
                               relspecies=True,            # allows for non-zero omega_gamma
                               flat=True)                  # ignores Ode0
    # colossus.cosmology.setCosmology('planck18')
    
    #> reduced hubble param
    h = ccosmo.H0 / 100
    
    #> kwargs
    log10Mmin = kwargs.get('log10Mmin', 11.5) # min log10(M/solM)
    log10Mmax = kwargs.get('log10Mmax', 14.5) # max log10(M/solM)
    mdef      = kwargs.get('mdef', 'vir')     # halo mass definition (200c, 200m, etc.)
    ngrid     = kwargs.get('ngrid', 4096)     # number grid
    rng       = kwargs.get('rng', np.random.default_rng(None)) # U random generator

    #> iterating through lens redshifts
    halom = np.zeros(len(zlens))
    for i, zl in enumerate(zlens):

        #> halo cdf at lens redshift
        lnm, cdf = halo_cdf(zl,                  # lens redshift
                            h=h,                 # H0/100
                            log10Mmin=log10Mmin, # min log10(M/solM)
                            log10Mmax=log10Mmax, # max log10(M/solM)
                            mdef=mdef,           # halo mass definition
                            ngrid=ngrid)         # number grid
        
        #> inverse-cdf sampling
        #> interps cdf based on U random #, with lnm values, then gets solM/h
        halom_h  = np.exp(np.interp(rng.random(), cdf, lnm)) # solM/h
        halom[i] = halom_h / h # solMass
    
    return halom


#> obtaining colossus mass function and cdf
def halo_cdf(zlens, h, log10Mmin=11.5, log10Mmax=14.5, mdef='vir', ngrid=4096):
    
    #> imports
    from scipy.integrate import cumulative_trapezoid
    from colossus.lss import mass_function
    
    #> logarithmic mass grid
    lnMmin = np.log(h * 10**log10Mmin)       # changing to solM/h and base e
    lnMmax = np.log(h * 10**log10Mmax)       # changing to solM/h and base e
    lnm = np.linspace(lnMmin, lnMmax, ngrid) # grid in base e
    mass_h = np.exp(lnm)                     # changing to Msol/h
    
    #> mass function ( returns dn/d[ln(solM/h)] per ln(solM/h) )
    dndlnm = mass_function.massFunction(mass_h,            # solM/h bins
                                        zlens,             # lens redshift
                                        q_in='M',          # in mass = solM/h (halo masses)
                                        q_out='dndlnM',    # out mass = dn/dlnM(solM/h)
                                        mdef=mdef,         # halo mass definition
                                        model='despali16', # model for mass function
                                        ellipsoidal=False) # If True, return the results for an ellipsoidal halo finder, otherwise standard SO.
    
    #> removing invalid numerical values (NaN, np.inf, -np.inf)
    dndlnm = np.asarray(dndlnm)
    dndlnm[~np.isfinite(dndlnm)] = 0
    dndlnm[dndlnm < 0] = 0
    
    #> integrating in ln(mass)
    cdf = cumulative_trapezoid(dndlnm, lnm, initial=0) # integrating full curve
    if cdf[-1] <= 0: error.phrase('HMF is zero over the selected mass range!')
    cdf /= cdf[-1] # normalizing based on total comoving numer density of halos in the selected mass interval
                   # i.e., normalizing between (0, 1)
    
    #> removing repeated values in the high-mass tail
    #> i.e., keeping only positively increasing cdf values
    keep = np.concatenate(([True], np.diff(cdf) > 0)) 
    lnm, cdf = lnm[keep], cdf[keep]
    cdf[-1] = 1
        
    return lnm, cdf # ln(solM/h), normalized cdf


""" #> HMASS-C REL ===================
================================== """

#> mass-concentration_vir relation
#> taken from Table 3 of 2014 Dutton & Maccio, plus eq 12 & 13
#> w/ Planck cosmology
def mass_c_vir_rel(M_vir, zl, h):
    
    #> declarations
    M = (M_vir * h / (1e12))
    
    #> getting coeffs (for c_vir vs. M_vir)
    a = 0.537 + ( (1.025-0.537) * np.exp(-0.718 * zl**(1.08)) )
    b = -0.097 + 0.024*zl
    
    #> mass-concentration relation
    log10_c_vir = a + ( b * np.log10(M) )
    c_vir = 10**(log10_c_vir)
    
    return c_vir


#> mass-concentration_200 relation
#> taken from Table 3 of 2014 Dutton & Maccio, plus eq 10 & 11
#> w/ Planck cosmology
def mass_c_200_rel(M_200, zl, h):
    
    #> declarations
    M = (M_200 * h / (1e12))
    
    #> getting coeffs (for c_vir vs. M_vir)
    a = 0.520 + ( (0.905 - 0.520) * np.exp(-0.617 * zl**(1.21)) )
    b = -0.101 + 0.026*zl
    
    #> mass-concentration relation
    log10_c_200 = a + ( b * np.log10(M) )
    c_200 = 10**(log10_c_200)
    
    return c_200


""" #> STELLAR-M-H REL ===============
================================== """

#> stellar-to-halo mass relation
def stellar_h_mass_rel():
    
    
    
    return


""" #> STELLAR-M-S REL ===============
================================== """

#> stellar mass-size relation
def mass_r_relation():
    
    
    
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    zlens = np.linspace(0.3, 1.0, 10)
    masses = halo_mass_fn(zlens=zlens)
    
    print(zlens, masses)
    
    # end
# thank