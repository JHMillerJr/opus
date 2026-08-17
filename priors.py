#> name: priors.py
#> author: John Miller Jr
#> descrp: priors for cosmological population of elliptical galaxies

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

#> modules
import cosmology
from modules.units import u; u=u()
import modules.error as error
from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN   

#> colossus
from colossus.lss import mass_function
from colossus.cosmology import cosmology as ccos


""" #> DRAWING SAMPLES =================
================================== """

#> sampling from the priors
def sample_priors(numgals, priorDict, cosmo=u.cosmo, **kwargs):
    
    #> kwargs
    zls       = kwargs.get('zls', None)       # list of lens redshifts
    log10Mmin = kwargs.get('log10Mmin', 11.5) # min log10(M/solM)
    log10Mmax = kwargs.get('log10Mmax', 14.5) # max log10(M/solM)
    mdef      = kwargs.get('mdef', 'vir')     # halo mass definition (200c, 200m, etc.)
    ngrid     = kwargs.get('ngrid', 4096)     # number grid
    seed      = kwargs.get('seed', None)      # random seed
    
    #> setting random seed
    np.random.seed(seed)
    
    #> sampling redshifts and halo masses
    # if zls is None: zls = sample_redshifts(nlens)
    
    #> halo mass function
    halom = halo_mass_fn(np.array(zls),       # array of lens redshifts
                         log10Mmin=log10Mmin, # lower bound of mass range [solM]
                         log10Mmax=log10Mmax, # upper bound of mass range [solM]
                         ngrid=ngrid,         # ln(M/h) grid size
                         mdef=mdef,           # halo mass definition (200c, 200m, etc.)
                         cosmo=cosmo)         # cosmology
    
    #> halo mass-concentration relation
    haloc = mass_c_vir_rel(M_virs=halom,      # array of virial halo masses
                           zls=zls,           # array of redshifts
                           cosmo=cosmo)       # cosmology
    
    #> stellar-to-halo mass relation
    starm = stellar_h_mass_rel(M_virs=halom,  # array of virial halo masses
                               zls=zls)       # array of lens redshifts
    
    #> stellar 
    reffs = mass_r_relation_18mowla(M_stars=starm,    # array of stellar masses [solM]
                                    zls=zls)          # array of lens redshifts
    
    #> converting masses
    log10_halom = np.log10(halom)
    log10_starm = np.log10(starm)
    
    #> dataframe
    names = ['zl', 'log10_M_vir', 'c_vir', 'log10_M_star', 'R_eff']
    data = np.vstack([zls, log10_halom, haloc, log10_starm, reffs]).T
    df = pd.DataFrame(data, columns=names)
    
    return df


""" #> REDSHIFT ======================
================================== """

#> returns random redshifts
def ranRedshifts(numGals, **kwargs):
    
    ### THESE SHOULD BE INFORMED FROM OBSERVATIONS/ THEORY
    ### I CAN DO A FIRST PASS LOOK W/ EINSTEIN RADIUS...? 
    ### OR LENSING PROBABILITY
    
    ### should be a joint probability distribution 
    
    #> correlation matrix
    cross_corl = 0.4
    R = ( np.identity(n=2) * (1-cross_corl) ) + cross_corl
    
    #> deflector: lb, ub, mu, sigma
    ub_zl    = kwargs.get('ub_zl', 1.5)
    lb_zl    = kwargs.get('lb_zl', 0.1)
    mu_zl    = kwargs.get('mu_zl', 0.5)
    sigma_zl = kwargs.get('sigma_zl', 0.1)
    
    #> source: lb, ub, mu, sigma
    ub_zs    = kwargs.get('ub_zs', 3.0)
    lb_zs    = kwargs.get('lb_zs', 0.5)
    mu_zs    = kwargs.get('mu_zs', 1.0)
    sigma_zs = kwargs.get('sigma_zs', 0.1)
    
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
def sample_redshifts(nlens, **kwargs):
    
    #> kwargs
    zobs = kwargs.get('zobs', None)
    
    #> cleaning redshift array
    zobs = np.asarray(zobs)
    zobs = zobs[np.isfinite(zobs)]
    
    if len(zobs) == 0: error.phrase('No valid lens redshifts supplied!')
    if any(zobs < 0): error.phrase('Lens redshifts must be non-negative!')
    
    #> bootstrap sampling
    return np.random.choice(zobs, nlens, replace=True)


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
def halo_mass_fn(zls, cosmo=u.cosmo, **kwargs):
    
    
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

    #> iterating through lens redshifts
    halom = np.zeros(len(zls))
    for i, zl in enumerate(zls):

        #> halo cdf at lens redshift
        lnm, cdf = halo_cdf(zl,                  # lens redshift
                            h=h,                 # H0/100
                            log10Mmin=log10Mmin, # min log10(M/solM)
                            log10Mmax=log10Mmax, # max log10(M/solM)
                            mdef=mdef,           # halo mass definition
                            ngrid=ngrid)         # number grid
        
        #> inverse-cdf sampling
        #> interps cdf based on U random #, with lnm values, then gets solM/h
        halom_h  = np.exp(np.interp(np.random.random(), cdf, lnm)) # solM/h
        halom[i] = halom_h / h # log10(solMass)
    
    return halom


#> obtaining colossus mass function and cdf
# https://bdiemer.bitbucket.io/colossus/lss_mass_function.html#lss.mass_function.massFunction
def halo_cdf(zl, h, log10Mmin=11.5, log10Mmax=14.5, mdef='vir', ngrid=4096):
    
    #> logarithmic mass grid
    lnMmin = np.log(h * 10**log10Mmin)       # changing to solM/h and base e
    lnMmax = np.log(h * 10**log10Mmax)       # changing to solM/h and base e
    lnm = np.linspace(lnMmin, lnMmax, ngrid) # grid in base e
    mass_h = np.exp(lnm)                     # changing to Msol/h
    
    #> mass function ( returns dn/d[ln(solM/h)] per ln(solM/h) )
    dndlnm = mass_function.massFunction(x=mass_h,          # solM/h bins
                                        z=zl,              # lens redshift
                                        q_in='M',          # in mass = solM/h (halo masses)
                                        q_out='dndlnM',    # out mass = dn/dlnM(solM/h)
                                        mdef=mdef,         # halo mass definition
                                        model='despali16', # model for mass function
                                        ellipsoidal=False) # If True, return the results for an ellipsoidal halo finder, otherwise standard SO.
    
    #> removing nans and infs
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
def mass_c_vir_rel(M_virs, zls, cosmo=u.cosmo):
    
    #> declarations
    h = cosmo['h0'] / 100 # reduced hubble param
    Ms = (M_virs * h / (1e12)) # solMass/h
    
    #> interating thru each lens
    haloc = np.zeros(len(zls))
    for i, M, zl in zip(range(len(zls)), Ms, zls):
        
        #> getting coeffs (for c_vir vs. M_vir)
        a = 0.537 + ( (1.025-0.537) * np.exp(-0.718 * zl**(1.08)) )
        b = -0.097 + 0.024*zl
        
        #> mass-concentration relation
        log10_c_vir = a + ( b * np.log10(M) )
        haloc[i] = 10**(log10_c_vir)
    
    return haloc


#> mass-concentration_200 relation
#> taken from Table 3 of 2014 Dutton & Maccio, plus eq 10 & 11
#> w/ Planck cosmology
def mass_c_200_rel(M_200, zl, h):
    
    #> declarations
    M = (M_200 * h / (1e12)) # solMass/h
    
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
def stellar_h_mass_rel(M_virs, zls, **kwargs):
    
    #> declarations
    h_shuntov = 0.7 # reduced hubble param of 2022 Shuntov
    
    #> iterating thru each lens
    starm = np.zeros(len(zls))
    for i, M_vir, zl in zip(range(len(zls)), M_virs, zls):
        
        #> converting
        logM_vir = np.log10(M_vir * h_shuntov) # log(solMass/h_shuntov)
        
        #> coefficients 
        #> table F.1 bounds are x<z<y, setting to x<=z<y for continuity
        #> the first bounds are 0.2 < z < 0.5 --> changing to z < 0.5
        #> implying no significant evolution between z=0.2 and z=0
        if zl < 0.5:
            logM_1, logM_star_0 = 12.629, 10.855
            beta, delta, gamma  = 0.487, 0.935, 1.939
            sigma_logM          = 0.268
        elif zl < 0.8:
            logM_1, logM_star_0 = 12.793, 10.927
            beta, delta, gamma  = 0.502, 0.802, 3.132
            sigma_logM          = 0.293
        elif zl < 1.1:
            logM_1, logM_star_0 = 12.730, 11.013
            beta, delta, gamma  = 0.454, 1.109, 1.925
            sigma_logM          = 0.250
        elif zl < 1.5:
            logM_1, logM_star_0 = 12.673, 10.967
            beta, delta, gamma  = 0.393, 0.746, 0.335
            sigma_logM          = 0.167
        elif zl < 2.0:
            logM_1, logM_star_0 = 12.787, 11.04
            beta, delta, gamma  = 0.41, 0.716, 1.312
            sigma_logM          = 0.211
        else: # limiting to z<2.0 due to lensing probability
            error.phrase(f'Lens redshift of {zl} is beyond z=2')
            
        #> residual of Eq. 9 from 2022 Shuntov
        def residual(logM_star):
            
            ratio = 10**(logM_star - logM_star_0)
            
            logM_vir_model = ( logM_1 + ( beta * np.log10(ratio) ) 
                              + ( ratio**delta / (1 + ratio**(-gamma)) )
                              - 0.5)
            
            return logM_vir_model - logM_vir
    
        #> minimizing
        lb_logM_star = 6                 # log10(solMass)
        ub_logM_star = 13                # log10(solMass)
        a = np.log10(10**lb_logM_star * h_shuntov**2) # log10(solMass/h_shuntov^2)
        b = np.log10(10**ub_logM_star * h_shuntov**2) # log10(solMass/h_shuntov^2)
        logM_star = brentq(f = residual, # function to minimize
                           a = a,        # lower bound
                           b = b)        # upper bound
        
        #> sampling from Gaussian with mu=logM_star
        if kwargs.get('sigma_logM', None) is not None:
            sigma_logM = kwargs.get('sigma_logM')
        sample_logM_star = norm.rvs(loc=logM_star, scale=sigma_logM)
        
        #> converting
        M_star = 10**sample_logM_star / h_shuntov**2 # solMass
        starm[i] = M_star
    
    return starm


""" #> STELLAR-M-S REL ===============
================================== """

#> stellar mass-size relation (half-light radius)
#> taken from 2025 Cook; Table 4 spheroid-dominated, Eq. 2
def mass_r_relation(M_stars, zls, morph='spheroid', **kwargs):
    
    #> 2025 Cook mass range of M_* >~ 3e9 M_sol, and 0.3 < z < 1.0
    
    #> setting slope and norm coeffs
    if morph == 'spheroid':
        a_alpha, a_beta =  0.82, 0.36
        b_alpha, b_beta = -1.66, 0.42
    elif morph == 'spheroid-dominated':
        a_alpha, a_beta =  2.9, 0.25
        b_alpha, b_beta = -3.1, 0.60
    
    #> iterating thru each lens
    r_eff = np.zeros(len(zls))
    for i, (M_star, zl) in enumerate(zip(M_stars, zls)):
        
        #> solving for slope and norm
        a = a_beta*(1+zl)**(a_alpha)
        b = b_beta*(1+zl)**(b_alpha)
        
        #> mean logarithmic effective radius
        logR_eff = np.log10(b) + a*np.log10(M_star/1e10) # log10(kpc)
        
        #> sigma eff radius (sigma_log_R_eff = 0.158)
        sigma_perp = 0.158
        sigma_log_R_eff = (sigma_perp) * np.sqrt(1 + a**2)
        
        #> sampling intrinsic size scatter
        if kwargs.get('sigma_log_R_eff', None) is not None:
            sigma_log_R_eff = kwargs.get('sigma_log_R_eff')
        sample_log_R_eff = np.random.normal(logR_eff, sigma_log_R_eff)
        r_eff[i] = 10**(sample_log_R_eff) # kpc
    
    return r_eff


#> stellar mass-size relation (half-light radius)
#> taken from Eq. 3, Table 1 of 2014 Van der Wel
def mass_r_relation_14wel(M_stars, zls):
    
    #### TWO THINGS -- IS IT 5E10? AND NEW COEFFS FROM 2018 MOWLA -- CAN I USE?
    
    #> iterating thru each lens
    r_eff = np.zeros(len(zls))
    for i, (M_star, zl) in enumerate(zip(M_stars, zls)):
        
        #> early-type coefficients from van der Wel 2014
        if zl < 0.5:
            logA, alpha, sigma_logR = 0.60, 0.75, 0.10
        elif zl < 1.0:
            logA, alpha, sigma_logR = 0.42, 0.71, 0.11
        elif zl < 1.5:
            logA, alpha, sigma_logR = 0.22, 0.76, 0.12
        elif zl < 2.0:
            logA, alpha, sigma_logR = 0.09, 0.76, 0.14
        else: # limiting to z<2.0 due to lensing probability
            error.phrase(f'Lens redshift of {zl} is beyond z=2')
        
        #> mean logarithmic effective radius
        #> Eq. 3 says m*=7e10, but it should be 5e10
        mean_logR = logA + (alpha * np.log10(M_star/(5e10)) ) # log10(kpc)
        
        #> sampling intrinsic size scatter
        sample_logR = np.random.normal(mean_logR, sigma_logR)
        
        #> converting to physical kpc
        r_eff[i] = 10**sample_logR
    
    return r_eff


#> stellar mass-size relation (half-light radius)
#> taken from Eq. 4, Table 2 of 2018 Mowla
def mass_r_relation_18mowla(M_stars, zls):
    
    #> iterating thru each lens
    r_eff = np.zeros(len(zls))
    for i, (M_star, zl) in enumerate(zip(M_stars, zls)):
        
        #> redshift evolution of the params
        #> taken from last paragraph of 6.1 2018 Mowla, quiescent
        logA = -0.52 * np.log10(1+zl) + 0.31
        alpha = 0.57
        sigma_logR = 0.25 # average of Table 1, quiescent
        
        #> mean logarithmic effective radius
        #> Eq. 4 says m*=7e10, but it should be 5e10
        mean_logR = logA + (alpha * np.log10(M_star/(5e10)) ) # log10(kpc)
        
        #> sampling intrinsic size scatter
        sample_logR = np.random.normal(mean_logR, sigma_logR)
        
        #> converting to physical kpc
        r_eff[i] = 10**sample_logR
    
    return r_eff


#> stellar mass-size relation (half-mass radius)
#> taken from Eq. 1, Table 3 of 2026 Xin
def mass_r_relation_26xin(M_stars, zls):
    
    #> iterating thru each lens
    r_eff = np.zeros(len(zls))
    for i, (M_star, zl) in enumerate(zip(M_stars, zls)):
        
        pass
        
    return r_eff


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    import params
    priorDict = params.getPriorDict(all=True)
    
    #> sampling
    numgals = 100
    zls = np.linspace(0.2, 1.0, numgals)
    df = sample_priors(numgals, priorDict, zls=zls, seed=10)
    print(df)
    
    # print(np.array2string(data, formatter={'float_kind': lambda x: f'{x:.2e}'}))
    
    import plot
    # plot.priors_correl(df, plot=True)
    
    # end
# thank