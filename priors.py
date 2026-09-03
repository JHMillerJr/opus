#> name: priors.py
#> author: John Miller Jr
#> descrp: priors for cosmological population of elliptical galaxies

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import pandas as pd
import time
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid

#> modules
import modules.lookup as lookup
from modules.units import u; u=u()
import modules.error as error
from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN   

#> colossus
from colossus.lss import mass_function
from colossus.cosmology import cosmology as ccos

#> declarations
maxTries = 100
fontsize, labelpad = 15, 10
obs_redshift_file = '../observed_quads/redshifts.csv'
    

""" #> DRAWING SAMPLES ===============
================================== """

#> sampling from the priors
def sample_priors(numGals, priorDict, nph, **kwargs):
    
    #> preloading zl files
    # lookup.preload_lookup_v3(verbose=True)
    
    #> parent kwargs
    redshifts  = kwargs.get('redshifts', None) # list of redshifts
    cosmo      = kwargs.get('cosmo', u.cosmo)  # cosmology
    obsFile    = kwargs.get('obsFile', obs_redshift_file) # observed redshift file
    z_func     = kwargs.get('z_func', 'assign')# the function to sample redshifts
    z_method   = kwargs.get('z_method', '2D')  # the method to sample redshifts
    seed       = kwargs.get('seed', None)      # random seed
    
    #> lookup interpolation kwargs
    interp_zl    = kwargs.get('interp_zl',    False)
    interp_zs    = kwargs.get('interp_zs',    False)
    interp_mvir  = kwargs.get('interp_mvir',  True)
    interp_mstar = kwargs.get('interp_mstar', True)
    interp_re    = kwargs.get('interp_re',    True)
    interp_kw    = {'interp_zl': interp_zl, 'interp_zs': interp_zs,
                    'interp_mvir': interp_mvir, 'interp_mstar': interp_mstar,
                    'interp_re': interp_re}
    
    #> setting random seed
    np.random.seed(seed)
    
    #> 
    msr_method = kwargs.get('msr_method', '18mowla') # what stellar mass-size relation to use
    
    
    #> declarations
    msr_func = {'14wel':   mass_r_relation_14wel,   # MSR from 14 van der Wel+
                '18mowla': mass_r_relation_18mowla, # MSR from 18 Mowla+
                '25cook':  mass_r_relation_25cook}  # MSR from 25 Cook+
    
    #> getting function to sample redshifts
    z_funcs = {'assign': assign_redshifts,          # assigns redshifts from obs dist
               'random': ranRedshifts,              # draws randomly (can edit redshift dist)
               'sample1D': sample_obsRedshifts_1D,  # draws randomly from obs marginal dist
               'sample2D': sample_obsRedshifts_2D,} # draws randomly from obs joint dist
    z_func = z_funcs[z_func]
    

    #> sampling redshifts
    if redshifts is None: redshifts = np.full((numGals,2), None, dtype=object)
    if np.all(redshifts[:,0] == None):
        
        #> if both redshifts are missing
        if np.all(redshifts[:,1] == None):
            redshifts = z_func(numGals, method=z_method)
        else: 
            #> if only zl is missing (zs is known)
            redshifts[:,0] = np.array([ 
                             conditional_redshifts(z=zs,
                                                   known='zs',
                                                   obsFile=obsFile)
                             for zs in np.array(redshifts[:,1]) ]) # iterating thru each zs
    #> if zl is supplied, if zs is not supplied
    elif np.all(redshifts[:,1] == None):
        redshifts[:,1] = np.array([ 
                         conditional_redshifts(z=zl,
                                               known='zl',
                                               obsFile=obsFile)
                         for zl in np.array(redshifts[:,0]) ]) # iterating thru each zl

    #> initializing dataframe
    #> dataframe
    names = ['zl', 'zs']
    data = np.vstack([redshifts[:,0],    # zl
                      redshifts[:,1]]).T # zs
    df = pd.DataFrame(data, columns=names)
        
    #> main sample function
    def sample(df):
    
        #> halo mass function
        halom = halo_mass_fn(df['zl'],      # array of lens redshifts
                             cosmo=cosmo,   # cosmology
                             **kwargs)      # prior kwargs       
        
        #> halo mass-concentration relation
        haloc = mass_c_vir_rel(M_virs=halom,      # array of virial halo masses
                               zls=df['zl'],      # array of redshifts
                               cosmo=cosmo)       # cosmology
        
        #> stellar-to-halo mass relation
        mstar_nsig = np.random.normal(size=len(df))
        starm = stellar_h_mass_rel(M_virs=halom,    # array of virial halo masses
                                   zls=df['zl'],    # array of lens redshifts
                                   nsig=mstar_nsig) # sigma offset
        
        #> stellar mass-size relation
        re_nsig = np.random.normal(size=len(df))
        reffs = msr_func[msr_method](M_stars=starm, # array of stellar masses [solM]
                                     zls=df['zl'],  # array of lens redshifts
                                     nsig=re_nsig)  # sigma offset
        
        #> converting masses
        log10_halom = np.log10(halom)
        log10_starm = np.log10(starm)
        
        #> saving
        df['log10_M_vir'] = log10_halom
        df['c_vir'] = haloc
        df['log10_M_star'] = log10_starm
        df['R_eff'] = reffs
        df['mstar_nsig'] = mstar_nsig
        df['re_nsig'] = re_nsig
        
        return df
    
    #> initial sample
    df = sample(df)
    
    srt = time.time()
    df = lookup.firstLook_selFunc_v3(df, nph=nph, **interp_kw)
    print(time.time()-srt)
    
    #> redraw rejected galaxies
    tries = maxTries * 10
    while any(~df['sel']) and tries > 0:
        
        #> rejected galaxies
        mask = ~df['sel']
        
        #> resampling at fixed redshifts
        df_resample = df.loc[mask,['zl','zs']].copy().reset_index(drop=True)
        df_resample = sample(df_resample)
        df_resample = lookup.firstLook_selFunc_v3(df_resample, nph=nph, **interp_kw)
        
        #> replacing
        cols = df_resample.columns
        df.loc[mask,cols] = df_resample.to_numpy()
        
        #> reducing tries
        tries -= 1
        
    #> checking if all lenses were accepted
    print(tries)
    if any(~df['sel']):
        pass
        #error.phrase(f'Could not find accepted samples for {sum(~df["sel"])} galaxies')
    
    cols = ['zl', 'zs', 'd_area', 'pix_arc', 'ER']
    # print(df[cols])
    # print(df.to_string())
    # df.to_csv('lookup.csv')
    
    return df


""" #> REDSHIFT ======================
================================== """

#> returns random redshifts
#> currently calibrated from obs distribution
def ranRedshifts(numGals, **kwargs):
    
    #> kwargs
    z_buffer = kwargs.get('z_buffer', 0.1)
    
    #> correlation matrix
    cross_corl = 0.15
    R = ( np.identity(n=2) * (1-cross_corl) ) + cross_corl
    
    #> deflector: lb, ub, mu, sigma
    ub_zl    = kwargs.get('ub_zl', 1.8)
    lb_zl    = kwargs.get('lb_zl', 0.01)
    mu_zl    = kwargs.get('mu_zl', 0.5)
    sigma_zl = kwargs.get('sigma_zl', 0.3)
    
    #> source: lb, ub, mu, sigma
    ub_zs    = kwargs.get('ub_zs', 5.0)
    lb_zs    = kwargs.get('lb_zs', 0.1)
    mu_zs    = kwargs.get('mu_zs', 2.0)
    sigma_zs = kwargs.get('sigma_zs', 0.95)
    
    #> constructing params
    lb = np.array([lb_zl, lb_zs])
    ub = np.array([ub_zl, ub_zs])
    mu = np.array([mu_zl, mu_zs])
    sigma = np.array([sigma_zl, sigma_zs])
    cov = np.diag(sigma) @ R @ np.diag(sigma) # converting correlation & std vectors to covariance matrix
    
    #> drawing samples
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    samples = tmvn.sample(numGals)
    df_samples = pd.DataFrame(np.array(samples).T, columns=['zl', 'zs'])
    
    #> checking for unphysical samples
    tries = maxTries
    mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | # zs >= zl + buffer
             (df_samples['zl'] <= 0) |                           # zl > 0
             (df_samples['zs'] <= 0) )                           # zs > 0
    while any(mask) and tries > 0:
        
        #> re sampling
        re_samples = np.array(tmvn.sample(sum(mask))).T
        
        #> replacing unphysical samples
        df_samples.loc[mask, ['zl','zs']] = re_samples
        
        #> checking mask again
        mask = ( (df_samples['zs'] - df_samples['zl'] >= z_buffer) | 
                 (df_samples['zl'] <= 0) | 
                 (df_samples['zs'] <= 0) )
        tries -= 1
        
    #> checking if enough physical samples were found
    if any(mask):
        error.phrase('Could not find enough physical samples')
    
    return df_samples.to_numpy()


#> sampling lens redshifts from the observed sample
def sample_obsRedshifts_1D(numGals, **kwargs):
    
    #> kwargs
    verbose = kwargs.get('verbose', False)
    z_buffer = kwargs.get('z_buffer', 0.1)
    obsFile = kwargs.get('obsFile', obs_redshift_file)
    
    #> observed redshifts file
    df = pd.read_csv(obsFile)
    
    #> getting redshifts
    zl = df[ df['zl'] != '-' ]['zl'].astype(float).to_numpy()
    zs = df[ df['zs'] != '-' ]['zs'].astype(float).to_numpy()
    
    #> if wanting print
    if verbose:
        print(f'> There are {len(zl)} lens and {len(zs)} source obs redshifts')
    
    #> constructing 1D normals
    zl_mu  = np.mean(zl, axis=0)
    zl_std = np.std(zl, axis=0)
    
    zs_mu  = np.mean(zs, axis=0)
    zs_std = np.std(zs, axis=0)
    
    #> sample
    zl_samples = np.random.normal(zl_mu, zl_std, numGals)
    zs_samples = np.random.normal(zs_mu, zs_std, numGals)
    df_samples = pd.DataFrame(np.vstack([zl_samples, zs_samples]).T, columns=['zl','zs'])
    
    #> checking for unphysical samples
    tries = maxTries
    mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | # zs >= zl + buffer
             (df_samples['zl'] <= 0) |                           # zl > 0
             (df_samples['zs'] <= 0) )                           # zs > 0
    while any(mask) and tries > 0:
        
        #> re sampling
        zl_samples = np.random.normal(zl_mu, zl_std, sum(mask))
        zs_samples = np.random.normal(zs_mu, zs_std, sum(mask))
        re_samples = np.vstack([zl_samples, zs_samples]).T
        
        #> replacing unphysical samples
        df_samples.loc[mask, ['zl','zs']] = re_samples
        
        #> checking mask again
        mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | 
                 (df_samples['zl'] <= 0) | 
                 (df_samples['zs'] <= 0) )
        tries -= 1
        
    #> checking if tries
    if any(mask): error.phrase('Could not find enough physical samples')
    
    return df_samples.to_numpy()


#> constructs bivariate normal from obs lens redshifts and samples
def sample_obsRedshifts_2D(numGals, **kwargs):
    
    #> kwargs
    verbose = kwargs.get('verbose', False)
    z_buffer = kwargs.get('z_buffer', 0.1)
    obsFile = kwargs.get('obsFile', obs_redshift_file)
    
    #> observed redshifts file
    df = pd.read_csv(obsFile)
    
    #> getting redshifts
    both_mask = (df['zl'] != '-') & (df['zs'] != '-')
    redshifts = df[both_mask][['zl', 'zs']].astype(float).to_numpy()
    
    #> the minimum z separation from the observed redshift pairs
    if False:
        z_buffer_obs = min(redshifts[:,1] - redshifts[:,0])
        print(z_buffer_obs)
    
    #> if wanting print
    if verbose:
        print(f'> There are {len(redshifts)} obs galaxies with both redshifts')
    
    #> constructing bivariate normal
    mu  = np.mean(redshifts, axis=0)
    cov = np.cov(redshifts, rowvar=False, ddof=1)
    
    #> sampling
    redshift_samples = np.random.multivariate_normal(mu, cov, numGals)
    df_samples = pd.DataFrame(redshift_samples, columns=['zl','zs'])
    
    #> checking for unphysical samples
    tries = maxTries
    mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | # zs >= zl + buffer
             (df_samples['zl'] <= 0) |                           # zl > 0
             (df_samples['zs'] <= 0) )                           # zs > 0
    while any(mask) and tries > 0:
        
        #> re sampling
        re_samples = np.random.multivariate_normal(mu, cov, sum(mask))
        
        #> replacing unphysical samples
        df_samples.loc[mask, ['zl','zs']] = re_samples
        
        #> checking mask again
        mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | 
                 (df_samples['zl'] <= 0) | 
                 (df_samples['zs'] <= 0) )
        tries -= 1
        
    #> checking if tries
    if any(mask): error.phrase('Could not find enough physical samples')
    
    return df_samples.to_numpy()


#> returns zl or zs from conditional distribution of obs redshifts
def conditional_redshifts(z, known, **kwargs):
    
    #> kwargs
    z_buffer = kwargs.get('z_buffer', 0.1)
    obsFile = kwargs.get('obsFile', obs_redshift_file)
    
    #> observed redshifts file
    df = pd.read_csv(obsFile)
    
    #> getting redshifts
    both_mask = (df['zl'] != '-') & (df['zs'] != '-')
    redshifts = df[both_mask][['zl','zs']].astype(float).to_numpy()
    
    #> constructing bivariate normal
    mu  = np.mean(redshifts, axis=0)
    cov = np.cov(redshifts, rowvar=False, ddof=1)
    
    #> conditional distributions
    if known == 'zs': # if z is z_s; zl | zs
        
        #> sample zl | zs
        mu_cond = mu[0] + cov[0,1]/cov[1,1] * (z - mu[1])
        sig_cond = np.sqrt(cov[0,0] - cov[0,1]**2/cov[1,1])
        
        #> physical condition
        physical = lambda zl: zl > 0 and z - zl >= z_buffer
        
    elif known == 'zl': # if z is z_l; zs | zl
        
        #> sample zs | zl
        mu_cond = mu[1] + cov[0,1]/cov[0,0] * (z - mu[0])
        sig_cond = np.sqrt(cov[1,1] - cov[0,1]**2/cov[0,0])
        
        #> physical condition
        physical = lambda zs: zs > 0 and zs - z >= z_buffer
        
    else:
        error.phrase('ztype must be either zl or zs')
    
    #> sampling
    tries = maxTries
    while tries > 0:
        
        #> sampling
        z_sample = np.random.normal(mu_cond, sig_cond)
        
        #> checking if physical
        if physical(z_sample): return z_sample
        tries -= 1
    
    error.phrase('Could not find physical conditional redshift')


#> assigning redshifts based on obs population and sampling missing redshifts
def assign_redshifts(numGals, method='2D', **kwargs):
    
    #> declarations
    sampleFunc = {'1D': sample_obsRedshifts_1D, # 1D sampling (no cov)
                  '2D': sample_obsRedshifts_2D} # 2D sampling (w/ cov)
    
    #> loading obs redshifts
    obsFile = kwargs.get('obsFile', obs_redshift_file)
    df = pd.read_csv(obsFile)
    
    #> random sampling for lenses with no redshifts
    no_z_mask = (df['zl']==df['zs']) & (df['zl']=='-')       # mask
    no_z_samples = sampleFunc[method](numGals=no_z_mask.sum(), # sampling w/ num from mask
                                      obsFile=obsFile)       # passing obsFile if given
    df.loc[no_z_mask,['zl','zs']] = no_z_samples             # saving samples w/ mask
    
    #> conditional zl sampling (inherently method='2D')
    no_zl_mask    = (df['zl'] == '-')                        # mask
    no_zl_samples = [ conditional_redshifts(z=float(zs),     # passing known redshift
                                            known='zs',      # known redshift (=zs)
                                            obsFile=obsFile) # passing obsFile if given
                      for zs in df.loc[no_zl_mask,'zs'] ]    # for ecah zs in mask
    df.loc[no_zl_mask,'zl'] = no_zl_samples                  # saving samples w/ mask
    
    #> conditional zs sampling
    no_zs_mask    = (df['zs'] == '-')                        # mask
    no_zs_samples = [ conditional_redshifts(z=float(zl),     # passing known redshift
                                            known='zl',      # known redshift (=zl)
                                            obsFile=obsFile) # passing obsFile if given
                      for zl in df.loc[no_zs_mask,'zl'] ]    # for ecah zl in mask
    df.loc[no_zs_mask,'zs'] = no_zs_samples                  # saving samples w/ mask
    
    #>>># all missing redshifts have been sampled #<<<#
    
    #> declarations
    numObs = len(df)
    num_full_sample = numGals // numObs # num complete copies
    num_left_over   = numGals % numObs  # num remaining to randomly sample
        
    #> getting full samples
    redshifts = df[['zl','zs']].to_numpy(dtype=float)
    z_sample = np.tile(redshifts, (num_full_sample,1))
    
    #> sampling left overs
    ran_inds = np.random.choice(len(redshifts), 
                                size=num_left_over, 
                                replace=False)
    z_remaining = redshifts[ran_inds]
    
    return np.vstack([z_sample,z_remaining])


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
    
    #> halo-mass function
    log10Mmin  = kwargs.get('hmf_log10Mmin', 11.5) # min log10(M/solM)
    log10Mmax  = kwargs.get('hmf_log10Mmax', 14.5) # max log10(M/solM)
    mdef       = kwargs.get('hmf_mdef', 'vir')     # halo mass definition (200c, 200m, etc.)
    ngrid      = kwargs.get('hmf_ngrid', 4096)     # number grid
    
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

#> main mass-concentration relation
def mass_c_rel(halom, zls, cosmo, **kwargs):
    
    #> kwargs
    mdef = kwargs.get('hmf_mdef', 'vir')
    
    #> for virial mass
    if mdef in ['vir']:
        return mass_c_vir_rel(M_virs=halom, # array of virial halo masses
                              zls=zls,      # array of lens redshifts
                              cosmo=cosmo)  # cosmology
    
    #> for m 200
    if mdef in ['200']:
        return mass_c_200_rel(M_200=halom,  # array of m200 halo masses
                              zls=zls,      # array of lens redshifts
                              cosmo=cosmo)  # cosmology
    
    #> if mdef not found
    error.phrase(f'Mass definition of {mdef} not found in mass-concentration relation')
    return

#> mass-concentration_vir relation
#> taken from Table 3 of 2014 Dutton & Maccio, plus eq 12 & 13
#> w/ Planck cosmology
def mass_c_vir_rel(M_virs, zls, cosmo=u.cosmo):
    
    #> declarations
    h = cosmo['h0'] / 100      # reduced hubble param
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
def mass_c_200_rel(M_200, zl, cosmo=u.cosmo):
    
    #> declarations
    h = cosmo['h0'] / 100    # reduced hubble param
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
    
    #> kwargs
    return_sample = kwargs.get('return_sample', True) # returns lognorm params instead
    nsig = kwargs.get('nsig', None)
    
    #> declarations
    h_shuntov = 0.7 # reduced hubble param of 2022 Shuntov
    
    #> iterating thru each lens
    starm = np.zeros(len(zls))
    logNorm_params = np.empty((len(zls), 2),  dtype='object')
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
        draw = np.random.normal() if nsig is None else nsig[i]
        sample_logM_star = logM_star + draw*sigma_logM
        
        #> saving logNorm params
        logNorm_params[i] = np.array([logM_star, sigma_logM])
        
        #> converting
        M_star = 10**sample_logM_star / h_shuntov**2 # solMass
        starm[i] = M_star
    
    #> returning samples (default)
    if return_sample: return starm
    
    #> returning logNorm params (must be requested)
    return np.array(logNorm_params)


#> returns look up bins for the shmr
def shmr_bins():
    
    #> dictionary
    shmrBins = {'zl':       {'min': 0, 'max':  2.0, 'bins': 5},
                'logmvir':  {'min': 11.5, 'max': 14.5, 'bins': 1000}}
    
    #> setting up bins
    
    #> zl
    #> edges of the bins
    grid_edges = np.array([0., 0.5, 0.8, 1.1, 1.5, 2.0])
    shmrBins['zl']['grid'] = grid_edges
    
    #> logmvir
    #> edges of the bins
    grid_edges  = np.linspace(shmrBins['logmvir']['min'], 
                              shmrBins['logmvir']['max'], 
                              shmrBins['logmvir']['bins'] + 1)
    
    #> centers of the bins
    grid_bins   = (grid_edges[:-1] + grid_edges[1:]) / 2
    
    #> saving depending on what grid
    shmrBins['logmvir']['grid'] = np.array([round(x,6) for x in grid_bins])
    
    return shmrBins

#> getting bin indices for the smhr table
def shmr_fromFile(df, **kwargs):
    
    #> declarations
    h_shuntov = 0.7
    
    #> getting bins
    shmrBins = shmr_bins()
       
    #> getting zl indices
    df['izl_shmr'] = np.digitize(df['zl'], bins=shmrBins['zl']['grid'], right=False) - 1
    
    #> getting logmvir indices
    df['ilogmvir_shmr'] = lookup.bin_indices(x=df['log10_M_vir'], grid=shmrBins['logmvir']['grid'])
    
    df = load_shmr(df['izl_shmr'], df['ilogmvir_shmr'])
    
    #> sampling
    sample_logM_star = norm.rvs(loc=df['logM_star'], scale=df['sigma_logM'])
    M_star = 10**sample_logM_star / h_shuntov**2 # solMass
    
    return M_star

#> returns stellar masses based on M_virs, zls, and 
def load_shmr(izl, ilogmvir, **kwargs):
    
    #> kwargs
    file = kwargs.get('file', './priors/look_up_tables/shmr_table.parquet')
    
    #> load file
    df = pd.read_parquet(file)
    
    #> loading bin sizes
    smhrBins = shmr_bins()

    #> find model
    i = np.ravel_multi_index((izl,        # zl bin
                              ilogmvir),  # M_vir bin
                             (smhrBins['zl']['bins'],
                              smhrBins['logmvir']['bins']))

    return df.iloc[i]

#> generates an array of smhr 'solutions' to act as a look up table
def generate_shmr_table():
    
    #> declarations
    smhrBins = shmr_bins()
    
    #> getting bins
    zl_grid = smhrBins['zl']['grid']
    zl_centers = (zl_grid[:-1] + zl_grid[1:]) / 2
    M_virs = np.array([10**x for x in smhrBins['logmvir']['grid']])
    
    #> declarations
    logM_vir = np.log10(M_virs)
    ilogmvir = np.array(range(len(M_virs)))
    
    #> calculating the stellar mass from M_vir and zl
    data = []
    for izl, zl in enumerate(zl_centers):
        
        #> getting stellar mass
        zls = np.tile(zl, len(M_virs))
        logNorm_params = stellar_h_mass_rel(M_virs=M_virs, zls=zls, return_sample=False)
        logM_star, sigma_logM = logNorm_params[:,0], logNorm_params[:,1]
        
        #> getting indices
        izl = np.tile(int(izl), len(M_virs))
        
        #> saving
        array = np.vstack([izl, ilogmvir, zls, logM_vir, logM_star, sigma_logM]).T
        data.append(array)
    
    #> converting to pandas
    data = np.vstack(data)
    df = pd.DataFrame(data, columns=['izl', 'ilogmvir', 'zl', 'logmvir', 'logM_star', 'sigma_logM'])
    
    #> writing to file
    loc = './priors/look_up_tables/'
    outFile = loc+f'shmr_table.parquet'
    df.to_parquet(outFile)
    
    return


""" #> STELLAR-M-S REL ===============
================================== """

#> stellar mass-size relation (half-light radius)
#> taken from 2025 Cook; Table 4 spheroid-dominated, Eq. 2
def mass_r_relation_25cook(M_stars, zls, morph='spheroid', **kwargs):
    
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
def mass_r_relation_18mowla(M_stars, zls, **kwargs):
    
    #> kwargs
    return_sample = kwargs.get('return_sample', True) # returns lognorm params instead
    nsig = kwargs.get('nsig', None)
    
    #> iterating thru each lens
    r_eff = np.zeros(len(zls))
    logNorm_params = np.empty(len(zls), dtype='object')
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
        draw = np.random.normal() if nsig is None else nsig[i]
        sample_logR = mean_logR + draw*sigma_logR
        
        #> saving logNorm params
        logNorm_params[i] = [mean_logR, sigma_logR]
        
        #> converting to physical kpc
        r_eff[i] = 10**sample_logR
    
    #> returning samples (default)
    if return_sample: return r_eff
    
    #> returning logNorm params (must be requested)
    return logNorm_params


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
    
    #> imports
    import sys
    import plot
    import params
    
    M_virs = np.array([1e12, 1e13])
    zls = np.array([0.2, 0.6])
    # df = pd.DataFrame([M_virs, zls], columns=[''])
    # shmr_fromFile(M_virs, zls)
    
    # redshifts = assign_redshifts(100, method='2D')
    # print(redshifts)
    
    #> sampling
    numgals = 100
    nph = 50
    zls = np.linspace(0.2, 1.0, numgals)
    priorDict = params.getPriorDict(all=True)
    df = sample_priors(numgals, priorDict, nph=nph, seed=13)
    
    # print(np.array2string(data, formatter={'float_kind': lambda x: f'{x:.2e}'}))
    # plot.priors_correl(df, plot=True)
    
    sys.exit()
    
    # end
# thank