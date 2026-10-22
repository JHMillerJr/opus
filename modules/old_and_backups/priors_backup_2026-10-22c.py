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

#> sampling from the priors (redshifts --> nfw --> hern --> autoscaling + selection)
def sample_priors(numGals, **kwargs):
    
    #> kwargs
    priorDict = kwargs.get('priorDict')         # which priors are on (nfw, hern, red)
    redshifts = kwargs.get('redshifts', None)   # (numGals, 2) redshifts, None = drawn
    seed      = kwargs.get('seed', None)        # random seed
    df_in     = kwargs.get('df_in', None)       # galaxy values for priors that are off (log10_M_vir, c_vir, log10_M_star, R_eff)
    redraw    = kwargs.get('redraw', None)      # redraw(inds) --> new df_in rows for rejected galaxies (varied params)
    
    #> checking the galaxy values
    need = {'nfw': ['log10_M_vir', 'c_vir'], 'hern': ['log10_M_star', 'R_eff']}
    for group in need:
        if priorDict[group]: continue
        if df_in is None or any(col not in df_in for col in need[group]):
            error.phrase(f'{group} priors are off, but {need[group]} were not supplied (df_in)')
    
    #> setting random seed
    np.random.seed(seed)
    
    #> redshifts (rows w/ both redshifts drawn can be redrawn)
    if redshifts is None: z_drawn = np.ones(numGals, dtype=bool)
    else: z_drawn = np.broadcast_to(np.all(np.array(redshifts, dtype=object).reshape(-1,2) == None, axis=1), numGals)
    df = pd.DataFrame(draw_redshifts(numGals, **kwargs), columns=['zl', 'zs'])
    
    #> main sample function
    def sample(df, df_in=None):
        
        #> nfw priors (halo mass function + mass-concentration relation)
        if priorDict['nfw']: 
            nfw = draw_nfw(df['zl'], **kwargs)
            df, M_halos = df.join(nfw), nfw.attrs['M_halo']                     # solMass (exact)
        else: 
            df[need['nfw']] = df_in[need['nfw']].to_numpy(dtype=float)
            M_halos = 10**df['log10_M_vir'].to_numpy(dtype=float)               # solMass
        
        #> hernquist priors (stellar-to-halo mass relation + stellar mass-size relation)
        if priorDict['hern']: df = df.join(draw_hern(M_halos, df['zl'], **kwargs))
        else:
            df[need['hern']] = df_in[need['hern']].to_numpy(dtype=float)
            df['mstar_nsig'], df['re_nsig'] = lookup_nsig(df) # sigma offsets (look-up coordinates)
        
        #> axis ratio (look-up corrections)
        if df_in is not None and 'axisrat' in df_in: df['axisrat'] = df_in['axisrat'].to_numpy(dtype=float)
        
        return df
    
    #> initial sample
    df = sample(df, df_in)
    df = autoscale(df, **kwargs)
    
    #> rejected galaxies w/ fixed masses can only be selected by redrawing their redshifts
    fixed_mass = (not priorDict['nfw']) and (not priorDict['hern']) and (redraw is None or not kwargs.get('redraw_mass', True))
    nz = 0 # number of redshift redraws
    
    #> redraw rejected galaxies
    factor = 100
    tries = maxTries * factor
    while any(~df['sel']) and tries > 0:
        
        #> rejected galaxies
        mask = ~df['sel']
        
        #> redrawing varied params of the rejected galaxies (priors that are off)
        df_in_resample = None
        if df_in is not None:
            if redraw is not None: df_in.loc[mask, :] = redraw(np.flatnonzero(mask))[df_in.columns].to_numpy()
            df_in_resample = df_in.loc[mask].reset_index(drop=True)
        
        #> redrawing redshifts of rejected galaxies w/ fixed masses
        zmask = mask & z_drawn
        if fixed_mass and zmask.any():
            df.loc[zmask, ['zl','zs']] = draw_redshifts(int(zmask.sum()), **(kwargs | {'redshifts': None})) # (only these galaxies)
            nz += int(zmask.sum())
        
        #> resampling at fixed redshifts
        df_resample = df.loc[mask,['zl','zs']].copy().reset_index(drop=True)
        df_resample = sample(df_resample, df_in_resample)
        df_resample = autoscale(df_resample, **kwargs)
        
        #> replacing
        cols = df_resample.columns
        df.loc[mask,cols] = df_resample.to_numpy()
        
        #> reducing tries
        tries -= 1
    
    #> checking if all lenses were accepted
    print(f'> Sampling priors took {maxTries*factor - tries} tries!')
    if nz > 0:
        error.highlight(f'Masses are fixed: redrew the redshifts of rejected galaxies {nz} time(s)')
    if any(~df['sel']):
        error.highlight(f'Could not find accepted samples for {sum(~df["sel"])} galaxies')
    
    return df


""" #> DRAW REDSHIFTS ================
================================== """

#> draws (zl, zs) for numGals galaxies; supplied redshifts are kept, None = drawn
def draw_redshifts(numGals, **kwargs):
    
    #> kwargs
    redshifts = kwargs.get('redshifts', None) # (numGals, 2) or (zl, zs) redshifts, None = drawn
    
    #> declarations
    if redshifts is None: redshifts = np.full((numGals,2), None, dtype=object)
    redshifts = np.array(redshifts, dtype=object).reshape(-1,2)
    if len(redshifts) == 1: redshifts = np.tile(redshifts, (numGals,1)) # same (zl, zs) for every galaxy
    no_zl = (redshifts[:,0] == None)
    no_zs = (redshifts[:,1] == None)
    
    #> both redshifts missing --> drawn from the redshift distribution
    both = no_zl & no_zs
    if both.any(): redshifts[both] = drawRedshifts(int(both.sum()), **kwargs)
    
    #> only zl missing --> zl | zs
    for i in np.flatnonzero(no_zl & ~no_zs):
        redshifts[i,0] = conditional_redshifts(z=float(redshifts[i,1]), known='zs', **kwargs)
    
    #> only zs missing --> zs | zl
    for i in np.flatnonzero(~no_zl & no_zs):
        redshifts[i,1] = conditional_redshifts(z=float(redshifts[i,0]), known='zl', **kwargs)
    
    return redshifts.astype(float)


""" #> DRAW NFW ======================
================================== """

#> draws halo masses (hmf) and concentrations (cmr) at lens redshifts zls (halo mass definition: mdef)
def draw_nfw(zls, **kwargs):
    
    #> declarations
    zls = np.asarray(zls, dtype=float)
    
    #> halo mass function
    halom = halo_mass_fn(zls, **kwargs)        # solMass
    
    #> halo mass-concentration relation
    haloc = mass_c_rel(halom, zls, **kwargs)   # concentration []
    
    #> (log10_M_vir & c_vir are the mdef halo mass & concentration)
    df = pd.DataFrame({'log10_M_vir': np.log10(halom), # log10(solMass)
                       'c_vir':       haloc})          # concentration []
    df.attrs['M_halo'] = halom                         # solMass (exact, for draw_hern)
    
    return df


""" #> DRAW HERN =====================
================================== """

#> draws stellar masses (shmr) and effective radii (msr) from halo masses M_halos [solMass] at zls
def draw_hern(M_halos, zls, **kwargs):
    
    #> kwargs
    msr_method = kwargs.get('msr_method', '18mowla') # what stellar mass-size relation to use
    
    #> declarations
    msr_func = {'14wel':   mass_r_relation_14wel,   # MSR from 14 van der Wel+
                '18mowla': mass_r_relation_18mowla, # MSR from 18 Mowla+
                '25cook':  mass_r_relation_25cook}  # MSR from 25 Cook+
    M_halos = np.asarray(M_halos, dtype=float)
    zls     = np.asarray(zls, dtype=float)
    
    #> stellar-to-halo mass relation
    mstar_nsig = np.random.normal(size=len(zls))
    starm = stellar_h_mass_rel(M_virs=M_halos,  # array of halo masses
                               zls=zls,         # array of lens redshifts
                               nsig=mstar_nsig) # sigma offset
    
    #> stellar mass-size relation
    re_nsig = np.random.normal(size=len(zls))
    reffs = msr_func[msr_method](M_stars=starm, # array of stellar masses [solM]
                                 zls=zls,       # array of lens redshifts
                                 nsig=re_nsig)  # sigma offset
    
    return pd.DataFrame({'log10_M_star': np.log10(starm), # log10(solMass)
                         'R_eff':        reffs,           # kpc
                         'mstar_nsig':   mstar_nsig,      # sigma offset from the shmr
                         're_nsig':      re_nsig})        # sigma offset from the msr


""" #> AUTOSCALING ===================
================================== """

#> look-up einstein radius --> pix_arc [pix/arcsec] and selection (selDict) for galaxies in df
#> df: zl, zs, log10_M_vir, log10_M_star, R_eff (+ mstar_nsig, re_nsig, otherwise calculated)
def autoscale(df, **kwargs):
    
    #> sigma offsets (look-up coordinates) if not given
    if 'mstar_nsig' not in df or 're_nsig' not in df:
        df = df.copy()
        df['mstar_nsig'], df['re_nsig'] = lookup_nsig(df)
    
    return lookup.firstLook_selFunc_newt(df, **kwargs)


#> sigma offsets of M_star & R_eff from the shmr & 18mowla msr (the v3 look-up coordinates)
def lookup_nsig(df):
    
    #> declarations
    zls     = df['zl'].to_numpy(dtype=float)
    M_halos = 10**df['log10_M_vir'].to_numpy(dtype=float)  # solMass
    M_stars = 10**df['log10_M_star'].to_numpy(dtype=float) # solMass
    R_effs  = df['R_eff'].to_numpy(dtype=float)            # kpc
    
    return shmr_nsig(M_stars, M_halos, zls), msr_nsig(R_effs, M_stars, zls)


""" #> REDSHIFT ======================
================================== """

#> main redshift prior function
def drawRedshifts(numGals, **kwargs):
    
    #> kwargs
    red_func = kwargs.get('red_func', 'random')
    
    #> getting function to sample redshifts
    z_funcs = {'assign': assign_redshifts,          # assigns redshifts from obs dist
               'random': ranRedshifts,              # draws randomly (can edit redshift dist)
               'sample1D': sample_obsRedshifts_1D,  # draws randomly from obs marginal dist
               'sample2D': sample_obsRedshifts_2D,} # draws randomly from obs joint dist
    z_func = z_funcs[red_func]
    
    return z_func(numGals, **kwargs)
    

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
    tmvn = TruncatedMVN(mu, cov, lb, ub, seed=np.random.randint(2**31 - 1))
    samples = tmvn.sample(numGals)
    df_samples = pd.DataFrame(np.array(samples).T, columns=['zl', 'zs'])
    
    #> checking for unphysical samples
    tries = maxTries * 10
    mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | # zs >= zl + buffer
             (df_samples['zl'] <= 0) |                           # zl > 0
             (df_samples['zs'] <= 0) )                           # zs > 0
    while any(mask) and tries > 0:
        
        #> re sampling
        re_samples = np.array(tmvn.sample(sum(mask))).T
        
        #> replacing unphysical samples
        df_samples.loc[mask, ['zl','zs']] = re_samples
        
        #> checking mask again
        mask = ( (df_samples['zs'] - df_samples['zl'] <= z_buffer) | 
                 (df_samples['zl'] <= 0) | 
                 (df_samples['zs'] <= 0) )
        tries -= 1
        
    #> checking if enough physical samples were found
    if any(mask):
        error.phrase('Could not find enough physical random redshift samples')
    
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
    E = lambda z: ( cosmo['omega_m_0']*(1 + z)**3 + cosmo['omega_lam_0'] )**0.5
    
    #> omega_m(z); 1997 Bryan & Norman
    omega_m_z = ( cosmo['omega_m_0'] * (1+z)**3 ) / E(z)**2
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
    mdef       = kwargs.get('mdef', 'vir')         # halo mass definition ('vir' or '200c')
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
        halom[i] = halom_h / h # solMass
    
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
def mass_c_rel(halom, zls, **kwargs):
    
    #> kwargs
    cosmo = kwargs.get('cosmo', u.cosmo) # cosmology
    mdef  = kwargs.get('mdef', 'vir')    # halo mass definition ('vir' or '200c')
    
    #> for virial mass
    if mdef in ['vir']:
        return mass_c_vir_rel(M_virs=halom, # array of virial halo masses
                              zls=zls,      # array of lens redshifts
                              cosmo=cosmo)  # cosmology
    
    #> for m 200
    if mdef in ['200c']:
        return mass_c_200_rel(M_200=halom,  # array of m200 halo masses
                              zl=np.asarray(zls, dtype=float), # array of lens redshifts
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


#> sigma offset of given stellar masses from the shmr (inverse of stellar_h_mass_rel)
def shmr_nsig(M_stars, M_virs, zls):
    
    #> declarations
    h_shuntov = 0.7 # reduced hubble param of 2022 Shuntov
    
    #> lognormal params (nsig=0 so no random draws)
    p = stellar_h_mass_rel(M_virs, zls, return_sample=False, nsig=np.zeros(len(zls))).astype(float)
    logM_star, sigma_logM = p[:,0], p[:,1] # log10(solMass/h_shuntov^2), dex
    
    return (np.log10(np.asarray(M_stars) * h_shuntov**2) - logM_star) / sigma_logM


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
        draw = np.random.normal() if kwargs.get('nsig', None) is None else kwargs.get('nsig')[i]
        sample_log_R_eff = logR_eff + draw*sigma_log_R_eff
        r_eff[i] = 10**(sample_log_R_eff) # kpc
    
    return r_eff


#> stellar mass-size relation (half-light radius)
#> taken from Eq. 3, Table 1 of 2014 Van der Wel
def mass_r_relation_14wel(M_stars, zls, **kwargs):
    
    #> kwargs
    nsig = kwargs.get('nsig', None)
    
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
        draw = np.random.normal() if nsig is None else nsig[i]
        sample_logR = mean_logR + draw*sigma_logR
        
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


#> sigma offset of given effective radii from the 18mowla msr (inverse; the v3 look-up is built w/ 18mowla)
def msr_nsig(R_effs, M_stars, zls):
    
    #> lognormal params (nsig=0 so no random draws)
    p = np.array(list(mass_r_relation_18mowla(M_stars, zls, return_sample=False, nsig=np.zeros(len(zls)))), dtype=float)
    mean_logR, sigma_logR = p[:,0], p[:,1] # log10(kpc), dex
    
    return (np.log10(np.asarray(R_effs)) - mean_logR) / sigma_logR


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
    import params
    
    #> declarations
    numgals = 10
    np.random.seed(13)
    
    #> each step on its own
    redshifts = draw_redshifts(numgals)                                  # (zl, zs)
    nfw       = draw_nfw(redshifts[:,0], mdef='vir')                     # log10_M_vir, c_vir
    hern      = draw_hern(10**nfw['log10_M_vir'], redshifts[:,0])        # log10_M_star, R_eff, nsigs
    df        = pd.DataFrame(redshifts, columns=['zl', 'zs']).join(nfw).join(hern)
    df        = autoscale(df, nph=50, selDict=params.getSelDict())       # ER, pix_arc, sel
    print(df.to_string())
    
    #> everything together
    df = sample_priors(numgals, priorDict=params.getPriorDict(all=True), nph=50, seed=13)
    print(df.to_string())
    
    # end
# thank