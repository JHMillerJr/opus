#> params.py
# the precursor to deflections.py
    
""" #> IMPORTS =======================
================================== """

#> reg imports
import os
import time
import numpy as np
import scipy as sp
from copy import deepcopy

#> modules
import cosmology
import modules.error as error
import modules.parse as parse
from modules.units import u; u = u()

#> truncated multivariate normal sampler
# from https://github.com/brunzema/truncated-mvn-sampler/blob/main/minimax_tilting_sampler.py
from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN    


""" #> PARAMS ========================
================================== """

#> returns the profile input parameters
def paramRanges():
    
    #> profiles
    ranges = {'zl': 0.5,     # lens redshift
              'zs': 1.0,     # source redshift
              'pix_arc': 60, # pix/arc conversion
              'nfw': [ # nfw
                     {'name': 'p01',
                      'x0':            {'init':  0.0, 'min':-0.1, 'max':0.1, 'fit': False},   # x center of profile 
                      'y0':            {'init':  0.0, 'min':-0.1, 'max':0.1, 'fit': False},   # y center of profile
                      'logmass':       {'init':   13, 'min':  10, 'max': 14, 'fit': False},   # log_10(M) [solMass]
                      'concentration': {'init':    7, 'min':   2, 'max': 10, 'fit': False},   # concentration []
                      'axisrat':       {'init': 0.75, 'min': 0.5, 'max':0.95,'fit': False},    # axis ratio []
                      'theta':         {'init':  0.0, 'min': 0.0, 'max':90.0,'fit': False},    # position angle [deg]
                      'add': True}],
              'hern': [ # hern
                      {'name': 'p02',
                       'x0':      {'init': 0.0, 'min':-0.1, 'max':0.1, 'fit': False},   # x center of profile 
                       'y0':      {'init': 0.0, 'min':-0.1, 'max':0.1, 'fit': False},   # y center of profile
                       'logmass': {'init': 11.1, 'min':   8, 'max': 12, 'fit': False},   # log_10(M) [solMass]
                       'effrad':  {'init':    5, 'min': 0.5, 'max': 10, 'fit': False},   # R_E; effective radius [kpc]
                       'axisrat': {'init':'p01', 'min': 0.5, 'max':0.95,'fit': False},   # axis ratio []
                       'theta':   {'init':  0.0, 'min':-180, 'max':180, 'fit': False},   # position angle [deg]
                       'add': True}],
              'mult': [ # multipoles
                      {'m': 1,  # m num
                       'x0':    {'init':0.0, 'min':-0.1, 'max': 0.1, 'fit': False},   # x center of profile 
                       'y0':    {'init':0.0, 'min':-0.1, 'max': 0.1, 'fit': False},   # y center of profile
                       'norm':  {'init':5e-3, 'min': 0.0, 'max':0.05, 'fit': False},    # normalization []
                       'slope': {'init': 2.0, 'min': 2.0, 'max': 2.0, 'fit': False},   # slope []
                       'theta': {'init':0.0, 'min': 0.0, 'max': 90,  'fit': False},    # position angle [deg]
                       'add': True},
                      {'m': 3,  # m num
                       'x0':    {'init':'p02', 'min':-0.1, 'max': 0.1, 'fit': False},   # x center of profile 
                       'y0':    {'init':'p02', 'min':-0.1, 'max': 0.1, 'fit': False},   # y center of profile
                       'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.05, 'fit': False},    # normalization []
                       'slope': {'init':  2.0, 'min': 2.0, 'max': 2.0, 'fit': False},   # slope []
                       'theta': {'init': 10.0, 'min': -180,'max': 180, 'fit': False},   # position angle [deg]
                       'add': True},
                      {'m': 4,  # m num
                       'x0':    {'init':'p02', 'min':-0.1, 'max': 0.1, 'fit': False},   # x center of profile 
                       'y0':    {'init':'p02', 'min':-0.1, 'max': 0.1, 'fit': False},   # y center of profile
                       'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.05, 'fit': False},    # normalization []
                       'slope': {'init':  2.0, 'min': 2.0, 'max': 2.0, 'fit': False},   # slope []
                       'theta': {'init':  0.0, 'min': -180,'max': 180, 'fit': False},   # position angle [deg]
                       'add': True}],
              'ex': [# ex shear
                    {'norm':  {'init': 1e-2, 'min': 0.0, 'max':0.03, 'fit': False},      # normalization []
                     'theta': {'init': 15.0,  'min':-180, 'max': 180, 'fit': False},      # position angle [deg]
                     'add': True}],
    }
    
    return ranges
    

#> prunes the non-varied params
def pruneParams(ranges=paramRanges()):
    
    #> creating copy (needed)
    ranges = deepcopy(ranges)
    
    #> iterating through ranges to create galaxy dict
    #> if not varied --> collapse
    #> if varied --> store path
    #> if dependant on another value --> store path
    varied_params, depend_params, names = [], [], {}
    for i, profile in enumerate(ranges.keys()):     # each profile (nfw, etc.)
        if profile in ['zl', 'zs', 'pix_arc']: continue # skipping redshifts & conversion
        ranges[profile] = [d for d in ranges[profile] if d['add']] # removing false adds
        for j, dic in enumerate(ranges[profile]):   # each dic in profile array (i.e., [m1, m3, ...]) 
            for k, param in enumerate(dic):         # each param in dic
                if param in ['add', 'm', 'name']:   # params are not fit (skips)
                    if param in ['name']: names[dic[param]] =  dic # saves name for dependant params
                    continue  # skips
                if not dic[param]['fit']:           # if not fitting
                    dic[param] = dic[param]['init'] # sets to init
                    if 'p' in str(dic[param]): depend_params.append([param, dic, deepcopy(dic[param])]) # if param depends on another
                else: varied_params.append([profile, j, param, dic[param]]) # if param is varied
                
    #> need to resolve varied params first then dependant params
    #> converting to numpy for later
    varied_params = np.array(varied_params)
    depend_params = np.array(depend_params)
    
    return ranges, varied_params, depend_params, names


#> toggles paramRanges (and prunes) based on paramDict values
def toggleParams(paramDict, ranges=None):
    
    #> gets default ranges if none provided
    if ranges is None: ranges = paramRanges()
    
    #> iterating through paramRanges
    for i, profile in enumerate(ranges.keys()):         # each profile (nfw, etc.)
        if profile in ['zl', 'zs', 'pix_arc']: continue # skipping redshifts & conversion
        if profile != 'mult': # if not mult
            ranges[profile][0]['add'] = paramDict[profile] # changing add bool
        else:
            mult_dic = deepcopy(ranges[profile][0])     # creating mult dic template
            ranges[profile] = []                        # clearing pre-set
            for j, m in enumerate(paramDict[profile]):  # each dic in profile array (i.e., [m1, m3, ...]) 
                ranges[profile].append(deepcopy(mult_dic))
                ranges[profile][j]['m'] = m
    
    return ranges
    

#> returns batch profiles for deflections based on parameter ranges
def bprofiles(numgals, ranges=paramRanges(), verbose=True, cov=None, mu=None, ub=None, lb=None, **kwargs):
    
    #> pruning non-varied params
    ranges, varied_params, depend_params, names = pruneParams(ranges)
    
    #> sets zl, zs, and pix_arc if provided
    zl = kwargs.get('zl', ranges['zl'])
    zs = kwargs.get('zs', ranges['zs'])
    pix_arc = kwargs.get('pix_arc', ranges['pix_arc'])
    uniform = kwargs.get('uniform', False)
    
    #> number of varied parameters
    d = len(varied_params) # num dims
    
    #> print the varied params
    if True:
        print(f'> The varied params are:')
        for var in varied_params:
            print(f'>   {var[0]}, {var[1]}, {var[2]}')
        print()
    
    #> IM NOT SURE IF I NEED THIS OR IF EVERYTHING AFTER THIS WILL BE OKAY WITH D=0 (i think it's okay)
    if d: # if there are varied parameters (otherwise can skip)
        
        #> creating mean vector and u&l bounds
        if mu is None: mu = np.array([ x['init'] for x in varied_params[:,3] ]) # means (init)
        if ub is None: ub = np.array([ x['max']  for x in varied_params[:,3] ]) # upper bound (max)
        if lb is None: lb = np.array([ x['min']  for x in varied_params[:,3] ]) # lower bound (min)
        
        #> getting parameter range
        rng = np.subtract(ub, lb) # param range
        std_range_perc = kwargs.get('std_range_perc', 10) # sets std based on % of parameter range
        
        #> if using truncated gaussian (the default)
        if not uniform:
        
            #> covariance matrix (identity for now)
            if cov is None:
                cov = np.identity(d)
                cov = cov * (rng * std_range_perc / 100)**2 # squares to std --> var
            
            if verbose: # if want to print cov and std
                # print('> The params being varied are:\n', varied_params[:,0:2])
                print('> The mean vector is:\n', mu)
                print('> The covariance matrix is:\n', cov)
                print('> The standard deviation vector is:\n', np.diag(cov)**0.5)
                print()
            
            #> sampling from the truncated multivariate normal distribution
            tmvn = TruncatedMVN(mu, cov, lb, ub)
            samples = tmvn.sample(numgals)
            samples = np.array(samples) #, dtype=np.float32) #> converts to cupy float32 
            
        #> if wanting a uniform distribution
        elif uniform:
            
            #> sampling from uniform dist
            samples = np.random.uniform(low=lb, high=ub, size=(numgals, d)).T
            
            
    #> creating batch profiles!
    batch_profiles = [] # each element is a galaxy
    for i in range(numgals):
        
        #> setting redshifts & pix_arc
        ranges['zl'] = zl[i]
        ranges['zs'] = zs[i]
        ranges['pix_arc'] = pix_arc[i]
        
        #> setting varied params
        for j, p in enumerate(varied_params):
            prof, ind, param, _ = p
            ranges[prof][ind][param] = samples[j][i]
            
        #> unit conversions
            
        #> setting depend params
        for j, p in enumerate(depend_params):
            param, dic, name = p
            dic[param] = names[name][param] # names[dependant_name][dependant_param]
            
        #> calculating cosmological params (can save in batch_profiles)
        ranges['rhoc']    = cosmology.critDensity(ranges['zl'])              # solMass/kpc^3
        ranges['sigCrit'] = cosmology.sigmaCrit(ranges['zl'], ranges['zs'])  # solMass/kpc^2
        ranges['angDist'] = cosmology.angDist(0, ranges['zl']) * 1e3         # kpc
        
        #> adding calculated params
        #> nfw: logmass, rhoc(z), c --> radius; rhoc(z), c --> rho0
        for prof in ranges.get('nfw', []):
            c = prof['concentration']
            prof['radius'] = (1/c) * ((3. * 10**prof['logmass']) / (800. * 3.1415926536 * ranges['rhoc']))**(1./3.)
            prof['rho0'] = 4 * (200./3.) * (c**3.) / (np.log(1+c) - (c/(1+c))) * ranges['rhoc']
        
        #> hern: Re --> r0; logmass, r0 --> normalization
        for prof in ranges.get('hern', []):
            prof['radius'] = prof['effrad'] * 0.551 # r0 = re*0.551
            prof['norm'] = (10**prof['logmass'])/(2.*np.pi*(prof['radius'])**3)  # radius = r0 (NOT RE!!)
            
        batch_profiles.append(deepcopy(ranges))
    
    return batch_profiles


""" #> PRIORS ========================
================================== """

#> halo mass function
def halo_mass_fn():
    
    
    
    return


#> mass-concentration relation
#> taken from Table 3 of 2014 Dutton & Maccio
# (c_vir or c_200?) and (Delta=200 or given by 2003 Mainini?)
def mass_c_rel(M_vir, zl, h):
    
    #> declarations
    M = (M_vir * h / (1e12))
    
    #> getting coeffs (for c_vir vs. M_vir)
    a = 0.537 + ( (1.025-0.537) * np.exp(-0.718 * zl**(1.08)) )
    b = -0.097 + 0.024*zl
    
    #> mass-concentration relation
    log10_c_vir = a + ( b * np.log10(M) )
    c_vir = 10**(log10_c_vir)
    
    return c_vir


#> stellar-to-halo mass relation
def stellar_h_mass_rel():
    
    
    
    return


#> stellar mass-size relation
def mass_r_relation():
    
    
    
    return


""" #> GRID ==========================
================================== """

#> returns grid
def grid(nph):
    w_range = np.arange(-nph, nph, 1, dtype=float)
    xgrid, ygrid = np.meshgrid(w_range, w_range)
    return xgrid, ygrid
    

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> import for test
    import plot
    import lensing
    import deflection
    
    import modules.parse as parse
    profiles = paramRanges()
    parse.toJSON('paramRanges-2', profiles)
    
    #> grid declarations
    nph = 50
    w_range = np.arange(-nph, nph, 1, dtype=float)
    xgrid, ygrid = np.meshgrid(w_range, w_range)
    pix_arc = 60
    
    numgals = 1
    
    profiles = bprofiles(numgals)
    
    #> calculating deflections
    srt = time.time()
    delx, dely = deflection.deflectionGPU(xgrid, ygrid, profiles)
    # cp.cuda.Stream.null.synchronize()
    
    lensing.lfims(xgrid/pix_arc, ygrid/pix_arc, 
                  delx[0], dely[0],
                  xs=0, ys=0, pix_arc=60, nph=60)

    kappa = lensing.angles_to_kappa(delx[0], dely[0], pix_arc=60)
    plot.kappa(xgrid, ygrid, kappa, pix_arc)
    
    print(time.time()-srt)
    print((time.time()-srt)/numgals)
    

    # end
# thank