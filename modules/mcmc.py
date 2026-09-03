#> name: mcmc.py
#> author: John Miller Jr
#> descrp: mcmc function for Opus (7D project)
#          fits population-level galaxy params

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import random
import time
import numpy as np
import pandas as pd
from copy import deepcopy

#> stats imports
import emcee
import corner
import scipy as sp
from scipy.stats import norm
from scipy.stats import truncnorm
from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN

#> multiprocessing
from multiprocessing import Pool

#> modules
import params
import lensing
import deflection
from modules.units import u; u=u()
import modules.error as error
import modules.parse as parse
import modules.geometry as geometry



""" #> OBSERVED ======================
================================== """



""" #> MODEL =========================
================================== """

def model(theta0, bounds):
    
    #> global params
    global numGals, numQuads_gal
    
    #> unpacking params
    wmu, wcov = theta0
    lb, ub = bounds
    d = len(wmu) # number dimensions
    
    #> recreates covariance matrix
    cov = np.identity(d)
    cov[np.tril_indices_from(cov)] = wcov

    #> sets off diagnols equal to each other (one side must=0!!)
    cov = cov + cov.T - np.diag(cov.diagonal())
    
    #> sampling from the truncated multivariate normal distribution
    tmvn = TruncatedMVN(wmu, cov, lb, ub)
    samples = tmvn.sample(numGals)
    samples = np.array(samples, dtype=np.float32) #> converts to cupy float32 
    
    return


""" #> LIKELIHOOD ====================
================================== """

def logLikelihood():
    
    
    return


""" #> PRIOR =========================
================================== """

#> log prior
def logPrior(profiles):

    #> checking if inside bounds (uniform)
    for i, key in enumerate(profiles.keys()):
        for j, param in enumerate(profiles[key]):

            #> if parameter is not being fit
            if param in ['profile', 'multipole']: continue
            if not profiles[key][param]['fit']: continue

            #> unpacking info
            paramInfo = profiles[key][param]
            high = paramInfo['max']
            low  = paramInfo['min']
            init = paramInfo['init']

            #> checking bounds
            if not low <= init <= high: 
                return -np.inf

    #> checking to make sure NFW logmass > hern logmass
    if profiles['01']['logmass']['init'] < profiles['02']['logmass']['init']:
        return -np.inf

    #> setting up for truncated gaussian (for ex)
    loc, scale = 0.0, 0.02
    a, b = 0.0, 1.0
    logp_exshear = truncnorm.logpdf(0.0, 
                                   (a - loc)/scale, 
                                   (b - loc)/scale, 
                                   loc=loc, 
                                   scale=scale)
    
    return 0.0


""" #> POSTERIOR =====================
================================== """

#> log posterior
def logPosterior(theta, profiles):

    #> saving new model params
    inc = 0
    for i, key in enumerate(profiles.keys()):
        for j, param in enumerate(profiles[key]):

            #> skipping profile & non-fit params
            if param in ['profile', 'multipole']: continue
            if not profiles[key][param]['fit']: continue

            #> assigning value
            profiles[key][param]['init'] = theta[inc]
            inc += 1

    #> calling prior
    prior = logPrior(profiles)
    if not np.isfinite(prior): return -np.inf

    # if inside prior
    return prior + logLikelihood(profiles)


""" #> DATA INPUT =====================
================================== """

#> initial walker arrays/ inputs (mean vector, cov matrix)
def walkArrays(paramRanges):
    
    #> declarations
    wmu = []         # mean vector
    wcov_tri = []    # triangle of cov matrix, flattened
    max_num_std_in_rng = 10 # a=3=(lb-mu)/std; std=(lb-mu)/3
    min_num_std_in_rng = 1 # a=1=(lb-mu)/std; std=(lb-mu)/1
    
    #> copying input
    ranges = deepcopy(paramRanges)
    
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
    
    # print(varied_params)
    # print(depend_params)
        
    #> number of varied parameters
    d = len(varied_params) # num dims
    freep = ((d)*(d+3))/2  # number of free params searched by mcmc
    nWalkers = freep * 2   # number of walkers (determines size of theta0)
    
    #> if no dimensions
    if d == 0: error.phrase('PARAMRANGES HAS NO VARIED PARAMS')
        
    #> creating mean vector and u&l bounds
    mu = np.array([ x['init'] for x in varied_params[:,3] ]) # means (init)
    ub = np.array([ x['max']  for x in varied_params[:,3] ]) # upper bound (max)
    lb = np.array([ x['min']  for x in varied_params[:,3] ]) # lower bound (min)
    rng = np.subtract(ub, lb) # param range
    
    #> randomizing mean vector
    # a, b = (a_trunc - loc) / scale [std], (b_trunc - loc) / scale [std]
    # a, b = (lb - mu) / mean_std, (ub - mu) / mean_std
    
    wmu = np.random.uniform(low=lb, high=ub, size=(len(mu)*2, len(mu)))
    print(wmu)
    
    #> prior on the variances
    N_min = 0.2
    N_max = 20
    N = np.random.uniform(low=N_min, high=N_max, size=(len(mu)*2, len(mu)))
    var = (ub-lb)/N
    
    #> prior on covariances
    wcov = []
    for walk in range(len(N)):
        print(walk)
        dummy = []
        for i in range(d):
            for j in range(d):
                if i==j: dummy.append(var[walk][i])
                if i<=j: continue # if diag or triu
                bound = (var[walk][i]*var[walk][j])**0.5
                dummy.append(np.random.uniform(-bound, bound))
        wcov.append(dummy)
    
    #> getting flattened array + off diag (without triu)
    # wcov = cov[np.tril_indices_from(cov)]
    
    print(var)
    print(wcov)
    
    #> combining input into single array
    theta0 = np.hstack([wmu, wcov])
    print(theta0)
    
    if False: # if want to print cov and std
        # print('> The params being varied are:\n', varied_params[:,0:2])
        print('> The mean vector is:\n', mu)
        print('> The covariance matrix is:\n', cov)
        print('> The standard deviation vector is:\n', np.diag(cov)**0.5)
        
    return theta0


""" #> MAIN MCMC FN ==================
================================== """

#> main mcmc function
def mcmc():
    
    #> declarations
    global numGals, numQuads_gal
    numGals = 100
    numQuads_gal = 10

    #> defining galaxy params (initial value, min, max, and if fitting)
    paramRanges = parse.fromJSON('paramRanges.json')
    
    #> initial walker arrays
    theta0 = walkArrays(paramRanges)
    nwalkers = len(theta0)
    ndim = len(theta0[0])
    print(f'> Num Walkers={nwalkers}, Num Dimensions={ndim}')
    
    #> setting up backend
    chainDir = './mcmc/chains/'
    fileName = chainDir+time.strftime('%y%m%d%H%M', time.localtime()) + '.h5'
    backend = emcee.backends.HDFBackend(fileName)
    backend.reset(nwalkers, ndim) # clears (even though it should be empty)
    
    print(f'> Starting pool with {min(nwalkers, os.cpu_count()-1)} workers')
    srt = time.time()
    with Pool(min(nwalkers, os.cpu_count()-1)) as pool:
        
        #> initializing sampler
        sampler = emcee.EnsembleSampler(nwalkers,         # number of walkers
                                        ndim,             # number of dimensions
                                        logPosterior,     # main function to call
                                        args=(paramRanges,), # extra arguments  
                                        backend=backend,  # backend file
                                        pool=pool)        # multiprocessing
        
        #> sampler declarations
        max_iters = 100000    # max number of iterations
        cTau = 50             # cTau*Tau for convergence
    
        #> running sampler
        flag = True
        converg_int = np.inf
        for sample in sampler.sample(theta0, iterations=max_iters):
            
            #> if converged; breaks
            if sampler.iteration >= (converg_int+max(0.1*converg_int, 1000)): break
        
            #> periodically checks autocorrelation
            if sampler.iteration % 500 == 0 and sampler.iteration > 100:
                
                try: #> if can calculate tau
                    tau = sampler.get_autocorr_time()
                    maxTau = max(tau)
                    print(f'> Iteration {sampler.iteration}: tau = {tau}')   
                    
                    if flag: #> only prints once
                        flag = False                # don't print again
                        converg_int = cTau * maxTau # convergence criteria
                        print(f'> Burn-in time: {3*maxTau:.0f}')
                        print(f'> Number of iterations for convergence: {cTau*maxTau:.0f}')
                        print(f'> Stopping at iteration {converg_int+max(0.1*converg_int, 1000)}')
                        
                except emcee.autocorr.AutocorrError: # if can't calculate tau
                    print(f'> Iteration {sampler.iteration}: chain too short for tau')
                    
    print(f'> Sampling took {time.time()-srt:.0f} seconds')
    
    return sampler


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    mcmc()
    
    # end
# thank