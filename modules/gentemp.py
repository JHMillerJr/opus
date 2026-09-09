#> name: gentemp.py
#> author: John Miller Jr
#> descrp: template file to generate galaxy/quad populations from generate.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import numpy as np
import pandas as pd

#> adding dir to sys paths (for MSI)
sys.path.append(os.path.dirname(__file__)) 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> GENERATE FN ===================
================================== """

#> generates a galaxy/ quad population based on the given params
def function():
    
    #> imports  (CHANGE IF GENERATE IS IN DIFF DIR)
    import generate
    import params
    from modules.units import u; u=u()
    
    #> declarations
    numGals       = 10                         # total # of galaxies to generate
    numSource_gal = 10                         # total # of sources per galaxy
    
    #> random seed
    seed = 42                                  # numpy random seed
    
    #> cosmology
    cosmo = u.cosmo                            # default cosmology dict
    cosmo['h0'] = 70.0                         # hubble param
    cosmo['omega_m_0'] = 0.27                  # matter density
    cosmo['omega_lam_0'] = 1-cosmo['omega_m_0']# cosmological constant
    
    #> output kwargs
    folder   = '+unsorted'                     # dir in dataDir to save data
    suffix   = ''                              # suffix to add to file names
    verbose  = False                           # if wanting extra print info
    timeFlag = False                           # if wanting time info
    saveFlag = generate.getSaveDict(False)     # what to save (images=im_obs=sources=bprofiles=True, im_mags=lens=False)
    plotFlag = generate.getPlotDict(False)     # what to plot (kappa=caustics=True, deflect=caustics_zoom=False)
    
    #> grid kwargs
    scale_factor = 1                           # increase (or decrease) grid resolution [default=1]
    pix_arc      = 60                          # pixel per arcsec conversion [default=60]
    nph          = 50                          # width / 2 of grid [default=50)
    pix_arc = np.tile(pix_arc, numGals)        # array for pix_arc
    
    #> galaxy profiles kwargs (what profiles to add)
    nfw  = True                                # adds a nfw profile [default=True]
    hern = True                                # adds a hernquist profile [default=True]
    mult = []                                  # adds multipole profiles corresponding to the numbers given [default=[]]
    ex   = False                               # adds external shear [default=False]
    galProfs = generate.galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex) # dictionary handled by the code
    
    #> galaxy profile params kwargs (values of said profiles)
    lb  = None                                 # lower bounds vector of varied galaxy params [example=[0, 0]]
    ub  = None                                 # upper bounds vector ... [example=[2, 2]]
    mu  = None                                 # mean vector         ... [example=[1, 1]]
    cov = None                                 # covariance matrix   ... [example=np.identity(n=2)]
    uniform = False                            # if sampling from uniform dist, i.e., (lb, ub), False=TruncNorm
    
    #> redshifts
    supplyRed = False                          # if wanting to supply redshifts (otherwise, draws from priors)
    if supplyRed:                              # can either be array or single
        zl = 0.5                               # redshift of lens
        zs = 1.0                               # redshift of source
        redshifts = np.tile((zl,zs), (numGals,1)) # combined redshifts (for code)
    else:
        redshifts = None                       # must be none (will draw from priors if red=True)
    
    #> priors (right now: turning on one turns on ALL)
    hmf  = True                                # halo mass function
    cmr  = True                                # concentration-mass relation
    shmr = True                                # stellar-to-halo mass relation
    msr  = True                                # stellar mass-size relation
    red  = not supplyRed                       # redshift distribution
    priorDict = params.getPriorDict(hmf=hmf, cmr=cmr, shmr=shmr, msr=msr, red=red) # if wanting to draw from priors (hmf, cmr, shmr, msr)
    
    #> image properties kwargs
    jims = 5                                   # number of request images from each source (5=quad)
    mags = None                                # if wanting image magnifications (will change saveFlag automatically)
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23'] # requested lensing observables
    
    #> bprofiles & paramRanges (can be edited)
    paramRanges = None
    bprofiles = None
    
    #> paramRanges: can be edited to change param values, fit, etc.
    if True:
        
        #> getting paramRanges
        paramRanges = params.toggleParams(galProfs) # the parameter ranges and values, can be edited
        
        #> place to edit values
        #> structure: dict.keys() = ['nfw', 'hern', 'mult', 'ex'], np.array [0, ...], 
        #>            dict.keys() = ['x0', ...], dict.keys() = ['init', 'min', 'max', 'fit']
        
        #> example
        # paramRanges['nfw'][0]['axisrat']['fit'] = True
        
        pass


    #> generating!
    generate.genPop(numGals,                      # # galaxies
                    numSource_gal=numSource_gal,  # # sources per galaxy
                    folder=folder,                # data folder to save to
                    suffix=suffix,                # suffix to file/folder name
                    verbose=verbose,              # if want print info
                    timeFlag=timeFlag,            # if want time info
                    saveFlag=saveFlag,            # what to save
                    plotFlag=plotFlag,            # what to plot
                    scale_factor=scale_factor,    # scaling window/ changing pixel resolution
                    pix_arc=pix_arc,              # pixel per arcsec conversion for grid
                    nph=nph,                      # width / 2 for grid
                    galProfs=galProfs,            # galaxy profiles
                    priorDict=priorDict,          # what to draw from priors
                    paramRanges=paramRanges,      # the galaxy params, (see params)
                    bprofiles=bprofiles,          # the main array used to gen galaxies
                    lb=lb,                        # lower bounds for gal params
                    ub=ub,                        # upper bounds for gal params
                    mu=mu,                        # mu vector for gal params
                    cov=cov,                      # covariance matrix for gal params
                    uniform=uniform,              # if want to sample uniformly
                    redshifts=redshifts,          # redshifts (or distribution of)
                    jims=jims,                    # requested number of images per source
                    mags=mags,                    # if want image mags
                    observables=observables,      # requested lensing observables
                    seed=seed,                    # random seed
                    gpu=False)                    # if want to run on gpu (CURRENTLY NO GPU IMPLEMENTATION)
    
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    #> calling generate function
    function()
    
    # end
# thank