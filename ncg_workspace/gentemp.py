#> name: gentemp.py
#> author: John Miller Jr
#> descrp: template file to generate galaxy/quad populations from generate.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import numpy as np

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__)) 


""" #> GENERATE FN ===================
================================== """

#> generates a galaxy/ quad population based on the given params
def function():
    
    #> imports  (CHANGE IF GENERATE IS IN DIFF DIR)
    import generate
    
    #> declarations
    numGals       = 1                          # total # of galaxies to generate
    numSource_gal = 1                          # total # of sources per galaxy
    
    #> output kwargs
    folder   = '+unsorted'                     # dir in dataDir to save data
    suffix   = ''                              # suffix to add to file names
    verbose  = True                            # if wanting extra print info
    timeFlag = False                           # if wanting time info
    saveFlag = generate.getSaveDict(False)     # what to save (images=im_obs=sources=bprofiles=True, im_mags=lens=False)
    plotFlag = generate.getPlotDict(True, kappa=False)     # what to plot (kappa=caustics=True, deflect=caustics_zoom=False)
    
    #> grid kwargs
    scale_factor = 1                           # increase (or decrease) grid resolution [default=1]
    pix_arc      = 86.46                         # pixel per arcsec conversion [default=60]
    nph          = 50                          # width / 2 of grid [default=50)
    
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
    zl = 0.7                                   # redshift of lens
    zs = 1.0                                   # redshift of source
    # redshifts = (0.37374414, 1.18590195)
    redshifts = (zl, zs)                       # combined redshifts (for code)
    # redshifts = generate.ranRedshifts(numGals)
    print(redshifts)
    
    #> image properties kwargs
    jims = None                                   # number of request images from each source (5=quad)
    mags = False                               # if wanting image magnifications (will change saveFlag automatically)
    observables = None # ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23'] # requested lensing observables
    
    #> paramRanges: can be edited to change param values, fit, etc.
    if True:
        
        #> imports
        import params
        
        #> getting paramRanges
        paramRanges = params.toggleParams(galProfs) # the parameter ranges and values, can be edited

        #> place to edit values
        #> structure: dict.keys() = ['nfw', 'hern', 'mult', 'ex'], np.array [0, ...], 
        #>            dict.keys() = ['x0', ...], dict.keys() = ['init', 'min', 'max', 'fit']
        
        #> example
        paramRanges['nfw'][0]['logmass']['init'] = 13
        
    else: paramRanges = None
        
    #> computatoin kwargs
    gpu = False                                # CURRENTLY NO GPU IMPLEMENTATION
    
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
                    paramRanges=paramRanges,      # the galaxy params: ub, lb, fit or not, etc.
                    lb=lb,                        # lower bounds for gal params
                    ub=ub,                        # upper bounds for gal params
                    mu=mu,                        # mu vector for gal params
                    cov=cov,                      # covariance matrix for gal params
                    uniform=uniform,              # if want to sample uniformly
                    redshifts=redshifts,          # redshifts (or distribution of)
                    jims=jims,                    # requested number of images per source
                    mags=mags,                    # if want image mags
                    observables=observables,      # requested lensing observables
                    gpu=gpu)                      # if want to run on gpu (CURRENTLY NO GPU IMPLEMENTATION)
    
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