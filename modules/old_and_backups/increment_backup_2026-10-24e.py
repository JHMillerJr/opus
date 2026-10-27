#> name: increment.py
#> author: John Miller Jr
#> descrp: increments galaxy population params

""" #> IMPORTS =======================
================================== """

#> imports
import os
import sys
import shutil
import numpy as np
import pandas as pd

#> adding opus path
sys.path.insert(0, '../opus')

#> modules
import generate
import params


""" #> DEFAULTS ======================
================================== """

#> returns defaults when incrementing certain params
def default(profile, param):
    
    #> declarations
    mult, ex = [], False
    
    #> setting galProfiles param
    if profile in ['mult']:
        mult.append(param[0])
        param = param[1]
    elif profile in ['ex']: ex = True
        
    #> getting galProfs
    galProf = generate.galProfiles(nfw=True, hern=True, mult=mult, ex=ex)
    
    #> getting initial param dict
    paramRanges = params.toggleParams(galProf)
    
    #> changing axisrat both
    if profile in ['both']:
        profile = 'nfw'
        paramRanges['hern'][0]['axisrat']['init'] = 'p01'
    else:
        paramRanges['hern'][0]['axisrat']['init'] = paramRanges['nfw'][0]['axisrat']['init']
        
    #> getting the parameter dictionary
    pdict = paramRanges[profile][0][param]
    
    #> setting values of 'other' parameters 
    #  i.e., prof=m, param=[1,theta], then m1_norm=0.1
    if profile in ['mult', 'ex'] and param in ['theta']:
        paramRanges[profile][0]['norm']['init'] = 0.01

    return paramRanges, pdict


""" #> FOLDERS =======================
================================== """

#> folder of an incremented param (e.g., nfw_axisrat, mult_m1_norm)
def param_folder(profile, param):
    if profile in ['mult']: return f'{profile}_m{param[0]}_{param[1]}'
    return f'{profile}_{param}'


#> file-name suffix of one bin (e.g., _bin00_0.5)
def bin_suffix(i, p):
    return f'_bin{i:02d}_{p:.4g}'


#> asks before overriding an out folder that already has data (False = stop)
def check_folder(outFolder):
    path = generate.dataDir + outFolder
    if not (os.path.isdir(path) and os.listdir(path)): return True
    if input(f'> data already exists in {path}, override it? [y/n] ').strip().lower() != 'y': return False
    shutil.rmtree(path)
    return True


""" #> INCREMENT =====================
================================== """

#> main increment function
def inc(incParams, bins, fit=False, **kwargs):
    
    #> unpacking kwargs
    if fit: numGals = kwargs.get('numGals', 1000)      # numGals when fit=True
    else:   numGals = 1
    
    #> run folder inside ./data/ (not passed to gen)
    outFolder = kwargs.pop('outFolder', '+unsorted')
    if not check_folder(outFolder): return
    
    #> kwargs
    kwargs |= {'priors':  kwargs.get('priors', False), # default, no priors
               'numGals': numGals,                     # numGals per bin (=1 for fit=False)
               }
    
    #> interating through profiles
    for prof in incParams:
        
        #> iterating through profile parameters
        for param in incParams[prof]:
            
            #> param folder: ./data/<outFolder>/<profile_param>/
            paramDir = f'{outFolder}/{param_folder(prof, param)}'
            
            #> setting paramRanges & binning
            paramRanges, pdict = default(prof, param)
            points = np.linspace(pdict['min'], pdict['max'], bins)
            
            #> changing 'both' to nfw (hern should have 'p01' in shared values)
            key = 'nfw' if prof in ['both'] else prof
            if prof in ['mult'] and isinstance(param, list): param = param[1]
            
            #> iterating through each point
            for i, p in enumerate(points):

                #> changing value
                paramRanges[key][0][param]['init'] = float(p)
            
                #> lensing!
                out = generate.gen(paramRanges=paramRanges,
                                   **kwargs)
                
                #> saving: ./data/<outFolder>/<profile_param>/<time>_<bin>/
                generate.save(out, folder=paramDir, suffix=bin_suffix(i, p), paramRanges=paramRanges)
                
            
        
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> outfolders (./data/<outFolder>/)
    outFolder = '../../arrival_order/data/increment'
    
    #> declarations
    bins = 19
    numSource = 10000
    redshifts = (0.5, 1.0)
    incParams = {'nfw':  ['axisrat', 'theta', 'x0', 'y0'], 
                 'hern': ['axisrat'], 
                 'both': ['axisrat'],
                 'mult': [[1,  'norm'],
                          [1, 'theta'], 
                          [3,  'norm'], 
                          [4,  'norm']], 
                 'ex':   ['norm', 'theta']}
    
    #> kwargs
    kwargs = {'method': 'lpot',
              'numSource': numSource,
              'seed': 42,
              'nfw_priors': False,
              'hern_priors': False,
              'add': ['d23'],
              'outFolder': outFolder,
              'redshifts': redshifts,
              'scprox': True,
              'order_comp': True,
              'ER': True}
    
    #> incrementing
    inc(incParams, bins, **kwargs)

    # end
# thank