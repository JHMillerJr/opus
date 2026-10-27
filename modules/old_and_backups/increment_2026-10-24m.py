#> name: increment.py
#> author: John Miller Jr
#> descrp: increments galaxy population params

""" #> IMPORTS =======================
================================== """

#> imports
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

#> adding opus path
sys.path.insert(0, '../opus')

#> modules
import modules.error as error
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


#> checks to see if data already exists (can override)
def check_folder(outFolder):
    
    #> declaration
    path = generate.dataDir + outFolder
    
    #> if folder/data does not exist
    if not (os.path.isdir(path) and os.listdir(path)): return True
    
    #> if it does & do not want to override
    if input(f'> data already exists in {path}, override it? [y/n] ').strip().lower() != 'y': return False
    
    #> if it does & want to override
    shutil.rmtree(path)
    return True


""" #> RUN INFO ======================
================================== """

#> creating bin values for incParams
def bin_points(incParams, bins):
    
    #> declaration
    points = {}
    
    #> iterating thru each profile & params
    for prof in incParams:
        for param in incParams[prof]:
            
            #> getting param min and max, and saving
            _, pdict = default(prof, param)
            points[param_folder(prof, param)] = np.linspace(pdict['min'], pdict['max'], bins).tolist()
            
    return points


#> writes the run settings to ./data/<outFolder>/run_info.json
def write_run_info(outFolder, incParams, bins, kwargs):
    
    #> run info
    info = {'date':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bins':      bins,
            'seeds':     f"seed + bin index (seed = {kwargs.get('seed', 42)})",
            'incParams': incParams,
            'points':    bin_points(incParams, bins),
            'kwargs':    kwargs}
    
    #> saving
    path = generate.dataDir + outFolder
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'run_info.json'), 'w') as f: json.dump(info, f, indent=2, default=str)
    return


""" #> REDRAWS =======================
================================== """

#> counting one bin's redraws
def redraw_row(folder, i, p, redraws):
    
    #> summing over galaxies
    counts = {}
    for r in redraws:
        for k, v in r.items(): 
            counts[k] = counts.get(k, 0) + v
    
    #> row (image counts in order, then what's left)
    nims = sorted(k for k in counts if k != 'left')
    row  = {'param': folder, 'bin': i, 'value': float(p)}
    row |= {f'nims_{k}': counts[k] for k in nims}
    row |= {'left': counts.get('left', 0)}
    
    return row


#> writes every bin's redraws to ./data/<outFolder>/redraws.csv
def write_redraws(outFolder, rows):
    
    #> table
    df   = pd.DataFrame(rows).fillna(0) # missed = 0
    nims = sorted([c for c in df if c.startswith('nims_')], key=lambda c: int(c[5:]))
    df[nims + ['left']] = df[nims + ['left']].astype(int)
    
    #> saving
    df[['param', 'bin', 'value'] + nims + ['left']].to_csv(os.path.join(generate.dataDir + outFolder, 'redraws.csv'), index=False)
    return


""" #> INCREMENT =====================
================================== """

#> main increment function
def inc(incParams, bins, fit=False, **kwargs):
    
    #> unpacking kwargs
    if fit: numGals = kwargs.get('numGals', 1000)      # numGals when fit=True
    else:   numGals = 1
    
    #> run folder inside ./data/
    outFolder = kwargs.pop('outFolder', '+unsorted')
    if not check_folder(outFolder): return
    
    #> kwargs
    kwargs |= {'priors':  kwargs.get('priors', False), # default, no priors
               'numGals': numGals,                     # numGals per bin (=1 for fit=False)
               }
    seed = kwargs.get('seed', 42)                      # base seed (bin i uses seed + i)
    saveDict = generate.getSaveDict(save_all=True)     # saving everything
    
    #> saving run info
    write_run_info(outFolder, incParams, bins, kwargs)
    
    #> interating through profiles
    rows = [] # redraws per bin
    for prof in incParams:
        
        #> iterating through profile parameters
        for param in incParams[prof]:
            
            #> param folder: ./data/<outFolder>/<profile_param>/
            folder   = param_folder(prof, param)
            paramDir = f'{outFolder}/{folder}'
            
            #> print statements
            print()
            error.highlight(f'Working on: {paramDir}')
            
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
                                   **(kwargs | {'seed': seed + i})) # changing seed per bin
                
                #> saving: ./data/<outFolder>/<profile_param>/<time>_<bin>/
                generate.save(out, folder=paramDir, suffix=bin_suffix(i, p), paramRanges=paramRanges, saveDict=saveDict)
                
                #> redraws: ./data/<outFolder>/redraws.csv
                rows.append(redraw_row(folder, i, p, out['redraws']))
                write_redraws(outFolder, rows)
                
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> outfolder (./data/<outFolder>/)
    outFolder = '../../arrival_order/data/increment'
    
    #> declarations
    bins = 19
    numSource = 100000
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
              'ER': True,
              'tau': True,
              'kappa_gamma': True,
              'ER_slope': True,
              'caus_area': True}
    
    #> incrementing
    inc(incParams, bins, **kwargs)

    # end
# thank