#> name: parse.py
#> author: John Miller Jr
#> descrp: functions to read and write to files 
#          also helpful search functions

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import glob
import inspect
import numpy as np
import pandas as pd

#> file imports
import codecs
import json
import pickle

#> larger folder glob
from pathlib import Path

#> modules
from modules import error


""" #> DIR/FILE CHECKING =============
================================== """

#> checks to see if dir exists, if not, then creates
def checkDir(dir):
    if not os.path.isdir(dir):
        print(f'> The directory {dir} has been created.')
        os.mkdir(dir)
    else: print(f'> The directory {dir} already exists.')
    return

#> checks to see if file exists
def checkFile(fileName):
    return os.path.isfile(fileName)


""" #> ARRAY SEARCH ==================
================================== """

#> returns index of word in N-D array
def wordSearch(array, word):
    for i, block in enumerate(array):
        try: 
            return i, block.index(word)
        except: 
            for j, cell in enumerate(block):
                if word in cell:
                    return i, j
    return '', ''



""" #> READ FNS ======================
================================== """

#> reads in json file, returns numpy array
def fromJSON(fileName):
    
    #> adjusting file name if needed & checks existance
    if not '.json' in fileName: fileName += '.json'
    if not os.path.isfile(fileName): error.phrase(f'> The file {fileName} does not exist!')
    
    #> reading file & returning data
    jsonDump = codecs.open(fileName, 'r', encoding='utf-8').read()
    jsonData = json.loads(jsonDump)
    return jsonData

#> reads in pickle file, returns mainParams object
def fromPickle(fileName):
    
    #> adjusting file name if needed & checks existance
    if not '.pickle' in fileName: fileName += '.pickle'
    if not os.path.isfile(fileName): error.phrase(f'> The file {fileName} does not exist!')
    
    #> reading pickle & returning data
    with open(fileName, 'rb') as F:
        unserialized_data = pickle.load(F)
    return unserialized_data

#> reads in numpy array, returns data and keys
def fromNPY(fileName):
    
    #> adjusting file name if needed & checks existance
    if not '.npy' in fileName: fileName += '.npy'
    if not os.path.isfile(fileName): error.phrase(f'> The file {fileName} does not exist!')
    
    #> reading file & returning data
    data = dict(np.load(fileName, allow_pickle=True).item())
    return data, data.keys()

#> reads in NPY to large dataframe (data is remaining skipped keys)
def fromNPYtoDF(fileName, observables=['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23']):
    
    #> imports
    import pandas as pd
    
    #> declarations
    columns = {'images':  ['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5'],
               'sources': ['xs', 'ys'],
               'im_mags': ['mu1', 'mu2', 'mu3', 'mu4', 'mu5'],
               'im_obs': observables}
    
    #> loading in data
    data, keys = fromNPY(fileName)
    
    #> iterating through keys
    df = pd.DataFrame()
    popkeys = []
    for key in keys:
        if key in ['lens', 'bprofiles']: continue
        if key in ['images']:
            if len(data[key][0]) != 5: error.phrase('The data in {fileName} are not quads!')
            df_dummy = pd.DataFrame(data[key].reshape(len(data[key]), 10), columns=columns[key])
        else: 
            df_dummy = pd.DataFrame(data[key], columns=columns[key])
        df = pd.concat([df, df_dummy], axis=1)
        popkeys.append(key)
    
    #> popping keys
    for key in popkeys:
        data.pop(key)
        
    #> checking for NANs
    if df.isna().any().any():
        error.phrase('NAN found while trying to load {fileName}!')
    
    return df, data


#> loads every file in a generate.save() folder: <dirPath>/<fileName>_<key>.npy (+ <fileName>_df.csv)
#> returns a dict keyed by the file ending: images, obs, mags, delays, srcs, nims, flag, gal, bprofiles, paramRanges, df, ...
def load(dirPath, verbose=False, **kwargs):
    
    #> unpacking kwargs
    requested_keys = kwargs.get('keys', None)   # only load these keys (default: every file)
    
    #> formatting
    dirPath = str(dirPath).replace('\\', '/')
    if dirPath[-1] != '/': dirPath += '/'
    
    #> ensuring that folder exists
    if not os.path.isdir(dirPath): error.phrase(f'{dirPath} does not exist')
        
    #> seeing what npy + csv files exist in dir
    files = sorted(glob.glob(dirPath + '*.npy') + glob.glob(dirPath + '*.csv'))
    if len(files) == 0: error.phrase(f'Could not find any .npy or .csv files in {dirPath}')
    
    #> key = file ending after the last '_' (the fileName itself may contain '_' from its suffix)
    organized_files = {}
    for file in files:
        stem = os.path.splitext(os.path.basename(file))[0]
        key  = stem.rsplit('_', 1)[-1]
        if key in organized_files: error.highlight(f'More than one file ending in _{key} in {dirPath}, keeping {os.path.basename(file)}')
        organized_files[key] = file
    
    #> reducing to only requested if requested
    if requested_keys is not None:
        missing = [key for key in requested_keys if key not in organized_files]
        if len(missing) > 0: error.phrase(f'Could not find files ending in {missing} in {dirPath}')
        organized_files = {key: organized_files[key] for key in requested_keys}
    
    #> loading (csv --> dataframe, dict saved as npy --> dict)
    dfs = {}
    for key, file in organized_files.items():
        if file.endswith('.csv'): dfs[key] = pd.read_csv(file); continue
        data = np.load(file, allow_pickle=True)
        dfs[key] = data.item() if (data.dtype == object and data.ndim == 0) else data

    #> if verbose
    if verbose:
        for key in dfs: print(f'> {key:12s}: {getattr(dfs[key], "shape", type(dfs[key]).__name__)}')
        
    return dfs


""" #> GEN OUTPUT ====================
================================== """

#> key of galaxy
def bnum(i, numGals):
    return str(i+1).zfill(max(4, len(str(numGals))))

#> converst gen() output split by galaxy ('0001', '0002', ...) to one row per source
#> keys: images, mags, delays, srcs, nims, flag, gal, obs, bprofiles (+ paramRanges, df)
def stack(out, **kwargs):
    
    #> imports
    import newtlens
    
    #> kwargs
    paramRanges = kwargs.get('paramRanges', None)   # galaxy params
    df          = kwargs.get('df', True)            # bprofiles as a dataframe (_df.csv)
    
    #> galaxies in order + common width
    gals = [out[k] for k in sorted(out.keys())]
    w    = max(o['mu'].shape[1] for o in gals)
    
    #> one row per source
    data = {'images': np.concatenate([newtlens.pad(g['images'], w) for g in gals]),  # (N, w, 2) arcsec
            'mags':   np.concatenate([newtlens.pad(g['mu'], w) for g in gals]),      # (N, w) signed mags
            'delays': np.concatenate([newtlens.pad(g['delays'], w) for g in gals]),  # (N, w) days
            'srcs':   np.concatenate([g['srcs'] for g in gals]),                     # (N, 2) arcsec
            'nims':   np.concatenate([g['nims'] for g in gals]),                     # (N,)
            'flag':   np.concatenate([g['flag'] for g in gals]),                     # (N,)
            'gal':    np.concatenate([np.full(len(g['nims']), i) for i, g in enumerate(gals)])} # (N,) index into bprofiles
    if all('obs' in g for g in gals): data['obs'] = np.concatenate([g['obs'] for g in gals]) # (N, numObs)
    
    #> galaxy info (one per galaxy)
    data['bprofiles'] = np.array([g['bprof'] for g in gals], dtype=object)
    if paramRanges is not None: data['paramRanges'] = paramRanges
    if df: data['df'] = bprof_to_df(bprofiles=data['bprofiles'])
    
    return data

#> converts one row per source to gen() output split by galaxy ('0001', '0002', ...)
def unstack(data):
    
    #> declarations
    numGals = len(data['bprofiles'])
    keys    = {'images': 'images', 'mu': 'mags', 'srcs': 'srcs', 'nims': 'nims',
               'flag': 'flag', 'delays': 'delays', 'obs': 'obs'}
    
    #> iterating thru galaxies
    out = {}
    for i in range(numGals):
        
        #> sources of this galaxy + its width (max(jmax=5, nims), as in newtlens.find_images)
        m = data['gal'] == i
        w = max(5, int(data['nims'][m].max(initial=0))) if 'nims' in data else None
        
        #> splitting (image-slot arrays trimmed back to this galaxy's width)
        o = {}
        for key, skey in keys.items():
            if skey not in data: continue
            o[key] = data[skey][m][:,:w] if key in ['images', 'mu', 'delays'] else data[skey][m]
        o['bprof'] = data['bprofiles'][i]
        
        #> saving
        out[bnum(i, numGals)] = o
    
    return out


""" #> WRITE FNS =====================
================================== """

#> encodes numpy array for json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

#> writes array to json file
def toJSON(fileName, array):
    if not '.json' in fileName: fileName += '.json'
    with open(fileName, 'w') as F:
        json.dump(array, F, cls=NumpyEncoder)
    print(f'> Created json @ {fileName}')
    return

#> pickles cucumber (pickles python object)
def toPickle(fileName, cucumber):
    if not '.pickle' in fileName: fileName += '.pickle'
    with open(fileName, 'wb') as F:
        pickle.dump(cucumber, F)
    print(f'> Created pickle @ {fileName}')
    return


""" #> BPROFILES TO DF ===============
================================== """

#> loads bprofiles depending on input and output
#> input options: csv, npy, bprofiles, df
#> output options: bprofiles, df
def load_bprofiles(fileName, input='csv'):
    
    #> loads csv
    if input == 'csv': return pd.read_csv(fileName)
    
    #> loads npy
    if input == 'npy': return np.load(fileName, allow_pickle=True)
    
    #> loads bprofiles
    if input == 'bprofiles': return bprof_to_df(bprofiles=fileName, save=False)
    
    error.phrase(f'load_bprofiles() does not understand input={input}')
    return

#> summarizing bprofiles to df
def bprof_to_df(bprofiles=None, fileName='', save=False, outFile='bprofiles_df.csv'):
    
    #> checking to ensure at least one thing is provided
    if bprofiles is None and fileName == '':
        error.phrase('Need to supply either bprofiles or its fileName!')
    
    #> loading if given file
    if fileName != '': bprofiles = np.load(fileName, allow_pickle=True)
    
    #> bprofiles should be defined by now (one row per galaxy, values keep their type)
    rows = []
    for i, b in enumerate(bprofiles):
        
        #> getting column names
        row = {}
        for key in b.keys():
            
            #> if singular value (float, int, ...)
            if np.isscalar(b[key]): 
                row[key] = b[key]
                continue
            
            #> iterating through dictionaries
            for array in b[key]: # for each array inside prof
                for sub_key in array.keys():
                    
                    #> chaning mult key
                    if key in ['mult']: key_ = key + str(array['m'])
                    else: key_ = key
                    
                    #> appending
                    row[f'{key_}_{sub_key}'] = array[sub_key]
        
        #> appending to rows
        rows.append(row)
    
    #> dataframe
    df = pd.DataFrame(rows)
    
    # print(df.to_string())
    if save:
        df.to_csv(outFile, index=False)
        
    return df

#> converts parameter keys into min/max arrays
def keys_to_ranges(keys, ranges=None):
    
    #> imports
    import params
    
    #> getting parameter ranges
    if ranges is None: ranges = params.paramRanges()
    
    #> iterating through profiles
    mins, maxs = {}, {}
    for prof in ['nfw', 'hern', 'mult', 'ex']:
        
        for array in ranges[prof]:
            
            #> changing mult key
            if prof == 'mult': prof_ = prof + str(array['m'])
            else: prof_ = prof
            
            #> iterating through parameters
            for key in array.keys():
                
                #> only parameters with ranges
                if isinstance(array[key], dict) and 'min' in array[key]:
                    mins[f'{prof_}_{key}'] = array[key]['min']
                    maxs[f'{prof_}_{key}'] = array[key]['max']
    
    #> getting requested ranges
    dmin = np.array([mins[key] for key in keys], dtype=float)
    dmax = np.array([maxs[key] for key in keys], dtype=float)
    
    return dmin, dmax

#> batch convert to txt
def batch(loc, pattern, target):
    
    #> iterating through files 
    files = Path(loc).rglob(pattern)
    for file in files:
        new_fileName = target + (str(file)[:-4] + '_df.csv').replace('\\', '/').split('/')[-1]
        bprof_to_df(fileName=file, outFile=new_fileName, save=True)
    
    return

""" #> TO LLRW =======================
================================== """

#> batch convert to txt
def batch_to_txt(loc, pattern, target):
    
    #> iterating through files 
    files = Path(loc).rglob(pattern)
    for file in files:
        npy_to_txt(file, target)
    
    return

#> converts numpy to txt for liliya
def npy_to_txt(fileName, target=''):
    
    #> opening file
    data = np.load(fileName, allow_pickle=True)
    
    #> moving to target dir if requested
    if target == '':
        new_fileName = (str(fileName)[:-3] + 'txt')
    else:
        new_fileName = target + (str(fileName)[:-3] + 'txt').replace('\\', '/').split('/')[-1]
        
    #> saving to txt
    np.savetxt(new_fileName, data, newline='\n')
    
    return

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    dirPath = './data/+unsorted/2607171519'
    # load(dirPath, verbose=True, keys=['images'])
    
    loc = '../opus_lmfi/data/260909_comp1/'
    pattern = '*_bprofiles.npy'
    # batch_to_txt(loc, pattern, loc)
    
    file = loc + '26090914180800_pop01/26090914180800_pop01_bprofiles.npy'
    # bprof_to_df(fileName=file, save=True)
    batch(loc, pattern, loc)
    
    #data, keys = fromNPY(file)
    #print(data['im_obs'][0])
    
    # end
# thank