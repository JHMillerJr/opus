#> name: parse.py
#> author: John Miller Jr
#> descrp: functions to read and write to files 
#          also helpful search functions

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import inspect
import numpy as np

#> file imports
import codecs
import json
import pickle

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



""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    file = './data/2607081254.npy'
    data, keys = fromNPY(file)
    print(data['im_obs'][0])
    
    # end
# thank