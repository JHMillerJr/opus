#> name: metric.py
#> author: John Miller Jr
#> descrp: calculates various cosmological params for lensing

""" #> IMPORTS =======================
================================== """

#> standard imports
import glob
import numpy as np
import pandas as pd
import scipy as sp

#> larger folder glob
from pathlib import Path

#> wasserstein
from ot import sliced_wasserstein_distance as swd

#> modules
from modules.units import u; u=u()
import modules.error as error
import modules.parse as parse

#> data dir
dataDir = './data/'


""" #> GRID COMP =====================
================================== """

#> returns ranges for each requested obesrvable
def paramRanges(observables):
    
    #> ranges
    rangesDict = {'t23': (0, 100),
                  'dt23': (-5, 5),
                  'd4/d1': (0, 1)}
    
    #> collecting ranges
    ranges = []
    for obs in observables:
        ranges.append(rangesDict[obs])
    
    return ranges


#> comparison of two quad populations based on square grid
def gridComp(inFile1, inFile2, bins, observables):
    
    #> getting parameter ranges
    ranges = paramRanges(observables)
    numParams = len(ranges)
    
    #> collecting data
    data1 = np.load(inFile1)
    data2 = np.load(inFile2)
    
    #> checking to make sure dimensions are the same
    dims_dont_have_to_match = True # if that is okay (mostly for testing)
    if numParams != len(data1[0]):
        if dims_dont_have_to_match:
            error.highlight('Dimensions do not match, but continuing.')
        else:
            error.phrase('DIMENSIONS DO NOT MATCH')
            
    #> initializing grids
    grid1 = np.zeros(shape=(np.full(numParams, bins)))
    grid2 = np.copy(grid1)
    
    #> populating both grids
    for data, grid in zip([data1, data2], [grid1, grid2]):
        
        #> iterating through data
        for row in data:
            
            #> iterating through each param in a row
            indx = []
            for i, value in enumerate(row):
                
                #> getting param ranges
                vmin, vmax = ranges[i]
                prange = vmax - vmin
                bin_width = prange / bins
                
                #> getting index & populating
                index = int((value-vmin) / bin_width)
                
                # print(value, vmin, vmax, prange, bin_width, index)
                if index >= bins or index < 0: indx.append(np.nan)
                else: indx.append(index)

            #> adds index UNLESS it falls outside the bounds (in any dim)
            if np.isnan(indx).any(): continue
            else: grid[*indx] += 1
            
        #> normalizing grids
        grid /= np.sum(grid)
    
    #> comparing grids
    fgrid1 = grid1.flatten()
    fgrid2 = grid2.flatten()
    diff_array = (grid1 - grid2).flatten()
    
    #> calculating p
    p_val = 0
    for i, diff in enumerate(diff_array):
        if diff * fgrid2[i] == 0: continue # if either are zero
        p_val += ( abs(diff)**2 / fgrid2[i] )
    
    return np.e**(-p_val)


""" #> WASSERSTEIN ===================
================================== """

#> normalizes the data using fixed limits
def normData(data, dmin, drange):
    
    #> reformatting
    if len(data.shape) == 1: dim = 1
    else: dim = data.shape[1]
    data = data.reshape(len(data), dim)
    
    #> checking on range
    drange = np.where(drange != 0.0, drange, 1.0)
    
    #> normalizing
    dnorm = (data - dmin) / drange
    
    return np.array(dnorm)

#> gets normalization limits
def normScale(data=None, dmin=None, dmax=None):
    
    #> getting scale from data
    if dmin is None or dmax is None:
        
        #> reformatting
        if len(data.shape) == 1: dim = 1
        else: dim = data.shape[1]
        data = data.reshape(len(data), dim)
        
        #> getting values
        dmin = np.min(data, axis=0)
        dmax = np.max(data, axis=0)
    
    #> getting range
    drange = np.array(dmax) - np.array(dmin)
    
    #> checking on range
    drange = np.where(drange != 0.0, drange, 1.0)
    
    return np.array(dmin), drange

#> main wasserstein metric function
def wass(df1, df2, keys=None, method='ND', slices=100, scale=None, return_all_1d=False, **kwargs):
    
    #> getting keys if not provided
    if keys is None:
        keys1 = df1.keys()
        keys2 = df2.keys()
        if not keys1.equals(keys2):
            error.hightlight('Did not provide keys AND keys1 != keys2')
        keys = keys1
    
    #> getting data
    data1 = df1[keys].to_numpy()
    data2 = df2[keys].to_numpy()
    
    #> getting normalization
    if scale is None:
        dmin, drange = normScale(data=data1)
    else:
        dmin, drange = scale
    
    #> normalizing
    data1 = normData(data1, dmin, drange)
    data2 = normData(data2, dmin, drange)
    
    #> marginalized distributions
    if method == '1D':
        
        #> iterates through dimensions
        sw_dists = []
        for i, key in enumerate(keys):
            sw_dists.append(swd(data1[:,i:i+1],
                                data2[:,i:i+1],
                                n_projections=1))
        
        sw_dists = np.array(sw_dists)
        sw_distance = np.sum(sw_dists)
        
        #> returning individual distances
        if return_all_1d:
            return sw_distance, dict(zip(keys, sw_dists))
        
    #> joint distributions
    elif method == 'ND':
        
        sw_distance = swd(data1,
                          data2,
                          n_projections=slices, 
                          seed=kwargs.get('seed', None))

    # print(f"> {method}-SWD: {sw_distance}")
    
    return sw_distance


#> wasserstein distance for galaxy properties
#> two methods: 1D = compares all dists in 1D, then adds Wass
#               ND = sliced wassersten metric
#> input methods: csv = bprofiles.csv
#                 npy = bprofiles.npy
#                 bprofiles = bprofiles array
#                 df = dataframe
#                 np = numpy array
def wassXi(fileName1, fileName2, **kwargs):
    
    #> kwargs
    keys = kwargs.get('keys', None)
    inpt = kwargs.get('input', 'csv')
    
    #> default keys
    if keys is None:
    
        #> default keys (removed zl and zs)
        keys = ['nfw_logmass', 'nfw_concentration', 'nfw_axisrat', 'nfw_theta', # nfw params
                'hern_logmass', 'hern_effrad', 'hern_axisrat',                  # hern params
                'mult1_norm', 'mult1_theta', 'mult3_norm', 'mult4_norm',        # multipole params
                'ex_norm', 'ex_theta']                                          # ex params
        
    #> gets min and max for each key
    dmin, dmax = parse.keys_to_ranges(keys)
    drange = dmax - dmin
    
    #> updating kwargs
    kwargs |= {'keys': keys, 'scale': (dmin, drange)}
    
    #> loading data
    if inpt not in ['df', 'np']:
        df1 = parse.load_bprofiles(fileName=fileName1, input=inpt)
        df2 = parse.load_bprofiles(fileName=fileName2, input=inpt)
    else:
        if inpt == 'df': df1, df2 = fileName1, fileName2 # if already dataframes
        if inpt == 'np': 
            if keys is None: error.phrase('Need keys!')
            df1, df2 = pd.DataFrame(fileName1, columns=keys), pd.DataFrame(fileName2, columns=keys)
    
    #> calling wass
    return wass(df1, df2, **kwargs)


#> wasserstein distance for lensing observables
#> two methods: 1D = compares all dists in 1D, then adds Wass
#               ND = sliced wassersten metric
#> input methods: csv = bprofiles.csv
#                 npy = bprofiles.npy
#                 bprofiles = bprofiles array
#                 df = dataframe
#                 np = numpy array
def wassTheta(fileName1, fileName2, **kwargs):
    
    #> kwargs
    keys = kwargs.get('keys', None)
    inpt = kwargs.get('input', 'npy')
    
    #> loading data
    if inpt not in ['df', 'np']:
        df1 = parse.load_bprofiles(fileName=fileName1, input=inpt)
        df2 = parse.load_bprofiles(fileName=fileName2, input=inpt)
    else:
        if inpt == 'df': df1, df2 = fileName1, fileName2 # if already dataframes
        if inpt == 'np': 
            if keys is None: error.phrase('Need keys!')
            df1, df2 = pd.DataFrame(fileName1, columns=keys), pd.DataFrame(fileName2, columns=keys)

    #> checking to see if object is a dataframe
    if not isinstance(df1, pd.DataFrame) and keys is None:
        error.phrase('Need keys!')
    else:
        if not isinstance(df1, pd.DataFrame):
            df1 = pd.DataFrame(df1, columns=keys)
            df2 = pd.DataFrame(df2, columns=keys)

    #> calling wass
    return wass(df1, df2, **kwargs)


""" #> POP COMP ======================
================================== """

#> compares populations by drawing subsets from many files
def popComp(loc, obsIndx=0, numObs=100, numMock=1000, obsPop=None, 
            numGals=100, numSource_gal=500, **kwargs):
    
    #> unpacking kwargs
    seed   = kwargs.get('seed', None)
    observables = kwargs.get('observables', ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23', 'd01'])
    
    #> cutting outliers
    cut_key = kwargs.get('cut_key', 'dt23')
    cut_max = kwargs.get('cut_max', 20)
    cut_indx = observables.index(cut_key)

    #> declarations
    columns = ['file1', 'file2', '1DSW_Xi', 'NDSW_Xi', '1DSW_Theta', 'NDSW_Theta']
    # columns = ['file1', 'file2', '1DSW_Xi', '1DSW_Theta']
    df = pd.DataFrame({}, columns=columns)
    np.random.seed(seed)
    
    #> gets all files
    files = sorted(Path(loc).rglob('*bprofiles_df.csv')) # finding bprofiles files
    
    
    #>>># OBSERVED DATA #<<<#
    
    #> drawing observed data if none is supplied
    #> cuts bad dt23, draws 1 quad per gal, ensures no repeats in self comp
    if obsPop is None:
        
        #> checking to ensure consistency
        if numObs != numGals:
            error.phrase(f'numObs ({numObs}) must equal numGals ({numGals})')
        
        #> loading obsFile and data
        obsFile = str(files[obsIndx]).replace('bprofiles_df.csv', 'obs.npy')
        obsData = parse.load_bprofiles(fileName=obsFile, input='npy')
        
        #> iteratres through each gal
        obs_ranIndx_gal = []
        for j in range(numGals):
            
            #> getting galaxy data
            galData = obsData[j*numSource_gal:(j+1)*numSource_gal]
            sourceIndx = np.where(np.abs(galData[:,cut_indx]) < cut_max)[0]
            
            #> checking availability
            if len(sourceIndx) == 0:  error.phrase(f'Galaxy {j} has no sources with {cut_key} < {cut_max}')
            
            #> selecting one source
            obs_ranIndx_gal.append(np.random.choice(sourceIndx))
            
        #> getting full indx and selecting
        obs_ranIndx_gal = np.array(obs_ranIndx_gal, dtype=int)
        obs_ranIndx = np.arange(numGals)*numSource_gal + obs_ranIndx_gal
        obs_df = pd.DataFrame(obsData[obs_ranIndx], columns=observables)
    
    else: # if obsPop is given
        
        if isinstance(obsPop, pd.DataFrame): obs_df = obsPop.copy()
        elif isinstance(obsPop, np.ndarray): obs_df = pd.DataFrame(obsPop, columns=observables)
        
        #> applying cut
        obs_df = obs_df[np.abs(obs_df[cut_key]) < cut_max].reset_index(drop=True)
        
        
    #>>># COMPARISONS & MOCK DATA #<<<#
    
    #> iterates through files
    for i, file in enumerate(files):
        
        #> declarations
        dummy = [obsFile, str(file)]
        
        #> comparing bprofiles
        dummy.append(wassXi(files[obsIndx], files[i], input='csv', method='1D', **kwargs))
        dummy.append(wassXi(files[obsIndx], files[i], input='csv', method='ND', **kwargs))
        
        #> loading mock file & converting to df
        file2 = str(files[i]).replace('bprofiles_df.csv', 'obs.npy')
        data2 = parse.load_bprofiles(fileName=file2, input='npy')
        
        #> declarations
        if numMock % numGals != 0: error.phrase(f'numMock {numMock} is not divisible by numGals ({numGals})')
        numMock_gal = int(numMock / numGals)
        
        #> selecting random mock sample
        ranIndx = []
        
        #> iterating through each galaxy
        for j in range(numGals):
            
            #> getting galaxy data
            galData = data2[j*numSource_gal:(j+1)*numSource_gal]
            sourceIndx = np.where(np.abs(galData[:,cut_indx]) < cut_max)[0]
            
            #> removing observed source for observed population
            if obsPop is None and i == obsIndx:
                sourceIndx = sourceIndx[sourceIndx != obs_ranIndx_gal[j]]
            
            #> checking availability
            if len(sourceIndx) < numMock_gal:
                error.phrase(f'Galaxy {j} only has {len(sourceIndx)} sources with {cut_key} < {cut_max}')
            
            #> drawing sources
            ranIndx_gal = np.random.choice(sourceIndx,
                                           size=numMock_gal,
                                           replace=False)
            
            #> converting to full array indices
            ranIndx.extend(j*numSource_gal + ranIndx_gal)
        ranIndx = np.array(ranIndx, dtype=int)
        mock_df = pd.DataFrame(data2[ranIndx], columns=observables)
        
        #> comparing lensing observables
        dummy.append(wassTheta(obs_df, mock_df, keys=observables, input='df', method='1D', **kwargs))
        dummy.append(wassTheta(obs_df, mock_df, keys=observables, input='df', method='ND', **kwargs))
        
        #> saving to df
        df.loc[i] = np.array(dummy)
    
    return df, obsIndx


#> plots the metrics against each other
def plotComp(df, obsIndx):
    
    #> imports
    import matplotlib.pyplot as plt
    
    #> removing file name keys
    keys = df.keys()[2:]
    df = df[keys].astype(float)
    
    #> normalizing each metric from 0 to 1
    data = df.to_numpy()
    dmin, drange = normScale(data=data)
    df = pd.DataFrame(normData(data, dmin, drange), columns=keys)
    
    #> getting metric keys
    Xi_keys    = keys[['Xi'    in key for key in keys]]
    Theta_keys = keys[['Theta' in key for key in keys]]
    
    #> making all key combos
    key_combos = np.array(np.meshgrid(Xi_keys, Theta_keys)).T.reshape(-1, 2)
    
    #> initializing plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.grid(ls=':', alpha=0.5)
    ax.set_xlabel(r'$\mathbf{\Xi}$', fontsize=15)
    ax.set_ylabel(r'$\mathbf{\Theta}$', fontsize=15)
    
    print(df)
        
    #> plotting!
    for pair in key_combos:
        key1, key2 = pair
        # if 'ND' in key1: continue
        print(key1, key2)
        ax.scatter(df[key1], df[key2], label=f'{key1[:2]}-{key2[:2]}')
        # ax.scatter(df[key1].loc[obsIndx], df[key2].loc[obsIndx], c='k', marker='*')
    
    plt.legend()
    plt.show(); plt.close()
    
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    #> declarations
    seed = None
    
    #> getting all populations
    loc = '../opus_lmfi/data/260909_comp1/'
    df, obsIndx = popComp(loc, seed=seed, slices=100, numGals=100, numSource_gal=500,
                          numObs=100, numMock=1000)
    plotComp(df, obsIndx)

    
    # end
# thank