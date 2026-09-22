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

#> wasserstein distance for lensing observables
def wassTheta(fileName1, fileName2, keys=None, method='ND', slices=100):
    
    #> collecting data
    data1 = np.load(fileName1)[:100]
    data2 = np.load(fileName2)[:100]

    # Calculate the distance
    sw_distance = swd(data1, data2, n_projections=slices)
    
    print(f"> Sliced Wasserstein Distance: {sw_distance}")
    
    return sw_distance


#> wasserstein distance for galaxy properties
#> two methods: 1D = compares all dists in 1D, then adds Wass
#               ND = sliced wassersten metric
#> input methods: csv = bprofiles.csv
#                 npy = bprofiles.npy
#                 bprofiles = bprofiles array
#                 df = dataframe
#                 np = numpy array
def wassXi(fileName1, fileName2, input='csv', keys=None, method='ND', slices=100):
    
    #> declarations
    if keys == None:
        
        #> default keys
        keys = ['zl', 'zs',                                                     # redshifts
                'nfw_logmass', 'nfw_concentration', 'nfw_axisrat', 'nfw_theta', # nfw params
                'hern_logmass', 'hern_effrad', 'hern_axisrat',                  # hern params
                'mult1_norm', 'mult1_theta', 'mult3_norm', 'mult4_norm',        # multipole params
                'ex_norm', 'ex_theta']                                          # ex params
    
    #> loading data
    if input not in ['df', 'np']:
        df1 = parse.load_bprofiles(fileName=fileName1, input=input)
        df2 = parse.load_bprofiles(fileName=fileName2, input=input)
    else:
        if input == 'df': df1, df2 = fileName1, fileName2 # if already dataframes
        if input == 'np': df1, df2 = pd.DataFrame(fileName1, columns=keys), pd.DataFrame(fileName2, columns=keys)
    
    #> separates based on methods
    #> marginalized distributions (1D)
    if method == '1D':
        
        #> iterates through keys
        sw_dists = []
        for key in keys: 
            sw_dists.append( swd(df1[key], df2[key], n_projections=1) )
        sw_dists = np.arrays(sw_dists)
        
    #> joint distributions (ND)
    elif method == 'ND':
        
        
        

    #> calculates the SW distance
    # sw_distance = swd(data1, data2, n_projections=slices)
    
    # print(f"> Sliced Wasserstein Distance: {sw_distance}")
    
    return sw_distance


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    #> getting all populations
    loc = '../opus_lmfi/data/260909_comp1/'
    pattern = '*.csv'
    
    #> iterates through files
    files = list(Path(loc).rglob(pattern))
    for i, file in enumerate(files):
        sw = wassXi(files[0], files[i], input='csv')
    
    
    # end
# thank