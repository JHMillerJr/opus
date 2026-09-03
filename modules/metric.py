#> name: metric.py
#> author: John Miller Jr
#> descrp: calculates various cosmological params for lensing

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import scipy as sp

import glob

import ot

#> modules
from modules.units import u; u=u()
import modules.error as error

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


#> wasserstein distance
def wass(inFile1, inFile2, observables):
    
    #> getting parameter ranges
    ranges = paramRanges(observables)
    numParams = len(ranges)
    
    #> collecting data
    data1 = np.load(inFile1)[:100]
    data2 = np.load(inFile2)[:100]
    
    print(data1.shape)
    
    #> 
    n_projections = 1000 # Number of random projections (higher is generally better for accuracy)

    # Calculate the distance
    sw_distance = ot.sliced_wasserstein_distance(data1,          # Source samples
                                                 data2,          # Target samples
                                                 n_projections=n_projections)
    
    print(f"Sliced Wasserstein Distance: {sw_distance}")
    
    return sw_distance


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    #> getting all files in data
    files = glob.glob('./data/*.npy')
    files = [ x.replace("\\", "/") for x in files ]
    
    #> plotting quad population
    observables = ['t23', 'dt23', 'd4/d1']
    file1 = dataDir+'2602251720-obs'+'.npy'
    # file2 = dataDir+'2602251720-obs'+'.npy'
    file2 = dataDir+'2602251721-mod1'+'.npy'
    file2 = dataDir+'2602251721-mod2'+'.npy'
    file2 = dataDir+'2602251722-mod3'+'.npy'
    file2 = dataDir+'2602251723-mod4'+'.npy'
    # file2 = dataDir+'2602231352-mod5'+'.npy'
    
    observables = ['t23', 'dt23', 'd4/d1']
    for i, file in enumerate(files):
        p_val = gridComp(files[1], file, bins=10, observables=observables)
        wass(files[1], file, observables)
        print(file, p_val)
        
    #### IN THE OUTPUT OF THE NPY FILES, INCLUDE THE HEADERS OF THE OBSERVABLES
    #### SO THAT THEY CAN BE SPECIFICALLY REQUESTED, PLUS THEN I KNOW WHAT
    #### OBSERVABLES I USED! :)
    #### OH AND MAKE AN OPTION TO SAVE THE PARAM DICTIONARY TO A SEPARATE FILE
    #### OR MAYBE THAT IS WHAT CONTAINS THE OBSERVABLES...
    
    
    # end
# thank