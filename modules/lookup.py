#> name: lookup.py
#> author: John Miller Jr
#> descrp: scrapes look up tables for quick prior draws

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import numpy as np
import pandas as pd

#> caching
from functools import lru_cache

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__)) 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> READING =======================
================================== """

#> loading zl look-up table
@lru_cache(maxsize=None)
def load_model_table(izl, **kwargs):
    
    #> kwargs
    returnFile = kwargs.get('returnFile', False)
    
    #> declarations
    lookUpBins = look_up_bins()
    zl = lookUpBins['zl']['grid'][int(izl)]
    loc = './priors/look_up_tables/'
    fileName = loc+f'model_zl_{zl:.2f}.parquet'
    
    #> if just wanting the file name
    if returnFile: return fileName

    #> loading table
    return pd.read_parquet(fileName)

#> loading a model from the look-up table
@lru_cache(maxsize=None)
def load_model(izl, izs, ilogmvir, ilogmstar, ire):
    
    #> converting indices
    izl, izs = int(izl), int(izs)
    ilogmvir, ilogmstar, ire = int(ilogmvir), int(ilogmstar), int(ire)
    
    #> loading
    lookUpBins = look_up_bins()
    df = load_model_table(izl)
    
    #> finding model
    i = np.ravel_multi_index((izs,
                              ilogmvir,
                              ilogmstar,
                              ire),
                             (lookUpBins['zs']['bins'],
                              lookUpBins['logmvir']['bins'],
                              lookUpBins['mstar_nsig']['bins'],
                              lookUpBins['re_nsig']['bins']))
    
    return df.iloc[i]

#> returns closest look-up bin indices
def bin_indices(x, grid):
    x, grid = np.asarray(x), np.asarray(grid)
    return np.abs(x[:,None] - grid[None,:]).argmin(axis=1)



""" #> EDITING =======================
================================== """

#> makes edits to all zl look-up tables
def edit_model_table(**kwargs):
    
    #> kwargs
    max_er = kwargs.get('max_er', 2.4)
    
    #> loading bins
    lookUpBins = look_up_bins()
    
    #> loading each file
    for izl, zl in enumerate(lookUpBins['zl']['grid']):
        
        #> loading model
        model = load_model_table(izl)
        fileName = load_model_table(izl, returnFile=True)
        
        #> iterating through each zs
        for izs in range(lookUpBins['zs']['bins']):
            
            #> filtering
            mask = (model['izs'] == izs) & (model['ER'] <=  max_er)
            d_area_max = model.loc[mask,'d_area'].max()
            
            #> applying to all at izs
            mask = (model['izs'] == izs)
            model.loc[mask,'d_area_max'] = d_area_max
            
        #> probability of a quad
        model['p_quad'] = (model['d_area'] / model['d_area_max']).clip(0,1).fillna(0)
        
        #> saving files
        print(fileName)
        model.to_parquet(fileName)
    
    return


""" #> LOOK-UP TABLES ================
================================== """

#> returns look up bins
def look_up_bins(**kwargs):
    
    #>  kwargs
    scale_factor = kwargs.get('scale_factor', 1)
    
    #> dictionary
    lookUpBins = {'zl':         {'min': 0.01, 'max':  1.8, 'bins': 20},
                  'zs':         {'min':  0.1, 'max':  5.0, 'bins': 20},
                  'logmvir':    {'min': 11.5, 'max': 14.5, 'bins': 20},
                  'mstar_nsig': {'min':   -3, 'max':    3, 'bins': 7},
                  're_nsig':    {'min':   -3, 'max':    3, 'bins': 7},
                  'pix_arc':    {'min':  np.log10(6*scale_factor),  
                                 'max':  np.log10(110*scale_factor), 
                                 'bins': 5}}
    
    #> setting up bins
    for key in lookUpBins.keys():
        
        #> edges of the bins
        if key in ['mstar_nsig', 're_nsig', 'pix_arc']: add = 0
        else: add = 1
        grid_edges  = np.linspace(lookUpBins[key]['min'], 
                                  lookUpBins[key]['max'], 
                                  lookUpBins[key]['bins'] + add)
        
        #> centers of the bins
        grid_bins   = (grid_edges[:-1] + grid_edges[1:]) / 2
        
        #> saving depending on what grid
        if key in ['mstar_nsig', 're_nsig', 'pix_arc']:
            if key in ['pix_arc']: grid_edges = np.array([10**x for x in grid_edges])
            lookUpBins[key]['grid'] = np.array([round(x,6) for x in grid_edges])
        else:
            lookUpBins[key]['grid'] = np.array([round(x,6) for x in grid_bins])
    
    return lookUpBins

#> maximum quad cross section at given redshifts
@lru_cache(maxsize=None)
def max_darea(izl, izs,**kwargs):
    df = load_model_table(int(izl))    # loads zl table
    mask = (df['izs'] == int(izs) ) & (df['ER'] <= kwargs.get('max_er', 2.4)) # mask @ zs
    return df.loc[mask,'d_area'].max() # finds maximum d_area


""" #> FIRST LOOK SELEC FN ===========
================================== """

#> applys the 'first look' selection functions
def firstLook_selFunc(df, **kwargs):
    
    #> kwargs
    resolution = kwargs.get('resolution', 0.15) # min threshold [arcsec]
    # max_er     = kwargs.get('max_er', 2.4)      # max ER for d_area normalization [arcsec]
    
    #> loading prior bins
    lookUpBins = look_up_bins()
    
    #> translation dictionary (lookUpBins key to df key)
    transDict = {'zl':         'zl',
                 'zs':         'zs',
                 'logmvir':    'log10_M_vir',
                 'mstar_nsig': 'mstar_nsig',
                 're_nsig':    're_nsig'}
    
    #> getting look-up indices
    for key in lookUpBins.keys():
        if key in ['pix_arc']: continue
        df[f'i{key}'] = bin_indices(x = df[transDict[key]].to_numpy(),# sampled values
                                    grid = lookUpBins[key]['grid'])   # prior bins
    
    #> loading look up tables
    ERs, dareas, max_dareas, p_quads = [], [], [], []
    for i in range(len(df)):
        
        #> getting galaxy
        g = df.iloc[i]
        
        #> loading model
        model = load_model(g['izl'], 
                           g['izs'], 
                           g['ilogmvir'], 
                           g['imstar_nsig'], 
                           g['ire_nsig'])
        
        ERs.append(model['ER'])
        dareas.append(model['d_area'])
        max_dareas.append(model['d_area_max'])
        p_quads.append(model['p_quad'])
    
    #> adding to df
    df['ER'] = ERs
    df['d_area'] = dareas
    df['d_area_max'] = max_dareas
    df['p_quad'] = p_quads
    
    #> quad probability
    # p_quad = (df['d_area'] / df['d_area_max']).clip(0,1).fillna(0)
    
    #> applying selection functions
    df['sel'] = ((df['ER'] >= resolution) &
                 (np.random.random(len(df)) < df['p_quad']))
    
    return df


""" #> PLOTTING ======================
================================== """

#> plotting various lookup table things
def plot_lookup(df):
    
    #> imports
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    
    #> color hist
    if False:
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.grid(ls=':', alpha=0.5)
        
        #> declarations
        p_param = 'p_quad'
        
        #> colormap
        c_param = 'ER'
        cmap = plt.get_cmap('RdYlBu')
        norm = plt.Normalize(vmin=df[c_param].min(), vmax=df[c_param].max())
        
        #> initializing bins
        num_bins = 10
        bins = np.linspace(df[p_param].min(), df[p_param].max() + 0.001, num_bins + 1)
        df['bin'] = np.digitize(df[p_param], bins)
        
        #> plotting
        for bin_id, bin_df in df.groupby('bin'):
            ax.imshow(bin_df[c_param].values.reshape(-1, 1), interpolation='nearest', cmap=cmap, norm=norm,
                      extent=[bins[bin_id - 1], bins[bin_id], 0, len(bin_df)], aspect='auto')
        
        #> extra stuff
        ax.set_xlabel(p_param)
        ax.use_sticky_edges = False # remove stickiness due to imshow
        ax.autoscale_view()
        ax.set_ylim(ymin=0)
        
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), label=c_param, ax=ax)
        plt.tight_layout()
        plt.show()
        
    #> two params against each other
    if True:
        
        #> declarations
        x_param = 'd_area_max'
        y_param = 'd_area'
        
        print(df[df[y_param]==df[y_param].min()].to_string())
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.grid(ls=':', alpha=0.5)
        ax.set_xlabel(x_param, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(y_param, fontsize=fontsize, labelpad=labelpad)
        
        #> plotting
        ax.scatter(np.log10(df[x_param]), np.log10(df[y_param]), alpha=0.6)
        
        #> line
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 1000)
        ax.plot(np.log10(x), np.log10(x), ls=':', c='k', alpha=0.4)
    
        plt.show()
        
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    #> loading models
    zl = 0.55
    izs = 1
    ilogmvir = 1
    ilogmstar = 1
    ire = 1
    data = load_model(zl, izs, ilogmvir, ilogmstar, ire)
    print(data)
    
    # end
# thank