#> name: configuration.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize

#> special imports
from shapely import geometry as geo
from matplotlib.patches import Arc

#> modules
import lensing

#> keys
global keys
keys = {'t12': r'$\mathbf{\theta_{12}}$', 't23': r'$\mathbf{\theta_{23}}$', 't34': r'$\mathbf{\theta_{34}}$',
       'd2/d1': r'$\mathbf{d_2/d_1}$', 'd3/d1': r'$\mathbf{d_3/d_1}$', 'd4/d1': r'$\mathbf{d_4/d_1}$',
       'dt23': r'$\Delta\theta_{23}$', 
       'dt1': r'$\mathbf{\Delta\theta_{1}=\theta_{12}-\theta_{34}}$', 
       'dt2_t12': r'$\mathbf{\Delta\theta_{2}=\theta_{12}-2\theta_{23}}$',
       'dt2_t34': r'$\mathbf{\Delta\theta_{2}=\theta_{34}-2\theta_{23}}$'}



""" #> PLOTTING ======================
================================== """

#> plots an arc (for angles)
def plot_arc(ax, t1, t2, r, angle, label, n=100):
    t = np.linspace(t1, t2, n)
    ax.plot(r*np.cos(t), r*np.sin(t), lw=4, alpha=1.0, label=label+f'={angle:.1f}'+r'$\degree$')


#> plotting SINGLE quad from angles
def angleQuad(angles, **kwargs):
    
    #> kwargs
    config = kwargs.get('config', None)
    name = kwargs.get('name', None)
    
    #> redefining the angles for printing
    angles = np.asarray(angles)
    angles = np.atleast_2d(angles)
    
    t12 = float(angles[:, 0][0])
    t23 = float(angles[:, 1][0])
    t34 = float(angles[:, 2][0])
    
    t23 = t12 - t23
    t34 = t23 + t34
    
    angles = np.column_stack([t12, t23, t34])

    #> translating from angles to x,y (r=1)
    rad_angles = np.deg2rad(angles)
    x = np.cos(rad_angles)
    y = np.sin(rad_angles)
    x = np.hstack([np.ones((angles.shape[0], 1)), x])
    y = np.hstack([np.zeros((angles.shape[0], 1)), y])
    
    #> initializing plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.grid(ls=':', alpha=0.5)
    if name is not None:
        if config is not None: ax.set_title(f'{name}: {config}', fontweight='bold', fontsize=15)
        else: ax.set_title(f'{name}', fontweight='bold', fontsize=15)
    else:
        if config is not None: ax.set_title(f'Quad from Angles: {config}', fontweight='bold', fontsize=15)
        else: ax.set_title('Quad from Angles', fontweight='bold', fontsize=15)
    
    #> plotting unit cirlce (reference)
    circle = plt.Circle((0, 0), 1.0, fill=False, ls=':', alpha=0.5)
    ax.add_patch(circle)
    
    #> plotting!
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xi = x[i, j]
            yi = y[i, j]
            ax.plot((0, xi), (0, yi), c='k', ls=':', lw=2, alpha=0.6)
            ax.scatter(xi, yi, marker='.', s=2000, c='w')
            ax.scatter(xi, yi, marker=f'${j+1}$', s=150)
    ax.scatter(0, 0, marker='+', c='k')
    
    #> arcs
    for i in range(angles.shape[0]):
        plot_arc(ax, 0.0, rad_angles[i, 0], 0.4, t12, r'$\theta_{12}$')
        plot_arc(ax, rad_angles[i, 0], rad_angles[i, 1], 0.3, t12-t23, r'$\theta_{23}$')
        plot_arc(ax, rad_angles[i, 1], rad_angles[i, 2], 0.2, t34-t23, r'$\theta_{34}$')
    
    #> axis
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    #> legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.savefig('./figures/'+name+'_angles.png', bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    return


#> plotting multiple quads at once (no angles)
def angleQuads(df, sort_value, **kwargs):

    #> other info/ fields
    lens = kwargs.get('lens', None)
    names = kwargs.get('names', None)
    
    #> sorting
    df = df.sort_values(sort_value)
    sort_values = df[sort_value].to_numpy(dtype=np.float64)
    cmap = LinearSegmentedColormap.from_list("my_map", ["blue", "red"])
    norm = Normalize(vmin=sort_values.min(), vmax=sort_values.max())
    sources = df[['xs', 'ys']].to_numpy()
    angles= df.to_numpy(dtype=np.float64)
    
    #> redefining the angles for printing
    angles = np.asarray(angles)
    angles = np.atleast_2d(angles)
    
    t12 = angles[:, 0]
    t23 = angles[:, 1]
    t34 = angles[:, 2]
    
    t23 = t12 - t23
    t34 = t23 + t34
    
    angles = np.column_stack([t12, t23, t34])
        
    #> translating from angles to x,y (r=1)
    rad_angles = np.deg2rad(angles)
    r = np.array([0.9, 0.8, 0.7])  # per-column radii
    x = r * np.cos(rad_angles)
    y = r * np.sin(rad_angles)

    # prepend origin (for each row)
    x = np.hstack([np.ones((angles.shape[0], 1)), x])
    y = np.hstack([np.zeros((angles.shape[0], 1)), y])
    
    #> initializing plot
    numFigs = 1
    if sources is not None: numFigs += 1
    fig, axes = plt.subplots(1,numFigs,figsize=(7.3*numFigs+(0.3*(numFigs-1)),6))
    if numFigs == 1: axes = np.array([axes])
    
    #> colors
    N, M = x.shape
    x_flat, y_flat = x.ravel(), y.ravel()
    cmap = LinearSegmentedColormap.from_list("my_map", ["blue", "red"])
    norm = Normalize(vmin=sort_values.min(), vmax=sort_values.max())

    #> titles
    titles = ['Image Positions from Angles', 'Sources']
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
    
        #> initializing
        ax.grid(ls=':', alpha=0.5)
        ax.set_title(titles[i], fontweight='bold', fontsize=15)
        
        #> lens plane
        if i == 0:
            
            #> plotting unit cirlce (reference)
            circle1 = plt.Circle((0, 0), 1.0, fill=False, ls=':', alpha=0.0)
            circle2 = plt.Circle((0, 0), 0.85, fill=False, ls=':', alpha=0.5)
            ax.add_patch(circle1); ax.add_patch(circle2)
            
            #> plotting!
            cvals = np.repeat(sort_values, M)
            q = ax.scatter(x_flat, y_flat, c=cvals, cmap=cmap, norm=norm, s=50, marker='.', alpha=1.0)
            ax.scatter(0, 0, marker='+', c='k')
            
            #> plotting end cases
            for i in [0, x.shape[0]-1]:
                for j in range(x.shape[1]):
                    xi = x[i, j]
                    yi = y[i, j]
                    ax.plot((0, xi), (0, yi), lw=2, c=cmap(norm(i)), ls=':')
                    
        #> source plane
        if i == 1:
            
            #> unpacking data
            gridx, gridy, delx, dely, gamma = lens
            
            #> getting caustics
            dcaustic, ocaustic = lensing.caustics(gridx, gridy, 
                                                  delx, dely, 
                                                  pix_arc=60)
            
            #> plotting diamond
            for line in dcaustic: ax.plot(*line.xy, c='k', ls=':')
            # for line in ocaustic: ax.plot(*line.xy, c='r')
            
            #> plotting sources
            source_cvals = sort_values[:len(sources)]  # or a proper mapping if needed
            ax.scatter(sources[:,0], sources[:,1], c=source_cvals, cmap=cmap, norm=norm, s=20, alpha=1.0)
            
            #> setting limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
        
        #> axis
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        
    cbar = plt.colorbar(q, ax=axes)
    cbar.set_label(sort_value, fontsize=15, fontweight='bold', labelpad=20)
        
    #> legend
    outFile = kwargs.get('outFile', None)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    return


#> plotting plane of configuratoins
def configPlane(angles, **kwargs):
    
    #> unpacking kwargs
    comparison = kwargs.get('comparison', None)
    title = kwargs.get('title', None)
    
    #> declarations
    sort_value = 't34'
    x_value = 'dt1'
    y_value = 'dt3'
    x = angles[x_value]
    y = angles[y_value]
    sort_values = angles[sort_value]
    
    #> plotting delcarations
    fontsize = 15
    
    #> colors
    cmap = LinearSegmentedColormap.from_list("my_map", ["blue", "red"])
    norm = Normalize(vmin=sort_values.min(), vmax=sort_values.max())
    
    #> initializing plots
    fig, ax = plt.subplots(1,1,figsize=(7.3,6))
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight='bold')
    else:
        ax.set_title('All Quads', fontsize=fontsize, fontweight='bold')
    ax.set_xlabel(keys[x_value], fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(keys[y_value], fontsize=fontsize, fontweight='bold')
    ax.grid(ls=':', alpha=0.5)
    
    #> plotting
    if comparison is not None: 
        ax.scatter(comparison[x_value], comparison[y_value], c='k', s=10, marker='.', alpha=0.1)
    q = ax.scatter(x, y, c=sort_values, cmap=cmap, norm=norm, s=10, marker='.', alpha=1.0)
    
    #> ticks
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    
    #> setting limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low, high = min(xmin, ymin), max(xmax, ymax)
    ax.set_xlim(low, high)
    ax.set_ylim(-5, high)
    
    #> color bar
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label(keys[sort_value], fontsize=fontsize, fontweight='bold', labelpad=20)
        
    #> showing and closing
    outFile = kwargs.get('outFile', None)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    return


#> tri-panel figure of image positions, sources, and angle space
def triPanel(df, sort_value, lens, **kwargs):
    
    #> other info/ fields
    comp_df = kwargs.get('comp_df', None)
    
    #> sorting
    df = df.sort_values(sort_value)
    sort_values = df[sort_value].to_numpy(dtype=np.float64)
    cmap = LinearSegmentedColormap.from_list("my_map", ["blue", "red"])
    norm = Normalize(vmin=sort_values.min(), vmax=sort_values.max())
    sources = df[['xs', 'ys']].to_numpy()
    angles = df[['t12', 't23', 't34']].to_numpy(dtype=np.float64)
    
    #> redefining the angles for printing
    angles = np.asarray(angles)
    angles = np.atleast_2d(angles)
    
    t12 = angles[:, 0]
    t23 = angles[:, 1]
    t34 = angles[:, 2]
    
    t23 = t12 - t23
    t34 = t23 + t34
    
    modified_angles = np.column_stack([t12, t23, t34])
        
    #> translating from angles to x,y (r=1)
    rad_angles = np.deg2rad(modified_angles)
    r = np.array([0.9, 0.8, 0.7])  # per-column radii
    x = r * np.cos(rad_angles)
    y = r * np.sin(rad_angles)

    # prepend origin (for each row)
    x = np.hstack([np.ones((modified_angles.shape[0], 1)), x])
    y = np.hstack([np.zeros((modified_angles.shape[0], 1)), y])
    
    #> initializing plot
    numFigs = 3
    fig, axes = plt.subplots(1,numFigs,figsize=(8.15*numFigs,6))
    
    #> plot declarations
    fontsize = 15
    
    #> colors
    N, M = x.shape
    x_flat, y_flat = x.ravel(), y.ravel()
    cmap = LinearSegmentedColormap.from_list("my_map", ["blue", "red"])
    norm = Normalize(vmin=sort_values.min(), vmax=sort_values.max())

    #> titles
    titles = ['Image Positions from Angles', 'Sources', 'Angle Space']
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
    
        #> initializing
        ax.grid(ls=':', alpha=0.5)
        ax.set_title(titles[i], fontweight='bold', fontsize=fontsize)
        
        #> lens plane
        if i == 0:
            
            #> labels
            ax.set_xlabel(r'$\mathbf{\Delta RA ~~[]}$', fontweight='bold', fontsize=fontsize)
            ax.set_ylabel(r'$\mathbf{\Delta DEC ~~[]}$', fontweight='bold', fontsize=fontsize)
            
            #> plotting unit cirlce (reference)
            circle1 = plt.Circle((0, 0), 1.0, fill=False, ls=':', alpha=0.0)
            circle2 = plt.Circle((0, 0), 0.85, fill=False, ls=':', alpha=0.5)
            ax.add_patch(circle1); ax.add_patch(circle2)
            
            #> plotting!
            cvals = np.repeat(sort_values, M)
            q = ax.scatter(x_flat, y_flat, c=cvals, cmap=cmap, norm=norm, s=50, marker='.', alpha=1.0)
            ax.scatter(0, 0, marker='+', c='k')
            
            #> plotting end cases
            for i in [0, x.shape[0]-1]:
                for j in range(x.shape[1]):
                    xi = x[i, j]
                    yi = y[i, j]
                    ax.plot((0, xi), (0, yi), lw=2, c=cmap(norm(i)), ls=':')
                    
        #> source plane
        if i == 1:
            
            #> labels
            ax.set_xlabel(r'$\mathbf{\Delta RA ~~[arcsec]}$', fontweight='bold', fontsize=fontsize)
            ax.set_ylabel(r'$\mathbf{\Delta DEC ~~[arcsec]}$', fontweight='bold', fontsize=fontsize)
            
            #> unpacking data
            gridx, gridy, delx, dely, gamma = lens
            
            #> getting caustics
            dcaustic, ocaustic = lensing.caustics(gridx, gridy, 
                                                  delx, dely, 
                                                  pix_arc=60)
            
            #> plotting diamond
            for line in dcaustic: ax.plot(*line.xy, c='k', ls=':')
            # for line in ocaustic: ax.plot(*line.xy, c='r')
            
            #> plotting sources
            if comp_df is not None: 
                ax.scatter(comp_df['xs'], comp_df['ys'], c='k', s=10, marker='.', alpha=0.01)
            source_cvals = sort_values[:len(sources)]  # or a proper mapping if needed
            ax.scatter(sources[:,0], sources[:,1], c=source_cvals, cmap=cmap, norm=norm, s=20, alpha=1.0)
            
            #> setting limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            
            
        #> angle plane
        if i == 2:
            
            #> declarations
            x_value = 'dt1'
            y_value = 'dt2_t12'
            x = df[x_value]
            y = df[y_value]
            sort_values = df[sort_value]
            
            ax.set_xlabel(keys[x_value] + r'$~~\mathbf{[\degree]}$', fontsize=fontsize, fontweight='bold')
            ax.set_ylabel(keys[y_value] + r'$~~\mathbf{[\degree]}$', fontsize=fontsize, fontweight='bold')
            
            #> plotting
            if comp_df is not None: 
                ax.scatter(comp_df[x_value], comp_df[y_value], c='k', s=10, marker='.', alpha=0.03)
            q = ax.scatter(x, y, c=sort_values, cmap=cmap, norm=norm, s=10, marker='.', alpha=1.0)

            #> ticks
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            
            #> setting limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(low, high)
            # ax.set_ylim(-5, high)
        
        #> axis
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        
    cbar = plt.colorbar(q, ax=axes)
    cbar.set_label(keys[sort_value], fontsize=15, fontweight='bold', labelpad=20)
        
    #> legend
    outFile = kwargs.get('outFile', None)
    if outFile is not None:
        plt.savefig(outFile, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()
    
    return

#> huge 9x9 plot that contains all of the configs
def plotAllAngles(df, **kwargs):
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    #> declarations
    global keys
    fontsize = kwargs.get('fontsize', 15)
    labelpad = kwargs.get('labelpad', 10)
    bins = kwargs.get('bins', 15)
    alpha = kwargs.get('alpha', 0.02)
    s = kwargs.get('s', 10)
    linewidth = kwargs.get('linewidth', 4)
    
    #> make sure new mask columns exist
    if ('algo_list' not in df.columns) or ('algo' not in df.columns):
        df = applyMasks(df)
    
    class_cols = ['EC', 'F', 'CM', 'Cm', 'A']
    
    colors = {'EC': 'tab:blue',
              'F': 'tab:orange',
              'CM': 'tab:green',
              'Cm': 'tab:red',
              'A': 'tab:purple'}
    
    labels = {'EC': 'Cross',
              'F': 'Fold',
              'CM': 'Major cusp',
              'Cm': 'Minor cusp',
              'A': 'Unknown'}
    
    #> initializing plot
    fig, axes = plt.subplots(3, 3, figsize=(6*3, 6*3))
    
    #> plotting
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        if i in [0, 1, 2]:
            ax.set_yticklabels([])
            
        for class_col in class_cols:
            
            #> separates data using new applyMasks output
            # class_type = class_to_type[class_col]
            
            if class_col == 'A':
                sub = df[df['algo'] == 'A']
            else:
                sub = df[df['algo_list'].apply(lambda x: class_col in x)]
            
            if len(sub) == 0:
                continue
            
            color = colors[class_col]
            label = labels[class_col]
            
            if i == 0:
                ax.hist(sub['t12'], bins=bins,
                        histtype='step', linewidth=linewidth,
                        color=color, label=label)
                ax.set_xlabel(keys['t12'], fontsize=fontsize, labelpad=labelpad)
            
            elif i == 1:
                ax.hist(sub['t23'], bins=bins,
                        histtype='step', linewidth=linewidth,
                        color=color, label=label)
                ax.set_xlabel(keys['t23'], fontsize=fontsize, labelpad=labelpad)
                
            elif i == 2:
                ax.hist(sub['t34'], bins=bins,
                        histtype='step', linewidth=linewidth,
                        color=color, label=label)
                ax.set_xlabel(keys['t34'], fontsize=fontsize, labelpad=labelpad)
                
            elif i == 3:
                ax.scatter(sub['t23'], sub['t12'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['t23'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['t12'], fontsize=fontsize)
                
            elif i == 4:
                ax.scatter(sub['t34'], sub['t12'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['t34'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['t12'], fontsize=fontsize)
                
            elif i == 5:
                ax.scatter(sub['t23'], sub['t34'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['t23'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['t34'], fontsize=fontsize)
                
            elif i == 6:
                ax.scatter(sub['dt1'], sub['dt2_t12'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['dt1'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['dt2_t12'], fontsize=fontsize)
                
            elif i == 7:
                ax.scatter(sub['dt1'], sub['dt2_t34'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['dt1'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['dt2_t34'], fontsize=fontsize)
                
            elif i == 8:
                # ax.axis('off')
                ax.scatter(sub['xs'], sub['ys'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(r'$\mathbf{\Delta RA ~~[arcsec]}$', fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(r'$\mathbf{\Delta DEC ~~[arcsec]}$', fontsize=fontsize)

        if i == 8:
            
            #> setting limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
    
    #> turnning off, to put legend
    # axes.flat[8].axis('off')
    
    #> shared legend below entire figure
    handles, legend_labels = axes.flat[3].get_legend_handles_labels()
    
    legend = fig.legend(handles, legend_labels,
                        loc='lower center', bbox_to_anchor=(0.5, -0.05),
                        ncol=len(legend_labels), fontsize=20,
                        frameon=False, markerscale=10,
                        columnspacing=1.5, handletextpad=0.4)
    
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return


#> plots the bipanel (sources and angle space)
def biPanel(df, **kwargs):
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    #> declarations
    global keys
    fontsize = kwargs.get('fontsize', 15)
    labelpad = kwargs.get('labelpad', 10)
    alpha = kwargs.get('alpha', 0.02)
    s = kwargs.get('s', 5)
    outFile = kwargs.get('outFile', None)
    
    #> make sure new mask columns exist
    if ('algo_list' not in df.columns) or ('algo' not in df.columns):
        df = applyMasks(df)
    
    class_cols = ['EC', 'F', 'CM', 'Cm', 'A']
    
    colors = {'EC': 'tab:blue',
              'F': 'tab:orange',
              'CM': 'tab:green',
              'Cm': 'tab:red',
              'A': 'tab:purple'}
    
    labels = {'EC': 'Cross',
              'F': 'Fold',
              'CM': 'Major cusp',
              'Cm': 'Minor cusp',
              'A': 'Unknown'}
    
    #> initializing plot
    fig, axes = plt.subplots(1, 2, figsize=(6*2+0.5, 6*1))
    
    #> plotting
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        for class_col in class_cols:
            
            #> separates data using new applyMasks output
            # class_type = class_to_type[class_col]
            
            if class_col == 'A':
                sub = df[df['algo'] == 'A']
            else:
                sub = df[df['algo_list'].apply(lambda x: class_col in x)]
            
            if len(sub) == 0:
                continue
            
            color = colors[class_col]
            label = labels[class_col]
                
            if i == 0:
                ax.scatter(sub['dt1'], sub['dt2_t12'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(keys['dt1'], fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(keys['dt2_t12'], fontsize=fontsize)
                
            elif i == 1:
                # ax.axis('off')
                ax.scatter(sub['xs'], sub['ys'], s=s,
                           alpha=alpha, color=color, label=label)
                ax.set_xlabel(r'$\mathbf{\Delta RA ~~[arcsec]}$', fontsize=fontsize, labelpad=labelpad)
                ax.set_ylabel(r'$\mathbf{\Delta DEC ~~[arcsec]}$', fontsize=fontsize)

        if i == 8:
            
            #> setting limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            low, high = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
    
    #> turnning off, to put legend
    # axes.flat[8].axis('off')
    
    #> shared legend below entire figure
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    
    legend = fig.legend(handles, legend_labels,
                        loc='lower center', bbox_to_anchor=(0.5, -0.1),
                        ncol=len(legend_labels), fontsize=15,
                        frameon=False, markerscale=4,
                        columnspacing=1.5, handletextpad=0.4)
    
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    
    if outFile is not None:
        plt.savefig(outFile+'.png', bbox_inches='tight', dpi=200)
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return
    

""" #> CONFIGURATIONS ================
================================== """

#> applying criteria
def getCriteria(df):
    
    #> cross / cusp
    df['dt1'] = (df['t12'] - df['t34'])
    
    #> fold / cusp
    df['dt2_t12'] = (df['t12'] - 2 * df['t23'])
    df['dt2_t34'] = (df['t34'] - 2 * df['t23'])
    
    return df


#> applys masks 
def applyMasks(df):
    
    #> gets criteria
    df = getCriteria(df)
    
    # cross_mask = ( (df['dt1'].abs() <= 20.0) & (df['dt2_t12'].abs() <= 20.0) )
    # fold_mask  = ( (df['dt1'].abs() <= 1000.0) & (df['dt2_t12'].abs() >= 30.0) )
    # major_cusp_mask = ( (df['t12'] < 120.0))
    # minor_cusp_mask = ( (df['t34'] < 120.0))

    #> gets masks
    lim_dt1 = 40.0
    lim_dt2_t1 = 30.0
    lim_dt2_t2 = 50.0
    masks = {'F':  (df['t23'] < 50.0) & (df['dt2_t12'].abs() > lim_dt2_t1) & (df['dt2_t34'].abs() > lim_dt2_t1),
             'EC': (df['t23'] > 65.0),
             #'A' : (df['t23'] > 65.0) & (df['t23'] < 70.0),
             'CM': (df['dt1'].abs() >= lim_dt1) & (df['dt2_t12'].abs() <= lim_dt2_t1) & (df['dt2_t34'].abs() >= lim_dt2_t2),
             'Cm': (df['dt1'].abs() >= lim_dt1) & (df['dt2_t34'].abs() <= lim_dt2_t1) & (df['dt2_t12'].abs() >= lim_dt2_t2),}
    
    #> initializes list of configs
    df['algo_list'] = [[] for _ in range(len(df))]

    #> applys masks
    for label, mask in masks.items():
        df.loc[mask, 'algo_list'] = df.loc[mask, 'algo_list'].apply(lambda x: x + [label])

    #> combine list to single type string
    df['algo'] = df['algo_list'].apply(lambda x: '/'.join(x) if len(x) > 0 else 'A')

    return df


#> determines a lenses configuration
def config(inFile, verbose=False):
    
    #> imports
    import modules.parse as parse
    
    #> loading data
    df, _ = parse.fromNPYtoDF(inFile)

    #> applying masks
    df = applyMasks(df)
    
    #> total rows
    ttl = len(df)
    
    #> count every exact configuration
    config_counts = df['algo'].value_counts()
    
    #> only single-class counts
    ambiguous = config_counts.get('A', 0)
    only_EC = config_counts.get('EC', 0)
    only_F  = config_counts.get('F', 0)
    only_CM = config_counts.get('CM', 0)
    only_Cm = config_counts.get('Cm', 0)
    
    #> total overlap: 2 or more classifications
    overlap_mask = df['algo_list'].apply(lambda x: len(x) >= 2)
    overlap = overlap_mask.sum()
    overlap_counts = df.loc[overlap_mask, 'algo'].value_counts()
    
    #> printing
    if verbose:
        
        #> single class
        ttl_class = ambiguous + only_EC + only_F + only_CM + only_Cm
        print(f'> Total quads: {ttl}')
        print('> Single-class configurations:')
        print(f'    {'A':<12}: {ambiguous:>6}')
        print(f'    {'EC':<12}: {only_EC:>6}')
        print(f'    {'F':<12}: {only_F:>6}')
        print(f'    {'CM':<12}: {only_CM:>6}')
        print(f'    {'Cm':<12}: {only_Cm:>6}')
        print(f'    {'Sum':<12}: {ttl_class:>6}')
        print()
    
        #> overlap classes
        print(f'> Overlap configurations: {overlap}')
        if overlap > 0:
            for overlap_type, count in overlap_counts.items():
                frac = 100 * count / overlap
                print(f'    {overlap_type:<12}: {count:>5}  ({frac:5.1f}% of overlaps)')
        else:
            print('    None')
    
        print()
        
    return df


#> function for testing different thresholds
def thresholds(inFile, **kwargs):
    
    #> imports
    import modules.parse as parse
    import modules.geometry as geometry
    
    #> declarations
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23']
    
    #> loading data
    df, data = parse.fromNPYtoDF(inFile)
    
    #> loading comparison if there
    comparison_obs_file = kwargs.get('comparison_obs_file', None)
    if comparison_obs_file is not None:  
        comp_df, comp_data = parse.fromNPYtoDF(comparison_obs_file)
        comp_df = getCriteria(comp_df)
    
    #> get masks
    df = getCriteria(df)
    
    #> gets masks
    df = applyMasks(df)

    folds = df[df['algo_list'].apply(lambda x: 'F' in x)]
    cross = df[df['algo_list'].apply(lambda x: 'EC' in x)]
    maj_cusps = df[df['algo_list'].apply(lambda x: 'CM' in x)]
    min_cusps = df[df['algo_list'].apply(lambda x: 'Cm' in x)]
    
    # angleQuads(major_cusps, sort_value='t23', lens=lens)#, outFile=f'./figures/dt1_{dt1:.0f}_dt2_{dt2:.0f}_srcs')
    # angleQuads(minor_cusps) #, outFile=f'./figures/dt1_{dt1:.0f}_dt2_{dt2:.0f}')
    
    sort_value='t23'
    triPanel(min_cusps, sort_value=sort_value, lens=data['lens'],
             comp_df=df)# , outFile=f'./figures/tri_dt2_{sort_value}_q0.85_edge2.png')
    
    # configPlane(df)
    # configPlane(line, comparison=df, title=f'All Quads: {keys['t23']}', outFile=f'./figures/')
    # configPlane(ecross, comparison=df)
    # configPlane(folds, comparison=df)
    # configPlane(major_cusps, comparison=df)
    # configPlane(minor_cusps, comparison=df)
    
    return


""" #> OBS QUADS =====================
================================== """

#> plots the observed lens configs
def obs_angle_config(inFile):
    
    #> loading in csv
    df = pd.read_csv(inFile)
    
    #> saving names
    names = df['Name'].to_numpy()
    
    #> angle columns needed by angleQuad
    angle_cols = ['t12', 't23', 't34']
    
    #> make sure they are numeric
    angles = df[angle_cols].apply(pd.to_numeric, errors='coerce')
    
    #> check for bad rows
    bad_rows = angles[angles.isna().any(axis=1)]
    if len(bad_rows) > 0:
        print('> Rows with missing/non-numeric angle values:')
        print(df.loc[bad_rows.index])
        return
    
    #> plotting
    for i, row in enumerate(angles.to_numpy(dtype=np.float64)):
        angleQuad(row, name=names[i])
    
    return


#> checking config columns
def values_differ(left, right):
    
    #> checking vals
    left_values = {value.strip().upper() for value in str(left).split('/')}
    right_values = {value.strip().upper()for value in str(right).split('/')}

    #> true only when the two columns have no value in common
    return left_values.isdisjoint(right_values)


#> getting the obs config of lenses
def obs_config(inFile, verbose=False, plot=False):
    
    #> outFile name
    outFile = inFile[:-4] + '_algo.csv'

    #> loading in csv
    df = pd.read_csv(inFile)
    
    #> angle columns needed by angleQuad
    angle_cols = ['t12', 't23', 't34']
    angles = df[angle_cols].apply(pd.to_numeric, errors='coerce')
    bad_rows = angles[angles.isna().any(axis=1)]
    if len(bad_rows) > 0:
        print('Rows with missing/non-numeric angle values:')
        print(df.loc[bad_rows.index])
        return
    
    #> gets masks
    df = applyMasks(df)
    
    #> renaming columns
    df = df.rename(columns={'llrw_configs': 'llrw',
                            'jhmj_configs': 'jhmj',
                            'diff_flag': 'lj_diff',
                            'type_list': 'algo_list',
                            'type': 'algo'})
    
    columns = ['Name', 'llrw', 'jhmj', 'algo', 'aj_diff', 'al_diff', 'algo_diff']
    
    #> checking to see if type and jhmj_configs match (when diffflag = '-' )
    df['al_diff'] = [values_differ(algo, llrw) for algo, llrw in zip(df['algo'], df['llrw'])]
    df['aj_diff'] = [values_differ(algo, jhmj) for algo, jhmj in zip(df['algo'], df['jhmj'])]
    df['algo_diff'] = ( df['aj_diff'] & df['al_diff'] )
    
    #> if verbose
    # if verbose: print(df[columns].to_string())
    
    #> saving to csv
    df.to_csv(outFile, index=False)
    
    #> get the incorrect ones
    wrong = df[ df['algo_diff'] == True ]
    
    #> removing 'bad' quads
    bad_quads = ['SDSSJ1538+5817', 'HE1113-0641']
    wrong = wrong[ wrong['Name'].apply(lambda x: x not in bad_quads)].reset_index()
    
    #> printing
    if verbose:
        
        #> columns
        columns = ['Name', 't23', 'dt1', 'dt2_t12', 'dt2_t34', 'llrw', 'jhmj', 'algo']
        
        #> enumerating number wrong
        print(f'> There were {len(wrong.to_numpy())} incorrect configs!')
        print(wrong[columns].to_string())
        
        #> enumerating number of configs
        configs = ['EC', 'CM', 'Cm', 'F', 'A']
        num_configs = []
        for config in configs:
            num = len(df[ df['algo'].apply(lambda x: config in x.split('/'))].index)
            num_configs.append(num)
        config_df = pd.DataFrame([num_configs], columns=configs)

        ttl = len(df.index)
        config_df.loc[1] = [ round(x,1) for x in np.array(num_configs)/ttl*100 ]
        print(config_df.to_string())
        print(np.sum(config_df.loc[0]))
        
        
    #> plotting wrong
    if plot:
        # print(wrong.loc[0])
        for i in wrong.index:
            angleQuad(wrong.loc[i][['t12', 't23', 't34']].to_numpy(), 
                      name=wrong.loc[i]['Name'],
                      config=wrong.loc[i]['algo'])
    
    
    #> parsing for systems agreed upon
    # df = df[ df['diff_flag'] == '-' ]
    
    # print(df['type'].value_counts())
    # df[df['algo_list'].apply(lambda x: 'F' in x)]
    
    return df


#> for mock data: applys masks, prints %, and total
def configs(inFile, verbose=False):
    
    #> loading data
    df, data = parse.fromNPYtoDF(inFile)
    
    #> angle columns needed by angleQuad
    angle_cols = ['t12', 't23', 't34']
    angles = df[angle_cols].apply(pd.to_numeric, errors='coerce')
    bad_rows = angles[angles.isna().any(axis=1)]
    if len(bad_rows) > 0:
        print('> Rows with missing/non-numeric angle values:')
        print(df.loc[bad_rows.index])
        return
    
    #> gets masks
    df = applyMasks(df)
    
    #> if wanting print
    if verbose:
        
        #> printing # of configs
        configs = ['EC', 'CM', 'Cm', 'F', 'A']
        num_configs = []
        for config in configs:
            num = len(df[ df['algo'].apply(lambda x: config in x.split('/'))].index)
            num_configs.append(num)
        config_df = pd.DataFrame([num_configs], columns=configs)

        ttl = len(df.index)
        config_df.loc[1] = [ round(x,1) for x in np.array(num_configs)/ttl*100 ]
        print(config_df.to_string())
        print(np.sum(config_df.loc[0]))
    
    return df

#> compares t23 between mock and observed data
def t23_comparison(obsFile, mockFile):
    
    #> imports
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    
    #> loading obs data
    df = pd.read_csv(obsFile)
    obs_df = df[['Name', 't12', 't23', 't34', 'algo']]
    
    #> loading mock data
    mock_df = configs(mockFile)
    
    #> declarations
    angle = 't23'
    config_array = ['CM', 'Cm', 'EC', 'A', 'F']
    colors = ['#003049', '#669BBC', '#780000', '#C1121F', '#9999A1']
    alpha=1.0
    bins = 27
    lw=0.5
    fontsize=15
    
    plt.style.use('default')
    
    #> 
    data_sets = [obs_df, mock_df]
    titles = [f'Observed Quads ({len(obs_df.index)})', f'Mock Quads ({len(mock_df.index)})']
    suffix = ['obs', 'mock']
    
    for df, title, suff in zip(data_sets, titles, suffix):
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6+0.5,6))
    
        #> setting limits
        xmin = 0
        xmax = 90
        ax.set_xlim(xmin, xmax)
        ax.grid(ls=':', alpha=0.5)
        
        #> making list of datasets
        datasets = [df[df['algo'].apply(lambda x: config in x.split('/'))][angle] for config in config_array]
        
        #> plotting stacked histogram
        ax.hist(datasets, bins=bins, range=(xmin, xmax), stacked=True,
                label=config_array, alpha=alpha, color=colors, edgecolor='w', linewidth=lw)
        
        #> labels
        ax.set_title(title, fontweight='bold', fontsize=fontsize)
        ax.set_xlabel(r'$\mathbf{\theta_{23}}$', fontsize=fontsize, labelpad=10)
        # ax.set_ylabel('Count', fontweight='bold', fontsize=fontsize, labelpad=5)
        
        ax.yaxis.set_major_locator(MaxNLocator(7))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.get_yaxis().set_ticklabels([])
        
        plt.savefig('./figures/t23_breakdown_'+suff+'.png', bbox_inches='tight', dpi=200)
        plt.legend()
        plt.show(); plt.close()
    
    return
    

""" #> MAIN FUNCTION =================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> imports
    import modules.parse as parse
    
    #> declarations
    obsFile = '../notes/projects/quads/all_obs_quads_configs.csv'
    
    df = obs_config(obsFile, verbose=True, plot=False)

    #> files
    dataDir = './data/baseline/'
    name = '2606301438q75'
    mockFile = dataDir + name + '.npy'
    
    # config(obs_file, verbose=True)
    
    #> comparison
    comparison_name = '2606301434q75_m3'
    comparison_obs_file  = dataDir + comparison_name + '.npy'
    
    # df, _ = parse.fromNPYtoDF(mockFile)
    # biPanel(df, alpha=1, outFile='./figures/biPanel_configs')
    
    # thresholds(obs_file, comparison_obs_file=comparison_obs_file)
    configs(mockFile, verbose=True)
    
    obsFile = '../notes/projects/quads/all_obs_quads_configs_algo.csv'
    
    t23_comparison(obsFile, mockFile)
    
    # end
# thank