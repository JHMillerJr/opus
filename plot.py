# file name: plotting.py
# author: John Miller Jr
# modified for the Opus lensing code

""" #> IMPORTS =======================
================================== """

#> imports
import numpy as np
import pandas as pd
from pathlib import Path

#> plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#> modules
import profiles
from modules.units import u; u=u()
import modules.error as error


""" #> DECLARATIONS ==================
================================== """

#> figure out directory
global fig_loc
fig_loc = 'figures/'

#> plotting params
dpi, imSize = 600, 80
fontsize, labelpad = 15, 10

#> contour levels
kappa_levels = np.logspace(np.log10(0.1), np.log10(2.5), 35)
lpot_levels = np.logspace(np.log10(2000), np.log10(100000), 35)
kappa_multiples_levels = np.linspace(-0.01, 0.01, 35)

#> color cycle
colors = ['blue', 'red', 'orange', 'magenta', 'teal', 'limegreen', 'brown', 'darkviolet']


""" #> STYLE =========================
================================== """

#> automatically sets style
def setStyle():
    plt.rcParams['axes.prop_cycle']   = plt.cycler(color=colors)
    plt.rcParams['axes.titlesize']    = fontsize
    plt.rcParams['axes.titleweight']  = 'bold'
    plt.rcParams['axes.labelsize']    = fontsize
    plt.rcParams['axes.labelweight']  = 'bold'
    plt.rcParams['axes.labelpad']     = labelpad
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor']  = '1'
    plt.rcParams['savefig.dpi']       = dpi
    plt.rcParams['savefig.bbox']      = 'tight'
    return
setStyle()

#> formats one axis
def styleAx(ax, title=None, xlabel=None, ylabel=None, sky=False, grid=None):
    
    #> sky axes
    if sky:
        if xlabel is None: xlabel = 'RA [arcsec]'
        if ylabel is None: ylabel = 'DEC [arcsec]'
        ax.invert_xaxis()
    
    #> labels
    if title  is not None: ax.set_title(title, fontsize=fontsize, fontweight='bold')
    if xlabel is not None: ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad, fontweight='bold')
    if ylabel is not None: ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad, fontweight='bold')
    
    #> grid (default: on for data plots, off for maps)
    if grid is None: grid = not sky
    if grid: ax.grid(ls=':', alpha=0.5)
    
    return ax

#> legend
def legend(ax=None):
    if ax is None: ax = plt.gca()
    ax.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    return

#> zooms, saves, shows + closes a figure
def finish(fig, ax=None, outFile=None, zoom=1, loc=None):
    
    #> zoom function! :)
    if ax is not None and zoom != 1:
        ax.set_xlim(np.array(ax.get_xlim())/zoom)
        ax.set_ylim(np.array(ax.get_ylim())/zoom)
    
    #> saving (before showing, so the file is not blank)
    if loc is None: loc = fig_loc
    if outFile not in (None, False): fig.savefig(loc + str(outFile) + '.png', dpi=dpi, bbox_inches='tight')
    
    #> showing & closing
    plt.show(); plt.close(fig)
    
    return

#> makes a box
def toBox(ax, margin=0.1):
    
    #> gets margins
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    #> gets maximum
    b = max(abs(xmin), xmax, abs(ymin), ymax) * (1+margin)
    
    #> sets lims
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    
    return

#> sets the tick spacing
def ticks(ax, xnum=0.5, ynum=0.5, method='multiple'):
    
    #> imports
    import matplotlib.ticker as ticker
    
    #> multiple locator
    if method == 'multipole':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xnum))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ynum))
    
    return


""" #> IMAGES ========================
================================== """

#> plotting images on figure
def plotImages(ax, images, **kwargs):
    
    for im in images:
        ax.scatter(im[:,0],
                   im[:,1],
                   zorder=5, s=imSize, label=int(np.isfinite(im[:,0]).sum()))
    legend(ax)
    
    return


#> plotting images on figure
def plotArrival(ax, quad, **kwargs):
    
    for i, im in enumerate(quad):
        ax.scatter(im[0],im[1], zorder=5, s=imSize, label=f'{i+1}')
    legend(ax)
    
    return


""" #> LENS MAPS =====================
================================== """

#> lens-plane maps of one galaxy
def lensMaps(bprof, nph=50):
    
    #> lens-plane grid 
    pix_arc = bprof['pix_arc'] 
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc   # arcsec
    xg, yg = np.meshgrid(w, w)                           # arcsec
    
    #> deflection angles + jacobian
    ax, ay, axx, axy, ayx, ayy = profiles.alpha_jac_points(xg, yg, bprof) # arcsec, --
    
    #> kappa, shear, ccurves, mags
    kappa  = 0.5*(axx + ayy)
    gamma  = np.hypot(0.5*(axx - ayy), 0.5*(axy + ayx))
    lamt   = 1 - kappa - gamma
    lamr   = 1 - kappa + gamma
    mu     = 1 / (lamt*lamr)     # signed mags
    
    return {'x': xg, 'y': yg, 'ax': ax, 'ay': ay, 'kappa': kappa, 'gamma': gamma,
            'lamt': lamt, 'lamr': lamr, 'mu': mu, 'pix_arc': pix_arc}


#> critical curves and caustics
def critCaustics(maps, bprof):
    
    #> imports
    from skimage import measure
    
    #> declarations
    x0, y0  = maps['x'][0,0], maps['y'][0,0]   # arcsec
    arc_pix = maps['x'][0,1] - x0              # arcsec per pixel
    
    #> lam = 0 contours
    curves, caus = {}, {}
    for key in ['lamt', 'lamr']:
        
        #> lam = 0 contours
        cs_pix = measure.find_contours(maps[key], 0.0)
        
        #> pix --> arcsec
        cs = [np.column_stack((x0 + c[:,1]*arc_pix,
                               y0 + c[:,0]*arc_pix)) for c in cs_pix]
        
        #> back-projecting
        ca = []
        for c in cs:
            ax, ay = profiles.alpha_points(c[:,0], c[:,1], bprof)
            ca.append(np.column_stack((c[:,0] - ax, c[:,1] - ay)))
        
        #> saving
        curves[key] = cs
        caus[key]   = ca
    
    return (curves['lamt'], curves['lamr']), (caus['lamt'], caus['lamr']) # arcsec


""" #> BASIC GALAXY PLOTS ============
================================== """

#> plotting deflection angles
def deflect(bprof, nph=50, outFile=False, images=None):

    #> lens maps (arcsec)
    maps = lensMaps(bprof, nph)
    X, Y, delx, dely = maps['x'], maps['y'], maps['ax'], maps['ay']

    #> initializing plot
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    for i, ax in enumerate(axes.flat):
        
        #> setting ax
        styleAx(ax, sky=True, ylabel='DEC [arcsec]' if i == 0 else '')
        
        if i == 0: # gradx
            ax.set_title(r'$\alpha_x$ [arcsec]', fontweight='bold')
            gx = ax.contourf(X, Y, delx)
            
            #> plotting images
            if images is not None:
                ax.scatter(images[:,0], images[:,1], zorder=5, s=imSize)
            
        if i == 1: # grady
            ax.set_title(r'$\alpha_y$ [arcsec]', fontweight='bold')
            gy = ax.contourf(X, Y, dely)
            
            #> plotting images
            if images is not None:
                ax.scatter(images[:,0], images[:,1], zorder=5, s=imSize)
     
    #> colorbar
    plt.colorbar(gx); plt.colorbar(gy)
    
    #> saving, showing & closing
    finish(fig, outFile=outFile)
    
    return
    

#> plots kappa
def kappa(bprof, nph=50, images=None, outFile=False):
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    styleAx(ax, title='Kappa', sky=True)
    
    #> calculating kappa (analytic)
    maps = lensMaps(bprof, nph)
    X, Y, kappa = maps['x'], maps['y'], maps['kappa']
    
    #> plotting!
    cs = ax.contour(X, Y, kappa, levels=kappa_levels, colors='k')
    # ax.clabel(cs, cs.levels, fontsize=10)
    
    #> plotting images
    if images is not None:
        plotImages(ax, images)
    
    #> saving, showing & closing
    finish(fig, outFile=outFile)

    return

#> plots kappa
def kappaMultipoles(bprof, nph=50, images=None, outFile=False):
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    styleAx(ax, title='Kappa', sky=True)
    
    #> calculating kappa (analytic)
    maps = lensMaps(bprof, nph)
    X, Y, kappa = maps['x'], maps['y'], maps['kappa']
    
    #> plotting!
    cs = ax.contour(X, Y, kappa, levels=kappa_multiples_levels, colors='k')
    # ax.clabel(cs, cs.levels, fontsize=10)
    
    #> plotting images
    if images is not None:
        plotImages(ax, images)
    
    #> saving, showing & closing
    finish(fig, outFile=outFile)

    return
    

""" #> FIMS ==========================
================================== """

#> plots the rpts/bpts of fims
def fims(X, Y, deltx, delty, rpts, bpts):
    
    #> initializing plot
    fig, axes = plt.subplots(1,3,figsize=(6*3,6))
    
    #> ploting!
    for i, ax in enumerate(axes.flat):    
        
        #> deltx
        if i == 0:
            axes.set_title(r'$\Delta$X', fontweight='bold')
            axes.set_ylabel('DEC [arcsec]')
            gx = axes.contourf(X, Y, deltx)
            
        #> delty
        if i == 1:
            axes.set_title(r'$\Delta$Y', fontweight='bold')
            axes.set_ylabel('DEC [arcsec]')
            gx = axes.contourf(X, Y, delty)
        
        #> intersections
        if i == 2:
            axes.set_title(r'$\Delta$Y', fontweight='bold')
            axes.set_ylabel('DEC [arcsec]')
            gx = axes.contourf(X, Y, delty)

    return


""" #> MAGNIFICATION =================
================================== """
    
#> plots magnification map
def mag(bprof, nph=50, images=None, outFile=None, zoom=1):
    
    #> getting magnification map
    maps   = lensMaps(bprof, nph)
    magmap = np.log(np.abs(maps['mu']))
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(7,6))
    styleAx(ax, title=r'$\ln|\mu|$', sky=True)
    
    #> magnification map
    gx = ax.contourf(maps['x'], 
                     maps['y'], 
                     magmap, levels=50)
    
    #> plotting images
    if images is not None:
        if len(images.shape) != 3:
            images = np.array([images])
        plotImages(ax, images)
    
    #> colorbar
    plt.colorbar(gx)
    
    #> zooming, saving, showing & closing
    finish(fig, ax, outFile=outFile, zoom=zoom)
    
    return


""" #> CRITICAL CURVES/ CAUSTIC ======
================================== """

#> plots critical curves
def ccurves(bprof, nph=50,
            xs=None, ys=None, images=None, outFile=None, zoom=1):
    
    #> calculating kappa (overplots on kappa)
    maps  = lensMaps(bprof, nph)
    kappa = maps['kappa']
    
    #> getting critical curves
    (tcurve, rcurve), _ = critCaustics(maps, bprof)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    styleAx(ax, title='Critical Curves + Kappa', sky=True)
    
    #> plotting!
    cs = ax.contour(maps['x'], maps['y'], 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in tcurve: ax.plot(*line.T, c='b', label=r'$\lambda_t$')
    for line in rcurve: ax.plot(*line.T, c='r', label=r'$\lambda_r$')
    
    #> plotting source + images
    if xs is not None: ax.scatter(xs, ys, c='r', zorder=2, marker='*', s=100)
    
    #> plotting images
    if images is not None:
        if len(images.shape) != 3:
            images = np.array([images])
        plotImages(ax, images)
    
    #> zooming, saving, showing & closing
    finish(fig, ax, outFile=outFile, zoom=zoom)
    
    return


#> plotting caustics
def caustics(bprof, nph=50, 
             xs=None, ys=None, images=None, outFile=None, zoom=1):

    #> calculating kappa (overplots on kappa)
    maps  = lensMaps(bprof, nph)
    kappa = maps['kappa']
    
    #> getting caustics (tangential, radial)
    _, (dcaustic, ocaustic) = critCaustics(maps, bprof)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    styleAx(ax, title='Caustics + Kappa', sky=True)
    
    #> plotting!
    cs = ax.contour(maps['x'], maps['y'], 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in dcaustic: ax.plot(*line.T, c='b')
    for line in ocaustic: ax.plot(*line.T, c='r')
    
    #> plotting source + images
    if xs is not None: ax.scatter(xs, ys, c='r', zorder=2, marker='*', s=100)
    
    #> plotting images
    if images is not None:
        if len(images.shape) != 3:
            images = np.array([images])
        plotImages(ax, images)

        
    #> zooming, saving, showing & closing
    finish(fig, ax, outFile=outFile, zoom=zoom)
    
    return


#> plotting caustics
def causticBox(bprof, boxPoints, nph=50, 
               xs=None, ys=None, outFile=None, zoom=1, images=None):

    #> calculating kappa (overplots on kappa)
    maps  = lensMaps(bprof, nph)
    kappa = maps['kappa']
    
    #> getting caustics (tangential, radial)
    _, (dcaustic, ocaustic) = critCaustics(maps, bprof)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    styleAx(ax, title='Caustics + Kappa', sky=True)
    
    #> plotting!
    cs = ax.contour(maps['x'], maps['y'], 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in dcaustic: ax.plot(*line.T, c='b')
    for line in ocaustic: ax.plot(*line.T, c='r')
    
    #> plotting images
    if images is not None:
        ax.scatter(images[:,0],
                   images[:,1],
                   zorder=5, s=imSize, label=int(np.isfinite(images[:,0]).sum()))
        legend(ax)
      
    #> this needs to be ordered... but how??
    box = mpl.patches.Polygon(boxPoints, facecolor='none', 
                              edgecolor='b', linestyle='--',
                              linewidth=2)
    # ax.add_patch(box)
    
    #> plotting source
    if xs is not None: 
        ax.scatter(xs, ys, c='r', zorder=2, marker='*', s=100)
            
    #> zooming, saving, showing & closing
    finish(fig, ax, outFile=outFile, zoom=zoom)
    
    return


""" #> LENS OBS ======================
================================== """

#> three dimensional plot
def plot3D(fileName, observables):
    
    import matplotlib.pyplot as plt
    
    #> loading in image data
    images = np.load(fileName)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': '3d'})
    
    print(np.max(images[:,0]))
    print(np.max(images[:,1]))
    print(np.max(images[:,2]))
    
    ax.scatter(images[:,0], images[:,1], images[:,2], s=1, alpha=0.1, c='m')
    ax.view_init(elev=10, azim=-30) # Set initial view
    styleAx(ax, xlabel=observables[0], ylabel=observables[1], grid=False)
    ax.set_zlabel(observables[2], fontsize=fontsize, labelpad=labelpad, fontweight='bold')
    
    finish(fig)
    
    return


#> plotting population of quads
def plotPop(fileName, observables):
    
    import matplotlib.pyplot as plt
    
    #> loading in image data
    images = np.load(fileName)
    
    #> initializing plot
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    
    #> plotting!
    alpha = 0.01
    size=3
    for i, ax in enumerate(axes.flat):
        
        if i == 0:
            styleAx(ax, xlabel=observables[0], ylabel=observables[1])
            ax.set_ylim(-20, 20)
            ax.set_xlim(0, 100)
            ax.scatter(images[:,0], images[:,1], alpha=alpha, c='m', s=size)
            
        if i == 1:
            styleAx(ax, xlabel=observables[1], ylabel=observables[2])
            ax.set_ylim(0.6, 1.0)
            ax.set_xlim(-3, 3)
            ax.scatter(images[:,1], images[:,2], alpha=alpha, c='m', s=size)
    
    finish(fig)
    
    return


""" #> PRIORS ========================
================================== """

#> plots the distribution of priors
def priors_z(df, **kwargs):
    
    #> declarations
    keys = ['log10_M_vir', 'c_vir', 'log10_M_star', 'R_eff'] # priors.sample_priors columns
    suptitle = kwargs.get('suptitle', None)
    
    #> initializing plot
    fig, axes = plt.subplots(2,2, figsize=(13,12))
    if suptitle is not None: 
        plt.suptitle(suptitle, fontsize=fontsize, fontweight='bold', y=0.94)
    
    for i, ax in enumerate(axes.flat):
        
        styleAx(ax, xlabel=r'$\mathbf{z_l}$', ylabel=keys[i] if 'log10' in keys[i] else r'$\mathbf{log_{10}}$'+f'{keys[i]}')
        
        #> plotting
        ax.plot(df['zl'], df[keys[i]] if 'log10' in keys[i] else np.log10(df[keys[i]]))
    
    #> saving, showing & closing
    finish(fig, outFile=kwargs.get('outFile', None))
    
    return

#> plots prior correlations
def priors_correl(df, **kwargs):
    
    #> declarations
    keys = ['log10_M_vir', 'c_vir', 'log10_M_star', 'R_eff'] # priors.sample_priors columns
    suptitle = kwargs.get('suptitle', None)
    
    #> initializing plot
    fig, axes = plt.subplots(2,2, figsize=(12,12))
    if suptitle is not None: 
        plt.suptitle(suptitle, fontsize=fontsize, fontweight='bold', y=0.94)
    
    for i, ax in enumerate(axes.flat):
        
        styleAx(ax, xlabel=r'$\mathbf{z_l}$', ylabel=keys[i])
        
        #> halo mass function
        if i == 0:
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.scatter(df['zl'], df['log10_M_vir'])
            ax.set_ylabel(r'$\mathbf{log_{10}M_{vir}}$', fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            ax.set_xlabel(r'$\mathbf{z_l}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            
            
        #> halo mass-concentration relation
        if i == 1:
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.scatter(df['log10_M_vir'], df['c_vir'])
            ax.set_ylabel(r'$\mathbf{log_{10}c_{vir}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            ax.set_xlabel(r'$\mathbf{log_{10}M_{vir}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            
            
        #> stellar-to-halo mass relation
        if i == 2:
            
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            
            #> plotting
            ax.scatter(df['log10_M_vir'], df['log10_M_star'])
            ax.set_ylabel(r'$\mathbf{log_{10}M_{star}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            ax.set_xlabel(r'$\mathbf{log_{10}M_{vir}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            
            #> saving min and max
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            
            #> adding guiding lines
            x = np.linspace(xlims[0], xlims[1], 1000)
            ax.plot(x, x, c='k', label='+0 dex')
            ax.plot(x, x-1, c='k', ls='--', label='-1 dex')
            ax.plot(x, x-2, c='k', ls=':', label='-2 dex')
            
            #> resetting limits
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            
            legend(ax)
            
        #> stellar mass-size relation
        if i == 3:
            
            #> ticks
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
            
            #> plotting!
            ax.scatter(df['log10_M_star'], np.log10(df['R_eff']))
            ax.set_ylabel(r'$\mathbf{log_{10}R_{eff}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            ax.set_xlabel(r'$\mathbf{log_{10}M_{star}}$',fontsize=fontsize, labelpad=labelpad, fontweight='bold')
            
            #> saving min and max
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            
            print(xlims)
            
            #> 2010 Auger SLACs lenses
            x = np.linspace(xlims[0], xlims[1], 1000)
            a = 0.81
            b = 0.53
            ax.plot(x, (a*(x-11) + b), c='k', label='2010 Auger')
            legend(ax)
    
    #> saving, showing & closing
    finish(fig, outFile=kwargs.get('outFile', None))
    
    return

""" #> BPROFILES =====================
================================== """

#> plots the distribution of parameters in bprofiles
def bprof(bprofiles=None, csv_fileName='', 
          bprof_fileName='', keys=[], shape='square',
          outFile=None):
    
    #> imports
    from modules import parse
    
    #> checking to ensure at least one thing is provided
    mask = [(bprofiles is not None), (csv_fileName != ''), (bprof_fileName != '')]
    if sum(mask) == 0:
        error.phrase('Need to supply bprofiles, bprof_fileName, or csv_fileName')
    
    #> checking that multiple are not supplied
    if sum(mask) > 1:
        error.phrase('Too many supplied arguments. Only need to provide one')
    
    #> getting df if bprofiles is provided
    file = None
    if mask[0]: df = parse.bprof_to_df(bprofiles=bprofiles)
    if mask[1]: df = pd.read_csv(csv_fileName); file=csv_fileName
    if mask[2]: df = parse.bprof_to_df(fileName=bprof_fileName); file=csv_fileName
    
    #> df should be correct now
    columns = df.columns.to_numpy()
    
    #> parsing keys we don't care about
    parseKey = ['add', 'slope', 'x0', 'y0', 
                'mult3_theta', 'mult4_theta', 'name', '_m',
                'rhoc', 'sigCrit', 'angDist', 'rho0', 'radius',
                'hern_norm', 'hern_theta']
    for key in parseKey:
        mask = [key not in x for x in columns]
        columns = np.array(columns[mask])
    print(columns)
    
    #> determining how many params to plot
    if len(keys) == 0: numKeys = len(columns); keys = columns
    else: numKeys = len(keys)
    
    #> if shape is square
    if shape == 'square':
        
        #> determining square size
        r = int(np.sqrt(numKeys)) + 1
        keys = np.hstack([keys, np.array([None] * (r**2 - numKeys) )])
        array = keys.reshape(r, r)
        
        #> initializing plot
        fig, axes = plt.subplots(r,r, figsize=(6*r,6*r))
        
        #> plotting
        for i, ax in enumerate(axes.flat):
            
            #> variable
            key = keys[i]
            if key is None: 
                ax.axis('off')
                continue
            
            #> initializing
            styleAx(ax, xlabel=key)
            
            #> plotting
            ax.hist(df[key].astype(float), density=True, bins=15)
    
    #> if shape will depend on the profiles
    elif shape == 'prof':
        
        #> getting shape
        misc, nfw, hern, mult, ex = [], [], [], [], []
        arrays = [misc, nfw, hern, mult, ex]
        for key in keys:
            if 'nfw' in key: nfw.append(key); continue
            if 'hern' in key: hern.append(key); continue
            if 'mult' in key: mult.append(key); continue
            if 'ex' in key: ex.append(key); continue
            misc.append(key)
        
        #> determines shape
        lengths = np.array([len(x) for x in arrays])
        numRows = np.sum((lengths > 0))
        numCols = max(lengths)
        
        #> master array
        for i, array in enumerate(arrays):
            arrays[i] = np.hstack((array, [None] * (numCols - lengths[i]) ))
        arrays = np.array(arrays)
        
        #> initializing plot
        fig, axes = plt.subplots(numRows,numCols, figsize=(6*numCols,6*numRows))
        
        #> settin title
        if file is not None:
            file = str(file)
            title = file.replace('\\', '/').split('/')[-1]
            plt.suptitle(title, fontweight='bold', fontsize=20, y=0.9)
        
        #> plotting
        for i, ax in enumerate(axes.flat):
            
            #> getting key
            key = arrays.flatten()[i]
            
            if key is None: 
                ax.axis('off')
                continue
            
            #> initializing
            styleAx(ax, xlabel=key)
            
            #> plotting
            ax.hist(df[key].astype(float), density=True, bins=15)
        
        pass
    

    
    #> saving (outFile is a full path here), showing & closing
    finish(fig, outFile=outFile, loc='')
    
    return


""" #> PNGS TO GIF ===================
================================== """

#> converts a set of pngs to a gif
def pngs_to_gif(loc, scale=1.0, colors=256, duration=100):

    #> imports
    import os
    import re
    import glob
    from PIL import Image

    #> pngs
    files = glob.glob(os.path.join(loc,'*.png'))
    files = sorted(files,
                   key=lambda x: int(re.findall(r'\d+',os.path.basename(x))[-1]))

    #> target size
    im0 = Image.open(files[0])
    size = (int(im0.width*scale),
            int(im0.height*scale))

    #> frames
    images = []

    for file in files:

        im = Image.open(file).convert('RGB')
        im = im.resize(size)

        #> reduce colors
        im = im.convert('P', palette=Image.ADAPTIVE, colors=colors)
        images.append(im)

    #> gif
    images[0].save(os.path.join(loc,'dynamic_pix_arc.gif'),
                   save_all=True,
                   append_images=images[1:],
                   duration=duration,
                   loop=0,
                   optimize=True)

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    
    loc = '../opus_lmfi/data/260909_comp1/'
    pattern = '*.csv'
    target=loc
    
    
    #> iterating through files 
    files = Path(loc).rglob(pattern)
    for file in files:
        new_fileName = target + (str(file)[:-4]).replace('\\', '/').split('/')[-1]
        bprof(csv_fileName=file, outFile=new_fileName, shape='prof')
    
    file = '../opus_lmfi/data/260909_comp1/26090914180800_pop01/26090914180800_pop01_bprofiles.npy'
    #bprof(bprof_fileName=file,
    #      outFile=file[:-4],
    #      shape='prof')
    
    # end
# thank