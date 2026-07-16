# file name: plotting.py
# author: John Miller Jr
# modified for the Opus lensing code

""" #> IMPORTS =======================
================================== """

#> imports
import numpy as np

#> plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

#> classes
import lensing

#> modules
from modules.units import u; u=u()
import modules.error as error


""" #> DECLARATIONS ==================
================================== """

#> figure out directory
global fig_loc
fig_loc = 'figures/'

#> saving parameters
dpi = 600
imSize = 80

#> time order labels
orderLabels = True

#> contour levels
kappa_levels = np.logspace(np.log10(0.1), np.log10(2.5), 35)
lpot_levels = np.logspace(np.log10(2000), np.log10(100000), 35)
kappa_multiples_levels = np.linspace(-0.01, 0.01, 35)

#> color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'red', 'orange', 'magenta',
                                                    'teal', 'limegreen', 'brown', 'darkviolet']) 
# cycle_colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


""" #> IMAGES ========================
================================== """

#> plotting images on figure
def plotImages(ax, images, **kwargs):
    
    for im in images:
        ax.scatter(im[:,0],
                   im[:,1],
                   zorder=5, s=imSize, label=len(im[:,0]))
    plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    
    return


#> plotting images on figure
def plotArrival(ax, quad, **kwargs):
    
    for i, im in enumerate(quad):
        ax.scatter(im[0],im[1], zorder=5, s=imSize, label=f'{i+1}')
    plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    
    return


""" #> BASIC GALAXY PLOTS ============
================================== """

#> plotting deflection angles
def deflect(X, Y, delx, dely, outFile=False, images=None):

    # initializing plot
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    for i, axes in enumerate(ax.flat):
        
        axes.set_xlabel('RA [arcsec]')
        axes.invert_xaxis() 
        
        if i == 0: # gradx
            axes.set_title('delx', fontweight='bold')
            axes.set_ylabel('DEC [arcsec]')
            gx = axes.contourf(X, Y, delx)
            
            #> plotting images
            if images is not None:
                axes.scatter(images[:,0], images[:,1], zorder=5, s=imSize)
            
        if i == 1: # grady
            axes.set_title('dely', fontweight='bold')
            gy = axes.contourf(X, Y, dely)
            
            #> plotting images
            if images is not None:
                axes.scatter(images[:,0], images[:,1], zorder=5, s=imSize)
     
    #> colorbar
    plt.colorbar(gx); plt.colorbar(gy)
    
    #> saving and closing
    if outFile != False: plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
    
    #> showing & closing
    plt.show(); fig.clear(); plt.close()
    
    return
    

#> plots kappa
def kappa(X, Y, delx, dely, pix_arc, images=None, outFile=False):
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Kappa', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> calculating kappa (overplots on kappa)
    gradxy, gradxx = np.gradient(delx * pix_arc * u.arc_rad)
    gradyy, gradyx = np.gradient(dely * pix_arc * u.arc_rad)
    kappa = 0.5 * (gradxx + gradyy)
    
    #> plotting!
    cs = ax.contour(X / pix_arc, Y / pix_arc, kappa, levels=kappa_levels, colors='k')
    # ax.clabel(cs, cs.levels, fontsize=10)
    
    #> plotting images
    if images is not None:
        plotImages(ax, images)
    
    #> saving and closing
    if outFile != False: plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
        
    #> showing & closing
    plt.show(); fig.clear(); plt.close()

    return

#> plots kappa
def kappaMultipoles(X, Y, delx, dely, pix_arc, images=None, outFile=False):
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Kappa', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> calculating kappa (overplots on kappa)
    gradxy, gradxx = np.gradient(delx * pix_arc * u.arc_rad)
    gradyy, gradyx = np.gradient(dely * pix_arc * u.arc_rad)
    kappa = 0.5 * (gradxx + gradyy)
    
    print(np.max(kappa))
    print(np.min(kappa))
    print(np.mean(kappa))
    print(np.std(kappa))
    
    #> plotting!
    cs = ax.contour(X / pix_arc, Y / pix_arc, kappa, levels=kappa_multiples_levels, colors='k')
    # ax.clabel(cs, cs.levels, fontsize=10)
    
    #> plotting images
    if images is not None:
        plotImages(ax, images)
    
    #> saving and closing
    if outFile != False: plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
        
    #> showing & closing
    plt.show(); fig.clear(); plt.close()

    return
    

""" #> FIMS ==========================
================================== """

#> plots the rpts/bpts of fims
def fims(X, Y, deltx, delty, rpts, bpts):
    
    #> initializing plot
    fig, axes = plt.subplots(1,3,figsize=(6*3,6))
    #ax.set_title('Kappa', fontweight='bold')
    #ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    #ax.invert_xaxis() 
    
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
def mag(x, y, gradx, grady, pix_arc, images=None, outFile=None, zoom=1):
    
    #> getting magnification map
    magmap = np.log(np.abs(lensing.magnification(gradx, grady, pix_arc)))
    
    # initializing plot
    fig, ax = plt.subplots(figsize=(7,6))
    ax.set_title('Magnification', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> magnification map
    gx = ax.contourf(x / pix_arc, 
                     y / pix_arc, 
                     magmap, levels=50)
    
    #> plotting images
    if images is not None:
        for i, src in enumerate(images.keys()): # each source
            ax.scatter(images[src][:,0],
                       images[src][:,1],
                       zorder=5, s=imSize, label=src)
        plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    
    #> colorbar
    plt.colorbar(gx)
    
    #> zoom function! :)
    ax.set_xlim(np.array(ax.get_xlim())/zoom)
    ax.set_ylim(np.array(ax.get_ylim())/zoom)
    
    #> saving and closing
    if outFile is not None: 
        plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
        
    #> showing & closing
    plt.show(); fig.clear(); plt.close()
    
    return


""" #> CRITICAL CURVES/ CAUSTIC ======
================================== """

#> plots critical curves
def ccurves(x, y, gradx, grady, pix_arc,
            xs=None, ys=None, images=None, outFile=None, zoom=1):
    
    #> calculating kappa (overplots on kappa)
    gradxy, gradxx = np.gradient(gradx * pix_arc * u.arc_rad)
    gradyy, gradyx = np.gradient(grady * pix_arc * u.arc_rad)
    kappa = 0.5 * (gradxx + gradyy)
    
    #> getting critical curves
    tcurve, rcurve = lensing.ccurves(x, y, gradx, grady, pix_arc)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Critical Curves + Kappa', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> plotting!
    cs = ax.contour(x / pix_arc, y / pix_arc, 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in tcurve: ax.plot(*line.xy, c='b', label=r'$\lambda_t$')
    for line in rcurve: ax.plot(*line.xy, c='r', label=r'$\lambda_r$')
    
    #> plotting source + images
    if xs is not None: ax.scatter(xs, ys, c='r', zorder=2, marker='*', s=100)
    
    #> plotting images
    if images is not None:
        ax.scatter(images[:,0],
                   images[:,1],
                   zorder=5, s=imSize, label=len(images[:,0]))
        plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    
    #> zoom function! :)
    ax.set_xlim(np.array(ax.get_xlim())/zoom)
    ax.set_ylim(np.array(ax.get_ylim())/zoom)
    
    #> saving and closing
    if outFile is not None: 
        plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
    
    #> showing & closing
    plt.show(); plt.close()
    
    return


#> plotting caustics
def caustics(x, y, gradx, grady, pix_arc, 
             xs=None, ys=None, images=None, outFile=None, zoom=1):

    #> calculating kappa (overplots on kappa)
    gradxy, gradxx = np.gradient(gradx * pix_arc * u.arc_rad)
    gradyy, gradyx = np.gradient(grady * pix_arc * u.arc_rad)
    kappa = 0.5 * (gradxx + gradyy)
    
    #> getting caustics
    dcaustic, ocaustic = lensing.caustics(x, y, gradx, grady, pix_arc)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Caustics + Kappa', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> plotting!
    cs = ax.contour(x / pix_arc, y / pix_arc, 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in dcaustic: ax.plot(*line.xy, c='b')
    for line in ocaustic: ax.plot(*line.xy, c='r')
    
    #> plotting source + images
    # if xs.all() != False: 
    ax.scatter(xs, ys, c='r', zorder=2, marker='.', s=100)
    
    #> plotting images
    if images is not None:
        plotImages(ax, images)

        
    #> zoom function! :)
    ax.set_xlim(np.array(ax.get_xlim())/zoom)
    ax.set_ylim(np.array(ax.get_ylim())/zoom)
            
    #> saving and closing
    if outFile is not None: 
        plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
    
    #> showing & closing
    plt.show(); plt.close()
    
    return


#> plotting caustics
def causticBox(x, y, gradx, grady, pix_arc, boxPoints, 
               xs=None, ys=None, outFile=None, zoom=1, images=None):

    #> calculating kappa (overplots on kappa)
    gradxy, gradxx = np.gradient(gradx * pix_arc * u.arc_rad)
    gradyy, gradyx = np.gradient(grady * pix_arc * u.arc_rad)
    kappa = 0.5 * (gradxx + gradyy)
    
    #> getting caustics
    dcaustic, ocaustic = lensing.caustics(x, y, gradx, grady, pix_arc)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Caustics + Kappa', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.invert_xaxis() 
    
    #> plotting!
    cs = ax.contour(x / pix_arc, y / pix_arc, 
                    kappa, levels=kappa_levels, colors='k', alpha=0.5)
    for line in dcaustic: ax.plot(*line.xy, c='b')
    for line in ocaustic: ax.plot(*line.xy, c='r')
    
    #> plotting images
    if images is not None:
        ax.scatter(images[:,0],
                   images[:,1],
                   zorder=5, s=imSize, label=len(images[:,0]))
        plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
      
    #> this needs to be ordered... but how??
    box = mpl.patches.Polygon(boxPoints, facecolor='none', 
                              edgecolor='b', linestyle='--',
                              linewidth=2)
    ax.add_patch(box)
    
    #> plotting source
    if xs is not None: 
        ax.scatter(xs, ys, c='r', zorder=2, marker='*', s=100)
            
    #> saving and closing
    if outFile is not None: 
        plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
    
    #> zoom function! :)
    ax.set_xlim(np.array(ax.get_xlim())/zoom)
    ax.set_ylim(np.array(ax.get_ylim())/zoom)
    
    #> showing & closing
    plt.show(); plt.close()
    
    return


""" #> LENS OBS ======================
================================== """

#> three dimensional plot
def plot3D(fileName, observables):
    
    import matplotlib.pyplot as plt
    
    #> loading in image data
    images = np.load(fileName)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    
    print(np.max(images[:,0]))
    print(np.max(images[:,1]))
    print(np.max(images[:,2]))
    
    ax.scatter(images[:,0], images[:,1], images[:,2], s=1, alpha=0.1, c='m')
    ax.view_init(elev=10, azim=-30) # Set initial view
    
    plt.show()
    
    return


#> plotting population of quads
def plotPop(fileName, observables):
    
    import matplotlib.pyplot as plt
    
    #> loading in image data
    images = np.load(fileName)
    
    #> plotting image data
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    
    #> plotting!
    fontsize = 15
    alpha = 0.01
    size=3
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.3)
        
        if i == 0:
            ax.set_ylim(-20, 20)
            ax.set_xlim(0, 100)
            ax.scatter(images[:,0], images[:,1], alpha=alpha, c='m', s=size)
            ax.set_xlabel(observables[0], fontsize=fontsize, fontweight='bold')
            ax.set_ylabel(observables[1], fontsize=fontsize, fontweight='bold')
            
        if i == 1:
            ax.set_ylim(0.6, 1.0)
            ax.set_xlim(-3, 3)
            ax.scatter(images[:,1], images[:,2], alpha=alpha, c='m', s=size)
            ax.set_xlabel(observables[1], fontsize=fontsize, fontweight='bold')
            ax.set_ylabel(observables[2], fontsize=fontsize, fontweight='bold')
    
    return
    

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    # end
# thank