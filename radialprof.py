#> name: radialprof.py
#> author: John Miller Jr
#> descrp: measures the 2D radial density profile for Opus lenses

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline

#> modules
import cosmology
from modules.units import u; u=u()
import modules.error as error 

#> declarations
fontsize = 15
labelpad = 10
fig_loc = 'figures/'
dpi = 600


""" #> RADIAL PROF FIT ===============
================================== """

#> finds power-law slope to radial prof
def radialprof_slope(log_r_arc, log_kappa_2d_r):
        return np.polyfit(log_r_arc, log_kappa_2d_r, 1)


""" #> RADIAL PROF ===================
================================== """

#> calculates the radial profile of a lens from a grid
#> can also interpolate kappa, R_eff in kpc
def grid_radialprof(kappa, R_eff, pix_arc, zl, **kwargs):
    
    #> declarations
    ny, nx = kappa.shape       # shape of kappa
    nph = nx / 2               # width / 2
    
    #> kwargs
    interp = kwargs.get('interp', 1)                    # how file to interpolate (1, 2, etc.)
    plot = kwargs.get('plot', False)                    # plot flag
    outFile = kwargs.get('outFile', False)              # where to save file
    center = kwargs.get('center', ((nx-1)/2, (ny-1)/2)) # center of summation
    
    #> checking for errors
    if nx != ny: error.phrase('kappa grid is not square')
    if int(nph) != nph: error.phrase('nph is not an integer')
    else: nph = int(nph)
    if int(interp) != interp: error.phrase('interp is not an integer')
    else: interp = int(interp)
    if interp < 1: error.phrase('interp must be >= 1')
    
    #> grid coordinates relative to center
    x0, y0 = center
    x = np.arange(nx) - x0
    y = np.arange(ny) - y0
    
    #> interpolating kappa
    f_kappa = spline(y, x, kappa)
    
    #> interpolated grid
    if interp > 1:
        x_interp = np.linspace(x[0], x[-1], (nx-1)*interp + 1)
        y_interp = np.linspace(y[0], y[-1], (ny-1)*interp + 1)
        kappa_interp = f_kappa(y_interp, x_interp)
    else: # no interp
        x_interp, y_interp = x, y
        kappa_interp = kappa
    
    #> radial grid
    xx, yy = np.meshgrid(x_interp, y_interp)
    rr = np.sqrt(xx**2 + yy**2)
    
    #> radial bins
    r_max = rr.max()                         # maximum r value
    dr = 1 / interp                          # dr width
    r_edges = np.arange(0, r_max + dr, dr)   # edges of pixel
    r_bin = (r_edges[:-1] + r_edges[1:]) / 2 # center of pixel

    #> assigning pixels (in grid) to radial bins
    inds = np.digitize(rr.flatten(), r_edges) - 1
    
    #> more declarations (eat me)
    kappa_r_bin     = np.zeros(len(r_bin))
    kappa_r_bin_pop = np.zeros(len(r_bin))
    
    #> iterating through grid
    for i in range(len(r_bin)):
        
        #> getting mask
        mask = (inds == i)
        
        #> saving to bin
        kappa_r_bin[i] = np.sum(kappa_interp.flatten()[mask])
        kappa_r_bin_pop[i] = np.sum(mask)
    
    #> removing empty bins
    mask = kappa_r_bin_pop > 0
    r_bin = r_bin[mask]
    kappa_r_bin = kappa_r_bin[mask]
    kappa_r_bin_pop = kappa_r_bin_pop[mask]
    
    #> normalizing
    kappa_2d_r = kappa_r_bin / kappa_r_bin_pop
    
    #> converting
    r_arc = r_bin / pix_arc                                           # pix-->arcsec
    R_eff_arc = (R_eff / 1000) / cosmology.angDist(0, zl) * u.arc_rad # kpc-->arcsec
    print(R_eff_arc)
    
    #> removing small radii
    percent = 10
    log10Rmin = (1-percent/100) * np.log10(R_eff_arc)
    log10Rmax = (1+percent/100) * np.log10(R_eff_arc)
    log_r_arc = np.log10(r_arc)
    print(log_r_arc)
    print(log10Rmin, log10Rmax)
    log_kappa_2d_r = np.log10(kappa_2d_r)
    log_mask = (log_r_arc > log10Rmin) & (log_r_arc < log10Rmax)
    
    #> calculating power law
    gamma, norm = radialprof_slope(log_r_arc[log_mask], log_kappa_2d_r[log_mask])
    
    #> plotting
    if plot:
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.grid(ls=':', alpha=0.5)
        ax.set_title('Circularly Averaged Kappa', fontsize=fontsize, fontweight='bold')
        ax.set_xlabel(r'$\mathbf{log_{10} r ~[arcsec]}$', fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(r'$\mathbf{log_{10} \kappa(r)}$', fontsize=fontsize, labelpad=labelpad)
        
        #> plotting!
        ax.plot(log_r_arc, gamma*log_r_arc + norm, c='r', ls=':', lw=3, label=r'$\gamma=$'+f'{gamma:.2f}')
        ax.plot(log_r_arc, log_kappa_2d_r, c='k', lw=4, alpha=1)
        
        ax.legend(fontsize=fontsize)
        
        #> saving and closing
        outFile = kwargs.get('outFile', None)
        if outFile is not None: 
            plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
    
    return r_arc, kappa_2d_r, gamma, norm


#> calculates the radial profile of a lens analytically (from a much finer grid)
def analytic_radialprof(bprofiles, **kwargs):
    
    
    
    return


""" #> BATCH FROM FILE ===============
================================== """

#> calculates the 2d density profile from a batch of galaxies
def batchRadialProfs(folder, **kwargs):
    
    #> imports
    import glob
    import generate
    
    #> kwargs
    plot = kwargs.get('plot', False)
    
    #> getting bprofiles from folder
    files = glob.glob(folder+'*bprofiles*.npy')
    if len(files) != 1:      # if not singlar bprofiles file
        if len(files) == 0: error.phrase(f'No bprofiles in {folder}')
        if len(files) > 1: error.phrase(f'Multiple bprofiles in {folder}')
    file = files[0] # reducing to one
    
    #> loading file
    bprofiles = np.load(file, allow_pickle=True)

    #> getting redshifts
    redshifts, R_effs = [], []
    for i, b in enumerate(bprofiles):
        redshifts.append([b['zl'], b['zs']])
        R_effs.append([b['hern'][0]['effrad']])
    redshifts = np.array(redshifts)
    R_effs = np.array(R_effs)
    
    #> generating galaxies
    lenses, bprofiles = generate.genGalPop(redshifts, bprofiles=bprofiles)
    xgrid, ygrid, delx, dely, lamt = lenses
    
    #> calculating power-law slopes
    slopes = np.zeros(shape=(len(delx),1))
    for i, delx_, dely_, R_eff in zip(range(len(slopes)), delx, dely, R_effs):
        kappa = lensing.angles_to_kappa(delx_, dely_, bprofiles[i]['pix_arc'])
        r_arc, kappa_2d_r, gamma, norm = grid_radialprof(kappa=kappa, 
                                                         R_eff=R_eff, 
                                                         pix_arc=bprofiles[i]['pix_arc'], 
                                                         zl=bprofiles[i]['zl'],
                                                         plot=False)
        slopes[i] = gamma
        
    #> plotting
    if plot:
        
        #> initializing plot
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        ax.grid(ls=':', alpha=0.5)
        ax.set_xlabel(r'$\mathbf{\gamma}$', fontsize=fontsize, labelpad=labelpad)
        ax.hist(slopes)
        
        plt.show()
        
    print(slopes)
    print(np.mean(slopes))
    print(np.std(slopes))
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    import os
    print('> '+os.path.basename(__file__))
    
    import generate
    import lensing
    import plot
    
    if False:
        data, paramRanges = generate.genGal(redshifts=(0.5, 1.0), 
                                            galProfiles=generate.galProfiles(mult=[], ex=False))
        xgrid, ygrid, xdef, ydef, lamt = data
        kappa = lensing.angles_to_kappa(xdef, ydef, paramRanges['pix_arc'])
        plot.kappa(xgrid, ygrid, xdef, ydef, paramRanges['pix_arc'])
        
        grid_radialprof(kappa, paramRanges['pix_arc'], paramRanges['zl'], plot=True)
    
    folder = './data/+unsorted/26081312592309/'
    batchRadialProfs(folder, plot=True)
    
    # end
# thank