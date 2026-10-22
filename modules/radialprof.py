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
    center = kwargs.get('center', (nph, nph))           # center of summation (theta=0 on the np.arange(-nph, nph) grid)
    
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
    
    #> checking that the fit window (up to 1.1 R_eff) is inside the grid (largest full circle)
    r_grid = nph / pix_arc                                            # arcsec
    if (1+percent/100) * np.max(R_eff_arc) > r_grid:
        error.highlight(f'R_eff = {float(np.max(R_eff_arc)):.2f} arcsec is not within the grid (half-width = {r_grid:.2f} arcsec); increase nph')
        return r_arc, kappa_2d_r, np.nan, np.nan
    log10Rmin = np.log10((1-percent/100) * R_eff_arc)
    log10Rmax = np.log10((1+percent/100) * R_eff_arc)
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


#> calculates the radial profile of a lens analytically (circle sampling, no grid)
#> kappa is averaged around circles of radius r centered on the hernquist profile, then fit over [rmin, rmax]*R_eff
def analytic_radialprof(bprofiles, **kwargs):
    
    #> imports
    import profiles
    
    #> kwargs
    rmin    = kwargs.get('rmin', 0.5)       # inner edge of the fit window [R_eff]
    rmax    = kwargs.get('rmax', 2.0)       # outer edge of the fit window [R_eff]
    nr      = kwargs.get('nr', 50)          # number of radii (log-spaced)
    ntheta  = kwargs.get('ntheta', 64)      # number of angles per circle
    plot    = kwargs.get('plot', False)     # plot flag
    outFile = kwargs.get('outFile', None)   # where to save file
    
    #> one galaxy --> list
    if isinstance(bprofiles, dict): bprofiles = [bprofiles]
    numGals = len(bprofiles)
    
    #> declarations
    th      = np.linspace(0, 2*np.pi, ntheta, endpoint=False)               # rad
    r_arc   = np.zeros((numGals, nr))                                        # arcsec
    kappa_r = np.zeros((numGals, nr))                                        # --
    gamma   = np.zeros(numGals)                                              # slope of log kappa vs log r
    norm    = np.zeros(numGals)                                              # intercept
    
    #> iterating thru galaxies
    for i, b in enumerate(bprofiles):
        
        #> effective radius + center (hernquist profile)
        hern   = b['hern'][0]
        R_eff  = (hern['effrad'] / 1000) / cosmology.angDist(0, b['zl']) * u.arc_rad # kpc --> arcsec
        x0, y0 = hern.get('x0', 0.0), hern.get('y0', 0.0)                   # arcsec
        
        #> circles of radius r (log-spaced over the fit window)
        r = R_eff * np.logspace(np.log10(rmin), np.log10(rmax), nr)          # arcsec
        x = x0 + r[:,None]*np.cos(th)[None,:]                                # arcsec, (nr, ntheta)
        y = y0 + r[:,None]*np.sin(th)[None,:]                                # arcsec, (nr, ntheta)
        
        #> kappa on each circle (analytic jacobian), averaged over angle
        _, _, axx, _, _, ayy = profiles.alpha_jac_points(x, y, b)
        kappa = 0.5*(axx + ayy)                                              # (nr, ntheta)
        
        #> saving + power-law fit
        r_arc[i], kappa_r[i] = r, kappa.mean(axis=1)
        gamma[i], norm[i] = radialprof_slope(np.log10(r_arc[i]), np.log10(kappa_r[i]))
    
    #> plotting (first 10 galaxies)
    if plot:
        
        #> initializing plot
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.grid(ls=':', alpha=0.5)
        ax.set_title('Circularly Averaged Kappa', fontsize=fontsize, fontweight='bold')
        ax.set_xlabel(r'$\mathbf{log_{10} r ~[arcsec]}$', fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(r'$\mathbf{log_{10} \kappa(r)}$', fontsize=fontsize, labelpad=labelpad)
        
        #> plotting!
        for i in range(min(numGals, 10)):
            lr = np.log10(r_arc[i])
            ax.plot(lr, np.log10(kappa_r[i]), lw=4, alpha=0.5, c=f'C{i}')
            ax.plot(lr, gamma[i]*lr + norm[i], ls=':', lw=2, c=f'C{i}', label=r'$\gamma=$'+f'{gamma[i]:.2f}')
        
        ax.legend(fontsize=fontsize)
        
        #> saving and closing
        if outFile is not None: 
            plt.savefig(fig_loc + outFile + '.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    
    return r_arc, kappa_r, gamma, norm # arcsec, --, --, --


""" #> BATCH FROM FILE ===============
================================== """

#> calculates the 2d density profile from a batch of galaxies
def batchRadialProfs(folder, **kwargs):
    
    #> imports
    import glob
    
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
    
    #> calculating power-law slopes (analytic, circle sampling)
    _, _, gamma, _ = analytic_radialprof(bprofiles, **(kwargs | {'plot': False}))
    slopes = gamma[:,None]
        
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
    import params
    import plot
    
    if False:
        bprof = params.bprofiles(1, ranges=params.toggleParams(generate.galProfiles(mult=[], ex=False)), 
                                 redshifts=np.array([[0.5, 1.0]]))[0]
        kappa = plot.lensMaps(bprof)['kappa']
        plot.kappa(bprof)
        
        grid_radialprof(kappa, bprof['hern'][0]['effrad'], bprof['pix_arc'], bprof['zl'], plot=True)
    
    folder = './data/+unsorted/26081312592309/'
    batchRadialProfs(folder, plot=True)
    
    # end
# thank