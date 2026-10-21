#> name: generate.py
#> author: John Miller Jr
#> descrp: main .py file to generate lenses

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import numpy as np
from datetime import datetime

#> declarations
dataDir = './data/'
figDir = './figures/'
sys.path.append(os.path.dirname(__file__)) 

#> modules
from modules.units import u; u=u()
import modules.geometry as geometry
import modules.error as error
import modules.parse as parse


""" #> DEFAULTS ======================
================================== """

global maxTries
maxTries = 100

#> returns plot dictionary
def getPlotDict(atleast_one_plot=True, **kwargs):
    
    #> imports
    import copy

    #> default plotting dict
    plotDict = {'kappa': kwargs.get('kappa', True), 
                'deflect': kwargs.get('deflect', False),
                'ccurves': kwargs.get('ccurves', False),
                'caustics': kwargs.get('caustics', True), 
                'caustics_zoom': kwargs.get('caustics_zoom', False)}
    
    #> making all false if wanted
    if not atleast_one_plot:
        plotDict = dict.fromkeys(plotDict, False)
    
    return copy.copy(plotDict)

#> returns the save dictionary
def getSaveDict(atleast_one_save=True, **kwargs):
    
    #> imports
    import copy

    #> default plotting dict
    saveDict = {'images': kwargs.get('images', True), 
                'im_obs': kwargs.get('im_obs', True), 
                'im_mags': kwargs.get('im_mags', False),
                'delays': kwargs.get('delays', False), 
                'sources': kwargs.get('sources', True),
                'lens': kwargs.get('lens', False), 
                'bprofiles': kwargs.get('bprofiles', True),
                'csv': kwargs.get('csv', True),
                'paramRanges': kwargs.get('paramRanges', True)}
    
    #> making all false if wanted
    if not atleast_one_save:
        saveDict = dict.fromkeys(saveDict, False)
        print(saveDict)
    
    return copy.copy(saveDict)

#> default lensing observables
def getObservables(add=None):
    
    #> base observables
    obs = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23', 'd01']
    
    #> available options
    available = ['x1', 'x2', 'x3', 'x4', 'x5',        # raw x positions
                 'y1', 'y2', 'y3', 'y4', 'y5',        # raw y positions 
                 'd01', 'd02', 'd03', 'd04',          # distance between orign & images
                 'd13', 'd14', 'd23', 'd24',          # distance between images
                 'd2/d1', 'd3/d1', 'd4/d1', 'd3/d2',  # distance ratios
                 't12', 't23', 't34', 'dt23']         # polar angles
    
    #> adding 
    if add is not None:
        
        #> ensuring correct format
        try: add = np.array(add)
        except: error.phrase(f'add is not an array: add={add}')
        
        #> checking to see if requested are available
        mask = np.array([a in available for a in add])
        
        #> adding
        obs = np.hstack([obs, add[np.nonzero(mask)]])
        
        #> printing not added
        if any(~mask):
            error.phrase(f'The following observables are not available: {add[np.nonzero(~mask)]}')
        
    return obs



""" #> GALPROFILES ===================
================================== """

#> default galProfiles
def galProfiles(nfw=True, hern=True, mult=[1,3,4], ex=True):
    
    #> '2' in mults not allowed --> remove, turn on EX (m=2)
    if 2 in mult: 
        mult.remove(2)  # remove m=2
        ex = True       # turns on EX
        error.highlight('m=2 is just ex, removing m=2 and setting ex=True')
    
    #> default 'on' mass profiles
    galProfiles = {'nfw': nfw, 'hern': hern,  # should always be true
                   'mult': mult, 'ex': ex}  # toggled
    
    return galProfiles


""" #> SAVING ========================
================================== """

#> saves info to outfile
def saveInfo(dirPath, fileName, suffix, out, **kwargs):
    
    #> globals
    global maxTries
    tries = maxTries
    
    #> imports
    from pathlib import Path
    
    #> adding / if needed
    if dirPath[-1]  != '/': dirPath  += '/'
    if fileName[-1] != '/': fileName += '/'
    
    #> declarations
    original_file_name = fileName

    #> checking to see if file name already exists, chooses new time if already exists
    while os.path.isdir(dirPath + fileName) and tries > 0:
        
        #> getting new time
        dt = str(int( (datetime.now().second*1e3 + datetime.now().microsecond/1e4) / 10 )).zfill(4) # second + millisecond info
        fileName = time.strftime('%y%m%d%H%M', time.localtime())
        fileName += dt + suffix + '/'
        tries -= 1
    
    #> printing if file name has changed
    if tries < maxTries:
        print(f'> The file {original_file_name} has been changed to {fileName}')
    
    #> making directory
    parent_dir = dirPath + fileName
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    
    #> iterating through out keys
    for key in out.keys():
        
        #> if wanting bprofiles csv
        if key in ['csv']:
            # parse.bprof_to_df(bprofiles, save=True, outFile=parent_dir + fileName[:-1]+'_df.csv')
            continue
        
        #> all other keys
        np.save(parent_dir + fileName[:-1] + f'_{key}', out[key])

    return


""" #> NEWTON METHOD =================
================================== """

#> lenses numSource sources behind one galaxy (all vectorized, newton method)
def genQuadNewt(bprof, numSource, nph=50, rng=None, **kwargs):

    #> modules
    import profiles
    import newtlens
    import order
    
    #> unpacking kwargs
    method      = kwargs.get('method', 'deflect')   # 'deflect' (morph order) or 'lpot' (fermat order + delays)
    observables = kwargs.get('observables', None)   # list of lenisng observables
    srcs        = kwargs.get('srcs', None)          # supplied sources [arcsec] (otherwise random in caustic)
    maxRedraw   = kwargs.get('maxRedraw', 10)       # redraw rounds for non-quads
    verbose     = kwargs.get('verbose', True)       # if wanting print statements
    if rng is None: rng = np.random
    
    #> declarations
    pix_arc = bprof['pix_arc']
    
    
    #>>># CREATING GALAXY #<<<#
    
    #> lens-plane grid (theta)
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc
    xg, yg = np.meshgrid(w, w)
    
    #> deflection angles, calculating beta
    ax, ay   = profiles.alpha_points(xg, yg, bprof)    # arcsec
    bxg = xg - ax          # beta_i = theta_i - alpha_i [arcsec]
    byg = yg - ay
    
    
    #>>># LENSING #<<<#
    
    #> sources
    if srcs is None:
        
        #> calculating tangential caustic
        axy_, axx_ = np.gradient(ax, w, w)
        ayy_, ayx_ = np.gradient(ay, w, w)
        kappa   = 0.5*(axx_ + ayy_) 
        gamma1  = 0.5*(axx_ - ayy_)
        gamma2  = 0.5*(axy_ + ayx_)
        lamt = 1 - kappa - np.hypot(gamma1, gamma2)
        caus = newtlens.caustic_points(xg, yg, bprof, lamt)
        
        #> no critical curve
        if caus is None:
            
            #> filling requested data with nans
            out = {'images': np.full((numSource, 5, 2), np.nan), 
                   'mu':     np.full((numSource, 5),    np.nan),
                   'srcs':   np.full((numSource, 2),    np.nan),    
                   'nims':   np.zeros(numSource,     dtype=int), 
                   'flag':   np.zeros(numSource,    dtype=bool),
                   'delays': np.full((numSource, 5),    np.nan)}
            
            #> if requesting obsrevables
            if observables is not None: 
                out['obs'] = np.full((numSource, len(observables)), np.nan)
            
            #> printing and returning
            error.highlight(f'Caustic does not exist! bprof={bprof}')
            return out
        
        #> drawing sources inside the caustic
        srcs = newtlens.sample_sources(caus, numSource, rng=rng)
        
        #> lensing!
        res = newtlens.find_images(xg, yg, bxg, byg, srcs, bprof)
        
        #> redrawing non-quads
        nEven = 0  # redraws due to even image count (a missed image)
        for _ in range(maxRedraw):
            
            #> finds non-quads
            bad = np.flatnonzero(res['nims'] != 5)
            if len(bad) == 0: break #> all quads
            
            #> counting even-count failures
            nEven += int(np.sum(res['nims'][bad] % 2 == 0))
            
            #> redraws bad sources + replaces
            srcs[bad] = newtlens.sample_sources(caus, len(bad), rng=rng)
            new_res = newtlens.find_images(xg, yg, bxg, byg, srcs[bad], bprof)
            w = max(res['mu'].shape[1], new_res['mu'].shape[1])
            
            #> resaving
            for i in ['images', 'mu']: 
                res[i]     = newtlens.pad(res[i], w)
                new_res[i] = newtlens.pad(new_res[i], w)
            for i in res: res[i][bad] = new_res[i]
        
        #> printing even-count failures
        if nEven > 0 and verbose: 
            print(f'> {nEven} redraw(s) were even-count failures (missed images)')
            
    else: #> if sources are provided
        srcs = np.asarray(srcs, dtype=float)
        res = newtlens.find_images(xg, yg, bxg, byg, srcs, bprof)
        
        
    #>>># LENSING OBSERVABLES #<<<#
    
    #> declarations
    ims  = res['images']     # image positions [arcsec]
    mu   = res['mu']         # image magnifications
    quad = res['nims'] == 5  # quad mask
    
    #> out dictionary (fills with nans if needed)
    out = {'images': np.full_like(ims, np.nan), 
           'mu':     np.full_like(mu, np.nan),
           'srcs':   srcs,                      # sources
           'nims':   res['nims'],               # number of images
           'delays': np.full_like(mu, np.nan)}  # time delays [days] (nan for deflect)
    
    #> if using lens potential (fermat order)
    if method == 'lpot':
        
        #> returns imgaes and mags in fermat order
        o_ims, o_mu, tau = order.fermat_order(ims[quad][:,:5], mu[quad][:,:5], srcs[quad], bprof)
        
        #> gets time delays
        D_dt = (1 + bprof['zl']) * (4 * np.pi * u.G_kpc_solMass) * bprof['angDist']**2 * bprof['sigCrit'] / u.c_kpc**2 # kpc
        out['delays'][quad,:5] = D_dt / u.c_kpc * (tau - tau[:,:1]) / 86400 # days
    
    else: #> if only using deflection angles (morphological order)
        o_ims, o_mu = order.morph_order(ims[quad][:,:5], mu[quad][:,:5])
        
    #> saving images and mags
    out['images'][quad,:5] = o_ims
    out['mu'][quad,:5]     = o_mu

    #> raw (unordered) images and mags of sources w/ more than jmax images
    flag = res['flag']
    out['images'][flag] = ims[flag]
    out['mu'][flag]     = mu[flag]
    out['flag']         = flag         # true if nims > jmax

    # !!! (i think i only need this extra step when flag na method == 'deflect')
    #> fermat orders all images of sources w/ more than jmax image
    if method == 'lpot' and flag.any():
        out['images'][flag], out['mu'][flag], f_tau = order.fermat_order(ims[flag], mu[flag], srcs[flag], bprof)
        out['delays'][flag] = np.where(np.isinf(f_tau), np.nan, D_dt / u.c_kpc * (f_tau - f_tau[:,:1]) / 86400) # days
    
    #> observables
    if observables is not None:
        out['obs'] = np.full((len(srcs), len(observables)), np.nan) # padding
        out['obs'][quad] = geometry.quad_obs(o_ims, observables)    # calculating
    
    return out

""" #> GENERATE ======================
================================== """

#> main generate function
def gen(**kwargs):
    
    #> imports
    import params
    
    #> random
    seed        = kwargs.get('seed', 42)        # random seed
    rng         = kwargs.get('rng', np.random.default_rng(seed))
    
    #> cosmology
    cosmo                = u.cosmo              # default cosmology dict
    cosmo['h0']          = 70.0                 # hubble param
    cosmo['omega_m_0']   = 0.27                 # matter density
    cosmo['omega_lam_0'] = 1-cosmo['omega_m_0'] # cosmological constant
    
    #> unpacking kwargs
    numGals     = kwargs.get('numGals'  ,    1) # number of galaxies to lens from
    numSource   = kwargs.get('numSource',    1) # number of sources per galaxy
    nph         = kwargs.get('nph'      ,   50) # half pixel grid width
    pix_arc     = kwargs.get('pix_arc'  , None) # pixel per arcsec conversion
    redshifts   = kwargs.get('redshifts', None) # redshifts of sources
    observables = kwargs.get('observables', getObservables(add=kwargs.get('add', None)))
    
    #> initializing galaxies
    nfw         = kwargs.get('nfw' , True)      # nfw profile
    hern        = kwargs.get('hern', True)      # hernquist profile
    mult        = kwargs.get('mult', [1,3,4])   # multipoles (i.e., m=1,3,4,...)
    ex          = kwargs.get('ex'  , True)      # external shear
    galProf     = galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)
    paramRanges = params.toggleParams(galProf)
    
    #> galaxy parameter priors
    nfw_priors      = kwargs.get('nfw_priors' , True)
    hern_priors     = kwargs.get('hern_priors', True)
    if nfw_priors or hern_priors: # if using galaxy priors
        
        #> imports
        import priors
        
        #> 
        
    else: # if not drawing from nfw or hern priors
        
        pass
        
    # if pix_arc = None, turn on autoscaling
    # if redshifts = None, random sampling
    

    bprofiles   = params.bprofiles(numGals=numGals, 
                                   ranges=paramRanges, 
                                   pix_arc=np.full(numGals, pix_arc), 
                                   redshifts=redshifts)
    
    #> removing keys passed not in kwargs
    popKeys = ['numGals', 'numSource', 'nph', 'rng']
    for key in popKeys:
        if kwargs.get(key, None) is not None:
            kwargs.pop(key)
    
    #> adding kwargs
    addKeys = [{'observables': observables}]
    for dic in addKeys:
        key = list(dic.keys())[0]
        if kwargs.get(key, None) is None:
            kwargs |= dic

    #> generating quads
    #> combine and add bprof index
    out = {}
    for i, bprof in enumerate(bprofiles):
        
        #> generating
        o = genQuadNewt(bprof=bprof,         # all galaxy parameters
                        numSource=numSource, # number sources per galaxy
                        nph=nph,             # half pixel grid width
                        rng=rng,             # random generator
                        **kwargs)            # everything else! yippee!
        
        #> saving to out
        bnum = str(i+1).zfill( np.max([2, int(np.log10(numGals))]) )
        o   |= {'bprof': bprof}
        out |= {f'{bnum}': o}
    
    return out


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    observables = getObservables(add=['d23'])
    print(observables)
    
    out = gen(numGals=10, numSource=10, observables=observables)
    print(out['01'].keys())

    # end
# thank