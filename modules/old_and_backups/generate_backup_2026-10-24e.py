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
    plotDict = {'kappa':         kwargs.get('kappa',          True), 
                'deflect':       kwargs.get('deflect',       False),
                'ccurves':       kwargs.get('ccurves',       False),
                'caustics':      kwargs.get('caustics',       True), 
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
    saveDict = {'images':      kwargs.get('images',      True), 
                'im_obs':      kwargs.get('im_obs',      True), 
                'im_mags':     kwargs.get('im_mags',     True),
                'delays':      kwargs.get('delays',      True), 
                'sources':     kwargs.get('sources',     True),
                'bprofiles':   kwargs.get('bprofiles',   True),
                'csv':         kwargs.get('csv',         True),
                'paramRanges': kwargs.get('paramRanges', True),
                'nims':        kwargs.get('nims',        True), # number of images per source
                'flag':        kwargs.get('flag',        True), # true if nims > jmax
                'gal':         kwargs.get('gal',         True), # galaxy index (into bprofiles) of each source
                'scprox':      kwargs.get('scprox',      True),
                'order_comp':  kwargs.get('order_comp',  True), # morph vs fermat arrival numbers (lpot)
                'order_match': kwargs.get('order_match', True), # morph vs fermat match (lpot)
                'ER':          kwargs.get('ER',          True)} # einstein radius: image mean radius, sqrt(area/pi)
    
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

#> saves the output of gen() (stacked or split by galaxy)
#> data/<folder>/<fileName>/<fileName>_<key>.npy, fileName = yymmddHHMM + ms + suffix
def save(out, **kwargs):
    
    #> globals
    global maxTries
    tries = maxTries
    
    #> imports
    from pathlib import Path
    
    #> kwargs
    saveDict    = kwargs.get('saveDict', getSaveDict())   # what to save
    folder      = kwargs.get('folder', '+unsorted')       # data output folder (inside dataDir)
    suffix      = kwargs.get('suffix', '')                # suffix to file name
    parent_dir  = kwargs.get('parent_dir', '')            # parent dir where the data folder is
    paramRanges = kwargs.get('paramRanges', None)         # galaxy params (not part of gen() output)
    
    #> directory path
    if folder != '' and folder[-1] != '/': folder += '/'
    if parent_dir != '' and parent_dir[-1] != '/': parent_dir += '/'
    dirPath = parent_dir + dataDir + folder
    
    #> file name from the time (+ millisecond info), new time if it already exists
    fileName = ''
    while (fileName == '' or os.path.isdir(dirPath + fileName)) and tries > 0:
        dt = str(int( (datetime.now().second*1e3 + datetime.now().microsecond/1e4) / 10 )).zfill(4) # second + millisecond info
        fileName = time.strftime('%y%m%d%H%M', time.localtime()) + dt + suffix + '/'
        tries -= 1
    
    #> making directory
    parent_dir = dirPath + fileName
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    
    
    #>>># STACKING GALAXIES (if needed) #<<<#
    
    #> one row per source
    data = out if 'images' in out else parse.stack(out, paramRanges=paramRanges)
    if paramRanges is not None: data = data | {'paramRanges': paramRanges}
    
    
    #>>># SAVING #<<<#
    
    #> saveDict key: data key (file = <fileName>_<data key>)
    ends = {'images': 'images', 'im_obs': 'obs', 
            'im_mags': 'mags' , 'delays': 'delays', 
            'sources': 'srcs' , 'nims': 'nims', 
            'flag': 'flag'    , 'gal': 'gal',
            'bprofiles': 'bprofiles', 'paramRanges': 'paramRanges',
            'scprox': 'scprox',
            'order_comp': 'order_comp', 'order_match': 'order_match',
            'ER': 'ER'}
    
    #> saving arrays
    for skey, key in ends.items():
        if saveDict.get(skey, False) and key in data: np.save(parent_dir + fileName[:-1] + f'_{key}', data[key])
    if saveDict.get('csv', False):
        df = data['df'] if 'df' in data else parse.bprof_to_df(bprofiles=data['bprofiles'])
        df.to_csv(parent_dir + fileName[:-1] + '_df.csv', index=False)
    
    #> print
    print(f'> Saved {len(data["nims"])} sources from {len(data["bprofiles"])} galaxies to {parent_dir}')
    
    return parent_dir


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
    order_comp  = kwargs.get('order_comp', False)   # compares morph vs fermat order of quads (lpot only)
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
    
    
    #> tangential eigenvalue (caustic + source-caustic proximity)
    axy_, axx_ = np.gradient(ax, w, w)
    ayy_, ayx_ = np.gradient(ay, w, w)
    kappa   = 0.5*(axx_ + ayy_) 
    gamma1  = 0.5*(axx_ - ayy_)
    gamma2  = 0.5*(axy_ + ayx_)
    lamt = 1 - kappa - np.hypot(gamma1, gamma2)
    
    
    #>>># LENSING #<<<#
    
    #> sources
    if srcs is None:
        
        #> calculating tangential caustic
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
            
            #> if requesting source-caustic proximity
            if kwargs.get('scprox', False): 
                out['scprox'] = np.full((numSource, 2), np.nan)
            
            #> if requesting einstein radii
            if kwargs.get('ER', False): 
                out['ER'] = np.full((numSource, 2), np.nan)
            
            #> if comparing morph vs fermat order
            if order_comp and method == 'lpot':
                out['order_comp']  = np.zeros((numSource, 5), dtype=int)
                out['order_match'] = np.full(numSource, np.nan)
            
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
            
        #> getting source-caustic proximity
        if kwargs.get('scprox', False):
            scprox = np.column_stack(newtlens.tcaus_prox(xg, yg, bprof, lamt, srcs)) # (N, 2): distance [arcsec], distance / sqrt(t_area) []
            
    else: #> if sources are provided
        srcs = np.asarray(srcs, dtype=float)
        res = newtlens.find_images(xg, yg, bxg, byg, srcs, bprof)
        
        #> getting source-caustic proximity
        if kwargs.get('scprox', False):
            scprox = np.column_stack(newtlens.tcaus_prox(xg, yg, bprof, lamt, srcs)) # (N, 2): distance [arcsec], distance / sqrt(t_area) []
        
        
    #>>># LENSING OBSERVABLES #<<<#
    
    #> declarations
    ims  = res['images']     # image positions [arcsec]
    mu   = res['mu']         # image magnifications
    quad = res['nims'] == 5  # quad mask
    flag = res['flag']       # true if nims > jmax
    
    #> out dictionary (fills with nans if needed)
    out = {'images': np.full_like(ims, np.nan), 
           'mu':     np.full_like(mu, np.nan),
           'srcs':   srcs,                      # sources
           'nims':   res['nims'],               # number of images
           'flag':   flag,                      # true if nims > jmax
           'delays': np.full_like(mu, np.nan)}  # time delays [days] (nan for deflect)
    
    #> if using lens potential (fermat order)
    if method == 'lpot':
        
        #> returns images and mags in fermat order (for nan: tau = inf, sorted last)
        sel = res['nims'] > 0  # every source w/ images
        out['images'][sel], out['mu'][sel], tau = order.fermat_order(ims[sel], mu[sel], srcs[sel], bprof)
        
        #> gets time delays (nan for padding)
        D_dt = (1 + bprof['zl']) * (4 * np.pi * u.G_kpc_solMass) * bprof['angDist']**2 * bprof['sigCrit'] / u.c_kpc**2 # kpc
        out['delays'][sel] = np.where(np.isinf(tau), np.nan, D_dt / u.c_kpc * (tau - tau[:,:1]) / 86400) # days
        
        #> morph vs fermat order (quads): arrival number (1-5) of each morph image (0 = not a quad), match (1/0, nan = not a quad)
        if order_comp:
            out['order_comp']  = np.zeros((len(srcs), 5), dtype=int)
            out['order_match'] = np.full(len(srcs), np.nan)
            if quad.any(): out['order_comp'][quad], out['order_match'][quad] = order.morph_vs_fermat(out['images'][quad][:,:5])
    
    else: #> if only using deflection angles (morphological order, quads only)
        out['images'][quad,:5], out['mu'][quad,:5] = order.morph_order(ims[quad][:,:5], mu[quad][:,:5])
        out['images'][flag], out['mu'][flag]       = ims[flag], mu[flag]
        if order_comp and verbose: error.highlight("order_comp needs method='lpot' (no arrival times), skipping")
    
    #> observables (quads only)
    if observables is not None:
        out['obs'] = np.full((len(srcs), len(observables)), np.nan)                     # padding
        out['obs'][quad] = geometry.quad_obs(out['images'][quad][:,:5], observables)    # calculating
        
    #> if getting 
    if kwargs.get('scprox', False):
        out['scprox'] = scprox
    
    #> einstein radius [arcsec]: (0) mean distance of the 4 outer images to the lens center (quads), (1) root of the tangential critical curve area / pi
    if kwargs.get('ER', False):
        out['ER'] = np.full((len(srcs), 2), np.nan)
        r = np.sort(np.hypot(out['images'][quad][:,:5,0], out['images'][quad][:,:5,1]), axis=1) # arcsec
        out['ER'][quad,0] = r[:,1:].mean(axis=1)                                             # outer 4 images
        out['ER'][:,1]    = np.sqrt(newtlens.tcrit_area(xg, yg, lamt) / np.pi)
    
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
    cosmo  = dict(kwargs.get('cosmo', u.cosmo)) # cosmology dict
    cosmo |= {k: kwargs[k] for k in ['h0', 'omega_m_0', 'omega_lam_0'] if k in kwargs}  # changes params if passed
    for p1, p2 in zip(('omega_m_0', 'omega_lam_0'), ('omega_lam_0', 'omega_m_0')):      # ensures flat cosmology
        if p1 in kwargs and p2 not in kwargs: cosmo[p2] = 1 - cosmo[p1] 
    if abs(cosmo['omega_m_0'] + cosmo['omega_lam_0'] - 1) > 1e-6:                       # if not flat, highlight message
        error.highlight(f'non-flat cosmology: omega_m_0 + omega_lam_0 = {cosmo["omega_m_0"] + cosmo["omega_lam_0"]:.3f}')
    
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
    nfw_priors  = kwargs.get('nfw_priors' , True)              # halo mass function + mass-concentration relation
    hern_priors = kwargs.get('hern_priors', True)              # stellar-to-halo mass relation + mass-size relation
    red_priors  = kwargs.get('red_priors' , redshifts is None) # redshift distribution (drawn if no redshifts given)
    priorDict   = kwargs.get('priorDict', params.getPriorDict(nfw=nfw_priors, hern=hern_priors, red=red_priors))
    selDict     = kwargs.get('selDict', params.getSelDict(**kwargs)) # selection effects (res, slcs, ER_resolution, ...)
    
    #> if only one redshift is supplied, tile
    if redshifts is not None and np.ndim(redshifts) == 1: 
        redshifts = np.tile(np.array(redshifts, dtype=object), (numGals, 1))
    
    #> setting pix_arc
    if pix_arc is None:                        # autoscaled if drawing from priors
        pix_arc = paramRanges['pix_arc']       # if not supplied, revert to paramRanges default
    
    #> galaxies
    bprofiles   = params.bprofiles(**(kwargs | {'numGals':   numGals,                   # number of galaxies
                                                'ranges':    paramRanges,               # galaxy params
                                                'pix_arc':   np.full(numGals, pix_arc), # pix/arcsec
                                                'redshifts': redshifts,                 # (numGals, 2) or None
                                                'priorDict': priorDict,                 # what to draw from priors
                                                'selDict':   selDict,                   # selection effects
                                                'nph':       nph,                       # half pixel grid width (autoscaling)
                                                'seed':      seed,                      # random seed
                                                'cosmo':     cosmo}))                   # cosmology
    
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
        o   |= {'bprof': bprof}
        out |= {parse.bnum(i, numGals): o}
    
    #> one row per source (same format as the saved files), unless stack=False
    if kwargs.get('stack', True): out = parse.stack(out, paramRanges=paramRanges)
    
    return out


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    #> test generate
    observables = getObservables(add=['d23'])
    out = gen(numGals=10, numSource=1000, observables=observables, scprox=True)
    parent_dir = save(out)
    data = parse.load(parent_dir)

    # end
# thank