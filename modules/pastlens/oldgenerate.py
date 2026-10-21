#> name: generate.py
#> author: John Miller Jr
#> descrp: main .py file to generate lenses

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import random
import numpy as np
from datetime import datetime

#> declarations
dataDir = './data/'
figDir = './figures/'
sys.path.append(os.path.dirname(__file__)) 

#> modules
from modules.units import u; u=u()
import plot
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


""" #> GRID ==========================
================================== """

#> returns grid
def grid(**kwargs):
    nph = kwargs.get('nph', u.nph)
    w_range = np.arange(-nph, nph, 1, dtype=float)
    return np.meshgrid(w_range, w_range)


""" #> GALPROFILES ===================
================================== """

#> default galProfiles
def galProfiles(nfw=True, hern=True, mult=[1,3,4], ex=True):
    
    #> '2' in mults not allowed --> remove, turn on EX (m=2)
    if 2 in mult: 
        mult.remove(2)  # remove m=2
        ex = True        # turns on EX
    
    #> default 'on' mass profiles
    galProfiles = {'nfw': nfw, 'hern': hern,  # should always be true
                   'mult': mult, 'ex': ex}  # toggled
    
    return galProfiles


""" #> SINGLE ========================
================================== """
###### THESE WILL BE RUN ON CPU ######

#> generates single galaxy
def singleGal(sysConfig, **kwargs):
    
    #> unpacking
    redshifts, galProfs, nph, pix_arc = sysConfig
    zs = redshifts[1]
    
    #> unpacking kwargs
    plotDict = kwargs.get('plotDict', getPlotDict)
    
    #> generating single galaxy
    lens, paramRanges = genGal(redshifts, galProfs, nph=nph, pix_arc=pix_arc)
    
    #> generating guad(s)
    images, sources = genQuad(lens, zs, pix_arc=pix_arc, 
                              numSource_gal=kwargs.get('numSource_gal', 1), 
                              source=kwargs.get('source', None))
    
    #> ploting lenses
    if plotDict:
        import plot
        plot.kappa(*lens[:-1], pix_arc)
        plot.caustics(*lens[:-1], pix_arc, images=images)
        plot.caustics(*lens[:-1], pix_arc, images=images, zoom=3, xs=sources[0][0], ys=sources[0][1])
        
    return


#> generates single galaxy
def genGal(redshifts, galProfiles=galProfiles(), **kwargs):
    
    #> unpacking
    zl, zs = redshifts
    
    #> grid declarations
    pix_arc = kwargs.get('pix_arc', u.pix_arc)
    
    #> getting paramRanges
    import params
    if kwargs.get('bprofile', None) is None:
        paramRanges = params.toggleParams(galProfiles)
        bprofile = params.bprofiles(1, paramRanges, 
                                    zl=[zl], zs=[zs], pix_arc=[pix_arc],
                                    verbose=False, cov=None, mu=None)
        print(bprofile)
    else:
        bprofile = kwargs.get('bprofile', None)
    
    #> grid declarations
    nph = kwargs.get('nph', u.nph)
    xgrid, ygrid = grid(nph=nph)
    
    #> calculating deflection angles
    import deflection
    delx, dely, lamt = deflection.deflection(xgrid, ygrid, bprofile, gpu=False)
    
    return (xgrid, ygrid, delx[0], dely[0], lamt[0]), paramRanges


#> generates single quad
def genQuad(lens, **kwargs):
    
    #> declarations
    global maxTries
    
    #> unpacking
    xgrid, ygrid, delx, dely, lamt = lens
    nph = int(len(xgrid)/2)
    pix_arc = kwargs.get('pix_arc', u.pix_arc)
    mags = kwargs.get('mags', False)
    
    #> images + source declarations
    requested_jims = kwargs.get('jims', None)
    numSource_gal = kwargs.get('numSource_gal', 1)
    observables = kwargs.get('observables', None)
    
    #> if wanting observables w/o specifiying jims
    quad_observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    if observables is not None and requested_jims is None:
        if any(x in observables for x in quad_observables):
            error.highlight('Requesting quad observables w/o setting jims=5!')
            error.highlight('Setting jims=5!')
            requested_jims = 5
    
    #> getting caustic / source positions
    import lensing
    if kwargs.get('source', None) is None:
        source_apexes = lensing.caustic(xgrid, ygrid, delx, dely, pix_arc, lamt)
    
    #> collecting all requested images
    images, im_obs, im_mags, sources = [], [], [], []
    for i in range(numSource_gal):
        
        #> enforcing requested num of jims
        ims, numTries, flag = [], maxTries, True
        while (len(ims) != requested_jims) and (numTries > 0) and flag:
        
            #> getting source positions
            if kwargs.get('source', None) is None:
                xs, ys = lensing.ranSources2(source_apexes)[0]
            else: 
                source = np.array(kwargs.get('source'))
                print(source.shape)
                if len(source.shape) == 1:
                    xs, ys = source
                    error.highlight('Requesting multiple sources w/ set source positions!')
                else:
                    if i >= len(source):
                        error.phrase(f'Not enough supplied sources! Want: {numSource_gal}, Supplied: {len(source)}')
                    xs, ys = source[i]
                        
            #> converting units
            xs /= u.arc_rad 
            ys /= u.arc_rad 
            
            #> lensing!
            ims = lensing.lfims(xgrid / pix_arc / u.arc_rad, # radians
                                ygrid / pix_arc / u.arc_rad, # radians
                                delx, dely,                  # radians
                                xs, ys,                      # radians 
                                pix_arc, nph)
            
            #> adjusting tries
            numTries-=1
            
            #> if there is no requested number of jims or requesting specific source
            if requested_jims is None or kwargs.get('source', None) is not None: 
                flag = False
                
        #> if wanting mags
        if mags: im_mags.append(getMag((xgrid, ygrid, delx, dely), pix_arc, ims))
                
        #> if wanting lensing observables
        if observables is not None and requested_jims == 5:
            obs = geometry.lensObs((0, 0), ims, observables)
            im_obs.append(obs)
            
        #> adding images
        images.append(ims)
        sources.append([xs * u.arc_rad , ys * u.arc_rad ])
        
        #> if could not find the number requested
        if numTries == 0:
            error.highlight(f'Source {i} could not find {requested_jims} images! Last source tried = ({xs},{ys})')
    
    #> to numpy
    images = np.array(images, dtype=object)
    sources = np.array(sources)
    im_obs = np.array(im_obs)
    im_mags = np.array(im_mags)
    
    return images, im_obs, sources, im_mags


""" #> POPULATION ====================
================================== """
###### GAL=CPU OR GPU, QUAD=CPU ######

#> main gal/quad population fn
def genPop(numGals, **kwargs):

    #> declarations
    method = kwargs.get('method', 'deflect')        # method for calculating deflection angles
    numGals = int(numGals)                          # number of galaxies to generate
    numSource_gal = kwargs.get('numSource_gal', 1)  # number of sources per galaxy
    
    #> setting random seed
    seed = kwargs.get('seed', None)
    np.random.seed(seed)

    #>>># unpacking kwargs
    
    #> output kwargs
    saveFlag = kwargs.get('saveFlag', getSaveDict())                       # saving info
    timeFlag = kwargs.get('timeFlag', False)                               # printing time info
    plotFlag = kwargs.get('plotFlag', getPlotDict(atleast_one_plot=False)) # flag of wether to plot or not
    parent_dir = kwargs.get('parent_dir', '')                              # parent dir where data folder is
    folder   = kwargs.get('folder', '+unsorted')                           # data output folder
    suffix   = kwargs.get('suffix', '')                                    # suffix to file name
    if folder is None: folder = '+unsorted'                                # fail safe for default output dir
    
    #> galaxy profiles kwargs
    nfw  = kwargs.get('nfw', True)                # if want nfw profile
    hern = kwargs.get('hern', True)               # if want hernquist profile
    mult = kwargs.get('mult', [])                 # multipoles requested to be in the galaxies
    ex   = kwargs.get('ex', False)                # if want external shear
    galProfs = kwargs.get('galProfs', galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)) # default galaxy profiles
    kwargs |= {'galProfs': galProfs}
    
    #> checking if requesting, but not saving (or vice-versa)
    delays = kwargs.get('delays', saveFlag['delays']) # if wanting time delays
    mags = kwargs.get('mags', saveFlag['im_mags'])# if wanting image magnifications
    observables = kwargs.get('observables', None) # lensing observables
    if mags and not saveFlag['im_mags']:
        error.highlight('Requesting mags without requesting to save them. Changing saveFlag[im_mags]=True')
        saveFlag['im_mags'] = True
    if observables is None and saveFlag['im_obs']:
        error.highlight('Requesting observables without supply which ones. Changing saveFlag[im_obs]=False')
        saveFlag['im_obs'] = False
    if delays and method != 'lpot':
        error.highlight(f'Requesting time delays with method={method}, changing to method=lpot')
        method = 'lpot'
    if delays and not saveFlag['delays']:
        error.highlight('Requesting time delays without requesting to save them. Changing saveFlag[delays]=True')
        saveFlag['delays'] = True
    kwargs |= {'saveFlag': saveFlag, 'method': method}
    
    #<<<# finished unpacking kwargs
    
    #> file information
    if folder != '' and folder[-1] != '/': folder += '/' # formatting, if necessary
    if parent_dir != '' and parent_dir[-1] != '/': parent_dir += '/' # formatting, if necessary
    dirPath = parent_dir + dataDir + folder                           # main outFile path
    dt = str(int( (datetime.now().second*1e3 + datetime.now().microsecond/1e4) / 10 )).zfill(4) # second + millisecond info
    fileName = time.strftime('%y%m%d%H%M', time.localtime())
    fileName += dt + suffix                              # file date information 
    kwargs |= {'fileName': fileName, 'dirPath': dirPath} # adding file information
    
    #> print statement
    galProfs_print = []
    for key in galProfs.keys():
        if galProfs[key]: 
            if isinstance(galProfs[key], list):
                if len(galProfs[key]) > 0: 
                    galProfs_print.extend([ 'm'+str(x) for x in galProfs[key] ])
                    continue
            galProfs_print.append(key)
    print(f'> Starting generate for {fileName}:')
    print(f'>   numGals = {numGals} \n>   numSource_gal = {numSource_gal}')
    print('>   Galaxy components are:', ', '.join(galProfs_print))
    print()
    
    #> generating galaxies (this should cover all possible arguments)
    srt = time.time()
    lenses, bprofiles = genGalPop(numGals, **kwargs)
    if timeFlag: print(f'> Generating galaxies took {time.time()-srt:.2f} s')
    
    #<&># object information
    #> bprofiles = [list]
    #> lenses = (xgrid, ygrid, delx, dely, lamt)
    #> xgrid  = (nph*2, nph*2)          [np.ndarray]
    #> ygrid  = (nph*2, nph*2)          [np.ndarray]
    #> delx   = (numGals, nph*2, nph*2) [np.ndarray]
    #> dely   = (numGals, nph*2, nph*2) [np.ndarray]
    #> lamt   = (numGals, nph*2, nph*2) [np.ndarray]
    
    #> updating bprofiles
    kwargs |= {'bprofiles': bprofiles}
    
    #> generating quads
    srt = time.time()
    quadObject = genQuadPop(lenses=lenses, **kwargs)
    if timeFlag: print(f'> Generating quads took {time.time()-srt:.2f} s')
    
    #> unpacking quad object
    images, im_obs, sources, im_mags, delays = quadObject
    
    #> creating pandas dataframe for obs
    #if observables is not None:
    #    im_obs = pd.DataFrame(im_obs, columns=observables)
    
    #<&># object information
    #> quadObject = images, im_obs, sources, im_mags, delays
    #> images  = (nph*2, nph*2)          [np.ndarray]
    #> im_obs  = (nph*2, nph*2)          [np.ndarray]
    #> sources = (numGals, nph*2, nph*2) [np.ndarray]
    #> im_mags = (numGals, nph*2, nph*2) [np.ndarray]
    #> delays  = (numGals*numSource_gal, jims) [np.ndarray]; days, lpot only
    
    #> saving data
    if any(saveFlag.values()):
        
        #> getting lens information (only stores information of the ONE lens)
        if saveFlag['lens']:
            lens_num = 0 # gets maps of this lens
            lens = np.array([lenses[0],lenses[1],             # xgrid, ygrid
                             lenses[2][lens_num],             # delx
                             lenses[3][lens_num],             # dely
                             lenses[4][lens_num]])            # lamt
        else: lens = None
        
        #> checking other info
        if not saveFlag['images']: images = None
        if not saveFlag['im_obs']: im_obs = None
        if not saveFlag['im_mags']: im_mags = None
        if not saveFlag['delays']: delays = None
        if not saveFlag['sources']: sources = None
        if not saveFlag['bprofiles']: bprofiles = None
        if not saveFlag['paramRanges']: paramRanges = None
        else: paramRanges = kwargs.get('paramRanges', None)
        
        #> saving
        saveInfo_multipleFiles(dirPath, fileName, suffix, # file/path naming
                               images=images,             # image positions
                               im_obs=im_obs,             # image observables
                               im_mags=im_mags,           # image magnifications
                               delays=delays,             # time delays (days)
                               sources=sources,           # source positions
                               lens=lens,                 # lens info (see above)
                               bprofiles=bprofiles,       # bprofiles (inclues pix_arc)
                               paramRanges=paramRanges,   # param ranges
                               csv=saveFlag['csv'])                   # converts bprofiles to csv
    
    #> plotting many lenses
    if any(plotFlag.values()):
        
        #> declarations
        maxPlot = 20
        
        #> unpacking info
        if method == 'deflect':
            xgrid, ygrid = lenses[:2]
            all_delx, all_dely, all_lamt = lenses[2:]
        elif method == 'lpot':
            xgrid, ygrid = lenses[:2]
            all_lpot = lenses[2]
            
        
        #> plotting!
        for i, _ in zip(range(numGals), range(maxPlot)):
            
            #> getting images
            ims = np.array([images[i*numSource_gal]])
            pix_arc = bprofiles[i]['pix_arc']
            
            #> parsing params depending on the method
            if method == 'deflect':
                delx = all_delx[i]
                dely = all_dely[i]
            elif method == 'lpot':
                lpot = all_lpot[i]
                dely, delx = np.gradient(lpot, axis=(0, 1), edge_order=2)
                delx *= pix_arc * u.arc_rad                                  # per pixel -> rad
                dely *= pix_arc * u.arc_rad
            
            #> kappa
            if plotFlag['kappa']:
                plot.kappa(xgrid, ygrid, delx, dely, pix_arc, images=ims)
                
            #> deflection angles
            if plotFlag['deflect']:
                plot.deflect(xgrid, ygrid, delx, dely, images=ims)
                
            #> critical curves
            if plotFlag['ccurves']:
                plot.ccurves(xgrid, ygrid, delx, dely, pix_arc=pix_arc, images=ims)
            
            #> caustics
            if plotFlag['caustics']:
                plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=ims)
            
            #> caustic zoom
            if plotFlag['caustics_zoom']: 
                zoom = plotFlag['caustics_zoom']    # getting zoom
                if isinstance(zoom, bool): zoom = 5 # if zoom=True, sets default zoom value
                plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=ims, zoom=int(zoom))
                
    #> final print
    print(f'> Done generating {fileName}!')
    
    return dirPath + fileName + suffix


#> generates a population of galaxies
def genGalPop(numGals, **kwargs):
    
    #> imports
    import params
    
    #> kwargs
    gpu = kwargs.get('gpu', False)
    bprofiles = kwargs.get('bprofiles', None)
    method = kwargs.get('method', 'deflect')
    
    #> creating bprofiles
    if bprofiles is None:
        
        #> getting precon paramRanges if given
        galProfiles_ = kwargs.get('galProfiles', galProfiles())
        paramRanges = kwargs.get('paramRanges', None)
        if paramRanges is None: 
            paramRanges = params.toggleParams(galProfiles_)
        
        #> creating bprofiles
        bprofiles = params.bprofiles(numGals, paramRanges, 
                                     **kwargs)
    
    #> grid declarations
    nph = kwargs.get('nph', u.nph)
    xgrid, ygrid = grid(nph=nph)
    
    #> deflection angles
    if method == 'deflect':
        
        #> imports
        if gpu: import deflectionGPU as deflection
        else: import deflectionCPU2 as deflection
        
        #> calculating deflection angles
        delx, dely, lamt = deflection.deflection(xgrid, ygrid, bprofiles)
        
        return (xgrid, ygrid, delx, dely, lamt), bprofiles
        
    #> lensing potential
    elif method == 'lpot':
        
        #> imports
        if gpu: import lpotGPU as lpot
        else: import lpotCPU as lpot
        
        #> calculating lens potential
        lpot = lpot.potential(xgrid, ygrid, bprofiles)
        
        return (xgrid, ygrid, lpot), bprofiles

    #> cannot parse method
    else: error.phrase(f'Method={method} not understood. Please use lpot or deflect')
    return 


#> generates a population of quads
def genQuadPop(lenses, bprofiles, **kwargs):
    
    #> imports
    import lensing
    
    #> setting random seed
    seed = kwargs.get('seed', None)
    np.random.seed(seed)
    random.seed(seed)
    
    #> method of lensing
    method = kwargs.get('method', 'deflect')
    
    #> if wanting print statements
    verbose = kwargs.get('verbose', False)
    warnings = kwargs.get('warnings', True)
    
    #> images + source declarations
    requested_jims = kwargs.get('jims', None)
    numSource_gal = kwargs.get('numSource_gal', 1)
    observables = kwargs.get('observables', None)
    mags = kwargs.get('mags', False)
    source = kwargs.get('source', kwargs.get('sources', None))
    resolution = kwargs.get('resolution', None)

    #> if wanting observables w/o specifiying jims
    quad_observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    if observables is not None and requested_jims is None:
        if any(x in observables for x in quad_observables):
            error.highlight('Requesting quad observables w/o setting jims=5!')
            error.highlight('Setting jims=5!')
            requested_jims = 5
        
    #> unpacking
    xgrid, ygrid = lenses[:2]
    nph = int(len(xgrid)/2)
    arc_rad = u.arc_rad
    
    #> unpacking lenses depending on the method
    if method == 'deflect':
        all_delx, all_dely, all_lamt = lenses[2:]
        numGals = len(all_delx)
    elif method == 'lpot':
        all_lpot = lenses[2]            # (numGals, nph*2, nph*2) [rad^2]
        numGals = len(all_lpot)
    else: error.phrase(f'Method={method} not lpot or deflect')
    
    #> iterating thru each galaxy
    images, im_obs, im_mags, delays, sources = [], [], [], [], []
    for numGal in range(numGals):
        
        #> print statement
        if verbose: print(f'> Lensing {numSource_gal} source(s) for galaxy {numGal+1}!')
        
        #> getting pix_arc
        pix_arc = bprofiles[numGal]['pix_arc']
        
        #> method specific parsing
        if method == 'deflect':
            delx = all_delx[numGal]
            dely = all_dely[numGal]
            lamt = all_lamt[numGal]
        elif method == 'lpot':
            lpot = all_lpot[numGal]                                   # (Ny, Nx), rad^2
            dely, delx = np.gradient(lpot, axis=(0, 1), edge_order=2)
            delx *= pix_arc * arc_rad                                  # per pixel -> rad
            dely *= pix_arc * arc_rad
            lamt = None
        
        #> getting caustic / source positions
        source_apexes = lensing.caustic(xgrid, ygrid, delx, dely, pix_arc, lamt, warnings=warnings)
        
        #> error collecting
        if len(source_apexes) == 0:
            print(numGal)
            print(source_apexes)
            print(lensing.ranSources2(source_apexes))
            plot.ccurves(xgrid, ygrid, delx, dely, pix_arc)
            plot.causticBox(xgrid, ygrid, delx, dely, pix_arc, source_apexes)
                    
        #> collecting all requested images
        for i in range(numSource_gal):
            
            #> enforcing requested num of jims
            ims, numTries, flag = [], maxTries, True
            while (len(ims) != requested_jims) and (numTries > 0) and flag:
                
                #> getting source positions
                if source is None: #> getting random source
                
                    #> random source within caustic
                    if requested_jims == 5:
                        xs, ys = lensing.ranSources2(source_apexes)[0]
                    else:
                        #> random source within window
                        xs, ys = random.uniform(-nph, nph) / pix_arc, random.uniform(-nph, nph) / pix_arc
                        
                else: # if source positions are provided
                    
                    #> converting just in case
                    if isinstance(source[0], float) or isinstance(source[0], int): source =[source]
                    source = np.array(source)
                    
                    #> checking for errors
                    if (len(source) != numSource_gal) and (len(source) != 1) and (len(source) != numSource_gal*numGals):
                        error.phrase(f'Incorrect # of sources: #s={len(source)}, #gal={numGals}, #s_gal={numSource_gal}')
                        
                    #> checking to see how to unpacking sources
                    if len(source) == 1:                       # only one source given
                        xs, ys = source[0]
                    elif len(source) == numSource_gal:         # one set of matching sources per each galaxy
                        xs, ys = source[i]
                    elif len(source) == numSource_gal*numGals: # one source per source per galaxy
                        xs, ys = source[(numGal*numSource_gal)+i]
                
                #> converting units
                xs /= u.arc_rad
                ys /= u.arc_rad
            
                #> lensing! (same for both methods)
                ims = lensing.lfims(xgrid / pix_arc / u.arc_rad,  # radians
                                    ygrid / pix_arc / u.arc_rad,  # radians
                                    delx, dely,                   # radians
                                    xs, ys,                       # radians 
                                    pix_arc, nph, 
                                    ordered=(method=='deflect'))  # geometric order only if deflect
                
                #> adjusting tries
                numTries-=1
                
                #> seeing if resolved (needs at least 2 images to compare)
                if resolution is not None and len(ims) > 1:
                    resolved = lensing.resolved(images=ims, resolution=resolution)
                    if not resolved:
                        ims = np.empty((0,2)) # no images [arcsec]
                
                #> if there is no requested number of jims or requesting specific source
                if requested_jims is None or source is not None: 
                    # print('changing flag')
                    flag = False
            
            #> if requested jims not found, filling w/ nans (keeps arrays aligned w/ supplied sources)
            if requested_jims is not None and len(ims) != requested_jims:
                error.highlight(f'Source {i} of galaxy {numGal} did not give {requested_jims} (resolved) images, gave {len(ims)} instead; filling w/ nans')
                images.append(np.full((requested_jims, 2), np.nan))                   # arcsec
                sources.append([xs * u.arc_rad, ys * u.arc_rad])                      # arcsec
                if method == 'lpot': delays.append(np.full(requested_jims, np.nan))   # days
                if observables is not None: im_obs.append(np.full(len(observables), np.nan))
                if mags: im_mags.append(np.full(requested_jims, np.nan))
                continue
            
            #> ordering by fermat potential + time delays (lpot only)
            if method == 'lpot' and len(ims) > 0:
                ims, tau = lensing.fermat(ims, lpot, xs, ys, pix_arc, nph) # arcsec, rad^2
                delays.append(getDelays(tau, bprofiles[numGal]))           # days
                    
            #> if wanting lensing observables
            if observables is not None and requested_jims == 5:
                if len(ims) != 5: # if NOT QUAD! UH OH, STINKY
                    print(ims, numTries, flag)
                    plot.ccurves(xgrid, ygrid, delx, dely, pix_arc)
                    plot.causticBox(xgrid, ygrid, delx, dely, pix_arc, source_apexes, xs=xs, ys=ys)
                obs = geometry.lensObs((0, 0), ims, observables)
                im_obs.append(obs)
             
            #> if wanting magnifications
            if mags: im_mags.append(getMag((xgrid, ygrid, delx, dely), pix_arc, ims))
            
            #> adding images
            images.append(ims)
            sources.append([xs * u.arc_rad, ys * u.arc_rad])

    #> creating space in prints
    if verbose: print()
    
    #> to numpy
    images = np.array(images, dtype=object)
    sources = np.array(sources, dtype=float).reshape(numGals*numSource_gal, 2)
    im_obs = np.array(im_obs, dtype=object)
    im_mags = np.array(im_mags, dtype=object)
    delays = np.array(delays, dtype=object)

    return images, im_obs, sources, im_mags, delays


""" #> MAGNIFICATION =================
================================== """

#> gets mags for list of lenses
def getMags(lenses, bprofiles, images, **kwargs):
    
    #> imports
    import lensing
    
    #> declarations
    mags = []
    
    #> unpacking
    gridx, gridy, delx, dely, gamma = lenses
    
    for gradx, grady, profile, ims in zip(delx, dely, bprofiles, images):
        im_mags = lensing.magnification(gridx, gridy, gradx, grady, profile['pix_arc'], ims)
        mags.append(im_mags)
        
    return mags


#> gets mags for list of lenses
def getMag(lens, pix_arc, images):
    import lensing
    return lensing.magnification(*lens, pix_arc, images)


""" #> TIME DELAYS ===================
================================== """

#> returns the time delays relative to the first arriving image
def getDelays(tau, bprof):
    
    #> declarations
    zl      = bprof['zl']      # lens redshift
    angDist = bprof['angDist'] # kpc
    sigCrit = bprof['sigCrit'] # solMass/kpc^2
    
    #> time-delay distance; D_dt = (1+zl) Dd Ds/Dds = (1+zl) 4 pi G Dd^2 sigCrit / c^2
    D_dt = (1+zl) * 4*np.pi*u.G_kpc_solMass * angDist**2 * sigCrit / u.c_kpc**2 # kpc
    
    #> time delays
    dt = D_dt / u.c_kpc * (tau - tau[0]) # s
    
    return dt / 86400 # days


""" #> SAVING ========================
================================== """

#> saves info to outfile
def saveInfo_multipleFiles(dirPath, fileName, suffix, **kwargs):
    
    #> globals
    global maxTries
    tries = maxTries
    
    #> imports
    from pathlib import Path
    
    #> adding / if needed
    if dirPath[-1] != '/': dirPath += '/'
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
    
    #> unpacking kwargs
    images    = kwargs.get('images', None)
    im_obs    = kwargs.get('im_obs', None)
    im_mags   = kwargs.get('im_mags', None)
    delays    = kwargs.get('delays', None)
    sources   = kwargs.get('sources', None)
    lens      = kwargs.get('lens', None)
    bprofiles = kwargs.get('bprofiles', None)
    paramRanges = kwargs.get('paramRanges', None)
    csv = kwargs.get('csv', None)
    
    #> saving data
    if images    is not None and images.size!=0 :  np.save(parent_dir + fileName[:-1] + '_images', images)
    if im_obs    is not None and im_obs.size!=0 :  np.save(parent_dir + fileName[:-1] + '_obs', im_obs)
    if im_mags   is not None and im_mags.size!=0 : np.save(parent_dir + fileName[:-1] + '_mags', im_mags)
    if delays    is not None and delays.size!=0 :  np.save(parent_dir + fileName[:-1] + '_delays', delays)
    if sources   is not None and sources.size!=0 : np.save(parent_dir + fileName[:-1] + '_srcs', sources)
    if lens      is not None: np.save(parent_dir + fileName[:-1] + '_lens', lens)
    if bprofiles is not None: np.save(parent_dir + fileName[:-1] + '_bprofiles', bprofiles)
    if paramRanges is not None: np.save(parent_dir + fileName[:-1] + '_paramRanges', paramRanges)
    if csv: parse.bprof_to_df(bprofiles, save=True, outFile=parent_dir + fileName[:-1]+'_df.csv')

    return


""" #> NEWT ==========================
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
    if rng is None: rng = np.random
    
    #> declarations
    pix_arc = bprof['pix_arc']
    
    
    #>>># CREATING GALAXY #<<<#
    
    #> lens-plane grid (theta)
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc
    xg, yg = np.meshgrid(w, w)
    
    #> deflection angles + beta
    ax, ay   = profiles.alpha_points(xg, yg, bprof)    # arcsec
    bxg, byg = xg - ax, yg - ay               # beta_i = theta_i - alpha_i [arcsec]
    
    
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
            out = {'images': np.full((numSource, 5, 2), np.nan), 'mu':   np.full((numSource, 5), np.nan),
                   'srcs':   np.full((numSource, 2), np.nan),    'nims': np.zeros(numSource, dtype=int), 'flag': np.zeros(numSource, dtype=bool)}
            out['delays'] = np.full((numSource, 5), np.nan)                     # days
            if observables is not None: out['obs'] = np.full((numSource, len(observables)), np.nan)
            error.highlight(f'Caustic does not exist! bprof={bprof}')
            return out
        
        #> drawing sources inside the caustic
        srcs = newtlens.sample_sources(caus, numSource, rng=rng)
        
        #> lensing!
        res = newtlens.find_images(xg, yg, bxg, byg, srcs, bprof)
        
        #> redrawing non-quads
        nEven = 0                                       # redraws caused by an even image count (missed image)
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
            for i in ['images', 'mu']: res[i], new_res[i] = newtlens.pad(res[i], w), newtlens.pad(new_res[i], w)
            for i in res: res[i][bad] = new_res[i]
        
        #> printing even-count failures
        if nEven > 0: print(f'> {nEven} redraw(s) were even-count failures (missed images)')
            
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
    out['flag']         = flag                                 # True if > jmax images

    #> fermat orders all images of sources w/ more than jmax images (not passed to morph_order)
    if method == 'lpot' and flag.any():
        out['images'][flag], out['mu'][flag], f_tau = order.fermat_order(ims[flag], mu[flag], srcs[flag], bprof)
        out['delays'][flag] = np.where(np.isinf(f_tau), np.nan, D_dt / u.c_kpc * (f_tau - f_tau[:,:1]) / 86400) # days
    
    #> observables
    if observables is not None:
        out['obs'] = np.full((len(srcs), len(observables)), np.nan)
        out['obs'][quad] = geometry.quad_obs(o_ims, observables)
    
    return out


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    

    # end
# thank

#> old code that might be useful later

#> parallelization declarations
# numCores = 48
# gpuMemory = 64 # gb
# print(tobytes(gpuMemory, 'g'))
# numProcesses =  min(numCores, int( tobytes(gpuMemory, 'g') / ( numGals * 80000 * 57.5 ) ))
# print(numProcesses)