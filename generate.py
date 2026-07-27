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

#> terminal imports
import argparse

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
                'sources': kwargs.get('sources', True),
                'lens': kwargs.get('lens', False), 
                'bprofiles': kwargs.get('bprofiles', True)}
    
    #> making all false if wanted
    if not atleast_one_save:
        saveDict = dict.fromkeys(saveDict, False)
        print(saveDict)
    
    return copy.copy(saveDict)


""" #> MEMORY CALC ===================
================================== """

#> converts memory units
def bytesto(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])

#> converts memory units
def tobytes(bytes, from_, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** -a[from_])


""" #> GRID ==========================
================================== """

#> returns grid
def grid(**kwargs):
    nph = kwargs.get('nph', u.nph)
    w_range = np.arange(-nph, nph, 1, dtype=float)
    return np.meshgrid(w_range, w_range)


""" #> REDSHIFT ======================
================================== """

#> returns random redshifts
def ranRedshifts(numGals, **kwargs):
    
    ### THESE SHOULD BE INFORMED FROM OBSERVATIONS/ THEORY
    ### I CAN DO A FIRST PASS LOOK W/ EINSTEIN RADIUS...? 
    ### OR LENSING PROBABILITY
    
    ### should be a joint probability distribution
    
    #> imports
    from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN    
    
    #> correlation matrix
    cross_corl = 0.4
    R = ( np.identity(n=2) * (1-cross_corl) ) + cross_corl
    
    #> deflector: lb, ub, mu, sigma
    ub_zl    = kwargs.get('ub_zl', 1.5)
    lb_zl    = kwargs.get('lb_zl', 0.1)
    mu_zl    = kwargs.get('mu_zl', 0.5)
    sigma_zl = kwargs.get('mu_zl', 0.1)
    
    #> source: lb, ub, mu, sigma
    ub_zs    = kwargs.get('ub_zs', 3.0)
    lb_zs    = kwargs.get('lb_zs', 0.5)
    mu_zs    = kwargs.get('mu_zs', 1.0)
    sigma_zs = kwargs.get('mu_zs', 0.1)
    
    #> constructing params
    lb = np.array([lb_zl, lb_zs])
    ub = np.array([ub_zl, ub_zs])
    mu = np.array([mu_zl, mu_zs])
    sigma = np.array([sigma_zl, sigma_zs])
    cov = np.diag(sigma) @ R @ np.diag(sigma) # converting correlation & std vectors to covariance matrix
    
    #> drawing samples
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    samples = tmvn.sample(numGals)
    samples = np.array(samples)
    
    return samples.T


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
def genGal(redshifts, galProfiles, **kwargs):
    
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
def genQuad(lens, zs, **kwargs):
    
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
    
    ##### ONE MORE KWARGS = SOURCE
    
    #> declarations
    numGals = int(numGals)                          # number of galaxies to generate
    numSource_gal = kwargs.get('numSource_gal', 1)  # number of sources per galaxy

    #>>># unpacking kwargs
    
    #> output kwargs
    verbose  = kwargs.get('verbose', False)                                # printing important 
    saveFlag = kwargs.get('saveFlag', getSaveDict())                           # saving info
    timeFlag = kwargs.get('timeFlag', False)                                   # printing time info
    plotFlag = kwargs.get('plotFlag', getPlotDict(atleast_one_plot=False))     # flag of wether to plot or not
    folder   = kwargs.get('folder', '+unsorted')                           # data output folder
    suffix   = kwargs.get('suffix', '')                                    # suffix to file name
    if folder is None: folder = '+unsorted'                                # fail safe for default output dir
    
    #> grid kwargs
    scale_factor = kwargs.get('scale_factor', 1)                           # a way to increase (or decrease) the grid resolution
    pix_arc = [kwargs.get('pix_arc', u.pix_arc) * scale_factor] * numGals  # pixel per arcsec conversion
    nph = kwargs.get('nph', u.nph)              * scale_factor             # width / 2 of grid
    
    #> galaxy properties kwargs
    nfw  = kwargs.get('nfw', True)                # if want nfw profile
    hern = kwargs.get('hern', True)               # if want hernquist profile
    mult = kwargs.get('mult', [])                 # multipoles requested to be in the galaxies
    ex   = kwargs.get('ex', False)                # if want external shear
    galProfs = kwargs.get('galProfs', galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)) # default galaxy profiles
    paramRanges = kwargs.get('paramRanges', None) # the paramRanges from params, list of all parmas for each profile
    redshifts = kwargs.get('redshifts', ranRedshifts(numGals))                  # randomized default redshifts (might result in NCGs)
    if isinstance(redshifts, tuple): redshifts = np.array([redshifts] * numGals)# converting redshifts if tuple, i.e., supplied as (zl, zs)
    uniform = kwargs.get('uniform', False)        # sampling galaxy params uniformly instead of Gaussian
    lb  = kwargs.get('lb', None)                  # lower bounds for galaxy params (i.e., bprofiles)
    ub  = kwargs.get('ub', None)                  # upper bounds for galaxy params
    mu  = kwargs.get('mu', None)                  # mu vector for galaxy params
    cov = kwargs.get('cov', None)                 # covariance matrix for galaxy params]
    
    #> image properties kwargs
    jims = kwargs.get('jims', None)               # number of images wanted from lens
    mags = kwargs.get('mags', saveFlag['im_mags'])# if wanting image magnifications
    observables = kwargs.get('observables', None) # lensing observables
    
    #> checking if requesting, but not saving (or vice-versa)
    if mags and not saveFlag['im_mags']:
        error.highlight('Requesting mags without requesting to save them. Changing saveFlag[im_mags]=True')
        saveFlag['im_mags'] = True
    if observables is None and saveFlag['im_obs']:
        error.highlight('Requesting observables without supply which ones. Changing saveFlag[im_obs]=False')
        saveFlag['im_obs'] = False
    
    #> computation kwargs
    gpu = kwargs.get('gpu', False) # if should use GPU (or CPU) [currently broken]
    
    #> converting to np arrays
    to_convert = [lb, ub, mu, cov]
    for i, var in enumerate(to_convert):
        if var is not None:
            to_convert[i] = np.asarray(var)
    lb, ub, mu, cov = to_convert
    
    #<<<# finished unpacking kwargs
    
    #> file information
    if folder != '' and folder[-1] != '/': folder += '/' # formatting, if necessary
    dirPath = dataDir + folder                           # main outFile path
    dt = str(int( (datetime.now().second*1e3 + datetime.now().microsecond/1e4) / 10 )).zfill(4) # second + millisecond info
    fileName = time.strftime('%y%m%d%H%M', time.localtime())
    fileName += dt + suffix                              # file date information 
    
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
    lenses, bprofiles = genGalPop(redshifts,               # (zl, zs) redshifts      [Nx2 np.array]
                                  galProfs,                # profiles to be included [dic]
                                  paramRanges=paramRanges, # pre-con paramRanges     [dic]
                                  nph=nph,                 # width / 2 for grid      [int]
                                  pix_arc=pix_arc,         # pixel / arcsec convers. [list]
                                  uniform=uniform,         # how to sample params    [bool]
                                  lb=lb,                   # lower bounds vector     [None or Nx1 np.array]
                                  ub=ub,                   # upper bounds vector     [None or Nx1 np.array]
                                  mu=mu,                   # mu vector               [None or Nx1 np.array]
                                  cov=cov,                 # covariance matrix       [None or NxN np.array]
                                  gpu=gpu,                 # if using GPU            [bool]
                                  verbose=verbose)         # if want print info      [bool]
    if timeFlag: print(f'> Generating galaxies took {time.time()-srt:.2f} s')
    
    #<&># object information
    #> bprofiles = [list]
    #> lenses = (xgrid, ygrid, delx, dely, lamt)
    #> xgrid  = (nph*2, nph*2)          [np.ndarray]
    #> ygrid  = (nph*2, nph*2)          [np.ndarray]
    #> delx   = (numGals, nph*2, nph*2) [np.ndarray]
    #> dely   = (numGals, nph*2, nph*2) [np.ndarray]
    #> lamt   = (numGals, nph*2, nph*2) [np.ndarray]
    
    #> generating quads
    srt = time.time()
    quadObject = genQuadPop(lenses,                      # lenses object           [list]
                            zs=redshifts[:,1],           # source redshifts        [Nx1 np.ndarray]
                            pix_arc=pix_arc,             # pixel / arcsec convers. [list]
                            numSource_gal=numSource_gal, # num sources per gal     [int]
                            observables=observables,     # lensing obesrvables     [None or list]
                            verbose=verbose,             # if want print info      [bool]
                            mags=mags,                   # if want magnifications  [bool]
                            jims=jims)                   # num ims per source      [int]
    if timeFlag: print(f'> Generating quads took {time.time()-srt:.2f} s')
    
    #> unpacking quad object
    images, im_obs, sources, im_mags = quadObject
    
    #<&># object information
    #> quadObject = images, im_obs, sources, im_mags
    #> images  = (nph*2, nph*2)          [np.ndarray]
    #> im_obs  = (nph*2, nph*2)          [np.ndarray]
    #> sources = (numGals, nph*2, nph*2) [np.ndarray]
    #> im_mags = (numGals, nph*2, nph*2) [np.ndarray]
    
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
        if not saveFlag['sources']: sources = None
        if not saveFlag['bprofiles']: bprofiles = None
        
        #> saving
        saveInfo_multipleFiles(dirPath, fileName, suffix, # file/path naming
                               images=images,             # image positions
                               im_obs=im_obs,             # image observables
                               im_mags=im_mags,           # image magnifications
                               sources=sources,           # source positions
                               lens=lens,                 # lens info (see above)
                               bprofiles=bprofiles)       # bprofiles (inclues pix_arc)
    
    #> plotting many lenses
    if any(plotFlag.values()):
        
        #> declarations
        maxPlot = 20
        
        #> unpacking info
        all_pix_arc = pix_arc
        xgrid, ygrid = lenses[:2]
        all_delx, all_dely, all_lamt = lenses[2:]
        
        #> plotting!
        for i, delx, dely, lamt, pix_arc, _ in zip(range(numGals), all_delx, all_dely, all_lamt, all_pix_arc, range(maxPlot)):
            
            #> getting images
            ims = np.array([images[i]])
            
            #> kappa
            if plotFlag['kappa']:
                plot.kappa(xgrid, ygrid, delx, dely, pix_arc, images=ims)
                
            #> deflection angles
            if plotFlag['deflect']:
                plot.deflect(xgrid, ygrid, delx, dely, images=ims)
            
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
def genGalPop(redshifts, galProfiles, **kwargs):
    
    #> imports
    import params
    
    #> unpacking
    numGals = len(redshifts)
    gpu = kwargs.get('gpu', False)
    pix_arc = kwargs.get('pix_arc', [u.pix_arc] * numGals)
    zl = redshifts[:,0]
    zs = redshifts[:,1]
    
    #> if on CPU or GPU
    if gpu: import deflectionGPU as deflection
    else: import deflectionCPU2 as deflection
    
    #> creating bprofiles
    bprofiles = kwargs.get('bprofiles', None)
    if bprofiles is None:
        
        #> getting precon paramRanges if given
        paramRanges = kwargs.get('paramRanges', None)
        if paramRanges is None: 
            paramRanges = params.toggleParams(galProfiles)
        
        #> creating bprofiles
        bprofiles = params.bprofiles(numGals, paramRanges, 
                                     zl=zl, zs=zs, pix_arc=pix_arc,
                                     verbose=kwargs.get('verbose', False), 
                                     cov=kwargs.get('cov', None), 
                                     mu=kwargs.get('mu', None),
                                     ub=kwargs.get('ub', None),
                                     lb=kwargs.get('lb', None),
                                     uniform=kwargs.get('uniform', False))
        
    #> grid declarations
    nph = kwargs.get('nph', u.nph)
    xgrid, ygrid = grid(nph=nph)
    
    #> calculating deflection angles
    delx, dely, lamt = deflection.deflection(xgrid, ygrid, bprofiles)
    
    return (xgrid, ygrid, delx, dely, lamt), bprofiles


#> generates a population of quads
def genQuadPop(lenses, zs, **kwargs):
    
    #> imports
    import lensing
    
    #> if wanting print statements
    verbose = kwargs.get('verbose', False)
    
    #> rejecting sources
    if kwargs.get('source', None) is not None:
        error.phrase('Supplied source positions are not supported rn. Cry about it.')
    
    #> images + source declarations
    requested_jims = kwargs.get('jims', None)
    numSource_gal = kwargs.get('numSource_gal', 1)
    observables = kwargs.get('observables', None)
    mags = kwargs.get('mags', False)
    
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
    
    #> iterating through each lens
    images, im_obs, im_mags, sources = [], [], [], []
    all_delx, all_dely, all_lamt = lenses[2:]
    numGals = len(all_delx)
    all_pix_arc = kwargs.get('pix_arc', [u.pix_arc] * numGals)
    
    for numGal, delx, dely, lamt, pix_arc in zip(range(numGals), all_delx, all_dely, all_lamt, all_pix_arc):
        
        #> print statement
        if verbose: print(f'> Lensing {numSource_gal} source(s) for galaxy {numGal+1}!')
        
        #> getting caustic / source positionsimport lensing
        source_apexes = lensing.caustic(xgrid, ygrid, delx, dely, pix_arc, lamt)
        
        #> collecting all requested images
        for i in range(numSource_gal):
            
            #> enforcing requested num of jims
            ims, numTries, flag = [], maxTries, True
            while (len(ims) != requested_jims) and (numTries > 0) and flag:
                
                #> getting random source
                xs, ys = lensing.ranSources2(source_apexes)[0]
                
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
                    # print('changing flag')
                    flag = False
                    
            #> if wanting lensing observables
            if observables is not None and requested_jims == 5:
                if len(ims) != 5: # if NOT QUAD! UH OH, STINKY
                    print(ims, numTries, flag)
                    plot.causticBox(xgrid, ygrid, delx, dely, pix_arc, source_apexes, xs=xs, ys=ys)
                    plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=[ims], xs=xs, ys=ys)
                    plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=[ims], zoom=5, xs=xs, ys=ys)
                obs = geometry.lensObs((0, 0), ims, observables)
                im_obs.append(obs)
                
            #> if wanting magnifications
            if mags: im_mags.append(getMag((xgrid, ygrid, delx, dely), pix_arc, ims))
            
            #> adding images
            images.append(ims)
            sources.append([xs * u.arc_rad, ys * u.arc_rad])
            
            #> if could not find the number requested
            if numTries == 0:
                error.highlight(f'Source {i} could not find {requested_jims} images! Last source tried = ({xs},{ys})')
    
    #> creating space in prints
    if verbose: print()
    
    #> to numpy
    images = np.array(images, dtype=object)
    sources = np.array(sources)
    im_obs = np.array(im_obs)
    im_mags = np.array(im_mags)

    
    return images, im_obs, sources, im_mags


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


""" #> SAVING ========================
================================== """

#> saves info to outfile
def saveInfo_oneFile(outFile, **kwargs):
    
    #> unpacking kwargs
    images    = kwargs.get('images', None)
    im_obs    = kwargs.get('im_obs', None)
    im_mags   = kwargs.get('im_mags', None)
    sources   = kwargs.get('sources', None)
    lens      = kwargs.get('lens', None)
    bprofiles = kwargs.get('bprofiles', None)
    
    #> declaration
    mainDict = {}
    
    #> saving data
    if images    is not None and images.size!=0 :  mainDict['images'] = images
    if im_obs    is not None and im_obs.size!=0 :  mainDict['im_obs'] = im_obs
    if im_mags   is not None and im_mags.size!=0 : mainDict['im_mags'] = im_mags
    if sources   is not None and sources.size!=0 : mainDict['sources'] = sources
    if lens      is not None: mainDict['lens'] = lens
    if bprofiles is not None: mainDict['bprofiles'] = bprofiles
    
    #> ensuring not empty, then saving
    if mainDict: # not empty
        np.save(outFile, mainDict)
        print(f'> Saved info to {outFile}.npy')
    
    return


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
    sources   = kwargs.get('sources', None)
    lens      = kwargs.get('lens', None)
    bprofiles = kwargs.get('bprofiles', None)
    
    #> saving data
    if images    is not None and images.size!=0 :  np.save(parent_dir + fileName[:-1] + '_images', images)
    if im_obs    is not None and im_obs.size!=0 :  np.save(parent_dir + fileName[:-1] + '_obs', im_obs)
    if im_mags   is not None and im_mags.size!=0 : np.save(parent_dir + fileName[:-1] + '_mags', im_mags)
    if sources   is not None and sources.size!=0 : np.save(parent_dir + fileName[:-1] + '_srcs', sources)
    if lens      is not None: np.save(parent_dir + fileName[:-1] + '_lens', lens)
    if bprofiles is not None: np.save(parent_dir + fileName[:-1] + '_bprofiles', bprofiles)

    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    #> declarations
    printTime = False

    #> lens declarations
    zl = 0.5
    zs = 1.0
    factor = 1
    nph = 50 * factor
    pix_arc = 60 * factor
    
    #> observables
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23']
    
    #> parses command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mult', type=int, nargs='+', default=None, help='adds multipole (can be more than one, separate by space)')
    parser.add_argument('--ex', action='store_true', help='adds external shear')
    parser.add_argument('--suffix', type=str, default='', help='adds suffix to end of file name')
    parser.add_argument('--folder', type=str, default='+unsorted', help='adds suffix to end of file name')
    args, unknown = parser.parse_known_args()
    args.args = unknown
    # print(args)
    
    #> getting galaxy profiles
    if args.mult is None: mult=[]
    else: mult=args.mult
    galProfs = galProfiles(mult=mult, ex=args.ex)
    galProfs = galProfiles(nfw=True, hern=True, mult=[], ex=False)
    
    #> single galaxy
    # sysConfig = (zl, zs), galProfs, nph, pix_arc
    # singleGal(sysConfig) # options: plotDict(False), numSource_gal, source=(0.0, 0.0)
    genPop(numGals=10, redshifts=(0.5, 1.0), verbose=True, suffix='', mags=False, mult=[1,3,4], mu=[0,1])
    

    # end
# thank

#> old code that might be useful later

#> parallelization declarations
# numCores = 48
# gpuMemory = 64 # gb
# print(tobytes(gpuMemory, 'g'))
# numProcesses =  min(numCores, int( tobytes(gpuMemory, 'g') / ( numGals * 80000 * 57.5 ) ))
# print(numProcesses)