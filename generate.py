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
    
    import copy

    #> default plotting dict
    plotDict = {'kappa': True, 'deflect': False, 'caustics': True, 'caustics_zoom': 1}
    
    #> making all false if wanted
    if not atleast_one_plot:
        plotDict = dict.fromkeys(plotDict, False)
    
    return copy(plotDict)


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
def ranRedshifts(num, **kwargs):
    
    ### THESE SHOULD BE INFORMED FROM OBSERVATIONS/ THEORY
    ### I CAN DO A FIRST PASS LOOK W/ EINSTEIN RADIUS...? 
    ### OR LENSING PROBABILITY
    
    #> deflector: lb and ub
    ub_zl = 1.5
    lb_zl = 0.1
    
    #> source: lb and ub
    ub_zs = 3.0
    lb_zs = 0.5
    
    #> params on deflector
    mu_zl = kwargs.get('mu_zl', 0.5)
    sigma_zl = kwargs.get('mu_zl', 0.1)
    
    #> params on source
    mu_zs = kwargs.get('mu_zs', 1.0)
    sigma_zs = kwargs.get('mu_zs', 0.1)
    
    return


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
    
    #> getting paramRanges
    bprofiles = kwargs.get('bprofiles', None)
    if bprofiles is None:
        paramRanges = params.toggleParams(galProfiles)
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
                    print('changing flag')
                    flag = False
                    
            #> if wanting lensing observables
            if observables is not None and requested_jims == 5:
                if len(ims) != 5: # if NOT QUAD! UH OH, STINKY
                    print(ims, numTries, flag)
                    plot.causticBox(xgrid, ygrid, delx, dely, pix_arc, source_apexes)
                    plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=[ims])
                    plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=[ims], zoom=5)
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
def saveInfo(outFile, **kwargs):
    
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


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
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
    args, unknown = parser.parse_known_args()
    args.args = unknown
    print(args)
    
    #> getting galaxy profiles
    if args.mult is None: mult=[]
    else: mult=args.mult
    galProfs = galProfiles(mult=mult, ex=args.ex)
    galProfs = galProfiles(nfw=True, hern=True, mult=[69], ex=False)
    
    #> single galaxy
    sysConfig = (zl, zs), galProfs, nph, pix_arc
    # singleGal(sysConfig) # options: plotDict(False), numSource_gal, source=(0.0, 0.0)
    
    #> generating galaxy populations
    gpu = False
    numGals = 10
    numSource_gal = 100
    redshifts = np.array([(zl, zs)] * numGals)
    pix_arc = [pix_arc] * numGals
    
    #> parallelization declarations
    # numCores = 48
    # gpuMemory = 64 # gb
    # print(tobytes(gpuMemory, 'g'))
    # numProcesses =  min(numCores, int( tobytes(gpuMemory, 'g') / ( numGals * 80000 * 57.5 ) ))
    # print(numProcesses)
    
    print(galProfs)
    
    #> getting offset params
    import cosmology
    lensplane_offset_kpc = 0.600 
    mu = np.array([0, 0])
    print(cosmology.angDist(0, zl))
    std = np.rad2deg(lensplane_offset_kpc / cosmology.angDist(0, zl) / 10**3) * 3600
    #cov = np.identity(len(mu)) * (std**2)
    lb, ub = np.array([-std, -std]), np.array([std, std])
    
    #> generating galaxies
    srt = time.time()
    lenses, bprofiles = genGalPop(redshifts, galProfs, 
                                  nph=nph, pix_arc=pix_arc, 
                                  uniform=True, lb=lb, ub=ub,
                                  gpu=gpu, verbose=True)#, mu=mu, cov=cov)
    if printTime: print(f'> Generating galaxies took {time.time()-srt:.2f} s')
    
    #> generating quads
    srt = time.time()
    images, im_obs, sources, im_mags = genQuadPop(lenses, zs, 
                                                  pix_arc=pix_arc, 
                                                  numSource_gal=numSource_gal,
                                                  observables=observables, 
                                                  verbose=False, mags=False, 
                                                  jims=5) # can include observables here (do not if you want magnifictions)
    if printTime: print(f'> Generating quads took {time.time()-srt:.2f} s')
    
    
    #> saving images
    # suffix = 'fsq_q0.85_std0.05_fit'
    suffix = args.suffix
    # suffix = ''
    outFile = dataDir+time.strftime('%y%m%d%H%M', time.localtime()) + suffix
    
    #> saving
    if True:
        lens_num = 0 # gets maps of this lens
        lens = np.array([lenses[0],lenses[1],  # xgrid, ygrid
                         lenses[2][lens_num],  # delx
                         lenses[3][lens_num],  # dely
                         lenses[4][lens_num]]) # lamt
        saveInfo(outFile, images=images,       # image positions
                          im_obs=im_obs,       # image observables
                          im_mags=im_mags,     # image magnifications
                          sources=sources,     # source positions
                          lens=lens,           # lens info (see above)
                          bprofiles=bprofiles) # bprofiles (inclues pix_arc)
    
    # geometry.d3d2(outFile+'.npy')
     
    #> loading to ensure it is okay
    # data, keys = parse.fromNPY(outFile)
    
    #> plotting many lenses
    if True:
        all_pix_arc = pix_arc
        xgrid, ygrid = lenses[:2]
        all_delx, all_dely, all_lamt = lenses[2:]
        for i, delx, dely, lamt, pix_arc in zip(range(numGals), all_delx, all_dely, all_lamt, all_pix_arc):
            ims = np.array([images[i]])
            plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=ims)
            # plot.caustics(xgrid, ygrid, delx, dely, pix_arc, images=ims, zoom=5)
    # print(images)

    # end
# thank