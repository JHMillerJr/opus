

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