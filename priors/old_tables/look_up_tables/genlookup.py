#> name: gentemp.py
#> author: John Miller Jr
#> descrp: template file to generate galaxy/quad populations from generate.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__)) 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> GENERATE FN ===================
================================== """

#> generates a galaxy/ quad population based on the given params
def function():
    
    #> imports  (CHANGE IF GENERATE IS IN DIFF DIR)
    import params
    import generate
    from modules.units import u; u=u()
    
    #> declarations
    numGals       = 1                          # total # of galaxies to generate
    numSource_gal = 1                          # total # of sources per galaxy
    
    #> output kwargs
    folder   = '+unsorted'                     # dir in dataDir to save data
    suffix   = ''                              # suffix to add to file names
    verbose  = True                            # if wanting extra print info
    timeFlag = False                           # if wanting time info
    saveFlag = generate.getSaveDict(False)     # what to save (images=im_obs=sources=bprofiles=True, im_mags=lens=False)
    plotFlag = generate.getPlotDict(True, kappa=False)     # what to plot (kappa=caustics=True, deflect=caustics_zoom=False)
    
    #> grid kwargs
    scale_factor = 1                          # increase (or decrease) grid resolution [default=1]
    pix_arc      = 60 * scale_factor            # pixel per arcsec conversion [default=60]
    nph          = 50 * scale_factor            # width / 2 of grid [default=50)
    
    #> galaxy profiles kwargs (what profiles to add)
    nfw  = True                                # adds a nfw profile [default=True]
    hern = True                                # adds a hernquist profile [default=True]
    mult = []                                  # adds multipole profiles corresponding to the numbers given [default=[]]
    ex   = False                               # adds external shear [default=False]
    galProfs = generate.galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex) # dictionary handled by the code
    
    #> galaxy profile params kwargs (values of said profiles)
    lb  = None                                 # lower bounds vector of varied galaxy params [example=[0, 0]]
    ub  = None                                 # upper bounds vector ... [example=[2, 2]]
    mu  = None                                 # mean vector         ... [example=[1, 1]]
    cov = None                                 # covariance matrix   ... [example=np.identity(n=2)]
    uniform = False                            # if sampling from uniform dist, i.e., (lb, ub), False=TruncNorm
    
    #> priors
    hmf  = False                               # halo mass function
    cmr  = False                               # concentration-mass relation
    shmr = False                               # stellar-to-halo mass relation
    msr  = False                               # stellar mass-size relation
    priorDict = params.getPriorDict(hmf=hmf, cmr=cmr, shmr=shmr, msr=msr) # if wanting to draw from priors (hmf, cmr, shmr, msr)
    
    #> redshifts
    zl = 0.5                                   # redshift of lens
    zs = 1.0                                   # redshift of source
    # redshifts = (0.37374414, 1.18590195)
    redshifts = (zl, zs)                       # combined redshifts (for code)
    # redshifts = generate.ranRedshifts(numGals)
    print(redshifts)
    
    #> image properties kwargs
    jims = None                                # number of request images from each source (5=quad)
    mags = False                               # if wanting image magnifications (will change saveFlag automatically)
    observables = None # ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23'] # requested lensing observables
    
    #> bprofiles
    bprofiles = None
    

    #> imports
    import priors
    import lensing
    import modules.geometry as geometry
    
    #> getting paramRanges
    paramRanges = params.toggleParams(galProfs) # the parameter ranges and values, can be edited
    
    #> getting the bins
    lookUpBins = priors.look_up_bins(scale_factor=scale_factor)
    
    #>>># SETTING UP BPROFILES
    
    #> declaratoins
    h_shuntov = 0.7
    
    #> zl
    for izl, z_l_bin in enumerate(lookUpBins['zl']['grid']): 
        
        # if izl != 0: continue
        
        #> declarations
        bprofiles = []
        
        #> setting zl
        paramRanges['zl'] = z_l_bin
        
        print(z_l_bin)
        
        #> zs
        indices, model_params = [], []
        for izs, z_s_bin in enumerate(lookUpBins['zs']['grid']): 
            
            #> setting zs
            paramRanges['zs'] = z_s_bin
            
            #> mvir
            for imvir, logmvir_bin in enumerate(lookUpBins['logmvir']['grid']):
                
                #> declarations (for c, M_star, and R_eff)
                M_virs = np.array([10**logmvir_bin])
                zls    = np.array([z_l_bin])
                
                #> setting log10(M_vir)
                paramRanges['nfw'][0]['logmass']['init'] = logmvir_bin
                
                #> setting concentration (deterministic)
                c = priors.mass_c_vir_rel(M_virs, zls, cosmo=u.cosmo)[0]
                paramRanges['nfw'][0]['concentration']['init'] = c
                
                #> mstar
                logM_star, sigma_logM = priors.stellar_h_mass_rel(M_virs, zls, return_sample=False)[0] # prior eq
                for imstar, logmstar_bin in enumerate(lookUpBins['mstar_nsig']['grid']): 
                    
                    #> sampling from prior based on sigma value
                    #> logmstar_bin = the sigma value away from the mean
                    sample_logM_star_h2 = logM_star + (logmstar_bin * sigma_logM)
                    sample_logM_star = np.log10(10**sample_logM_star_h2 / h_shuntov**2)
                    
                    
                    #> setting log10(M_star)
                    paramRanges['hern'][0]['logmass']['init'] = sample_logM_star
                    
                    #> declarations
                    M_stars = np.array([10**sample_logM_star])
                    
                    #> re
                    mean_logR, sigma_logR = priors.mass_r_relation_18mowla(M_stars, zls, return_sample=False)[0]
                    for ire, re_bin in enumerate(lookUpBins['re_nsig']['grid']): 
                        
                        #> sampling from prior based on sigma value
                        #> re_bin = the sigma value away from the mean
                        sample_logR = mean_logR + (re_bin * sigma_logR)
                        
                        #> setting R_eff
                        paramRanges['hern'][0]['effrad']['init'] = 10**(sample_logR)
                        
                        indices.append([izs, imvir, imstar, ire])
                        model_params.append([z_s_bin, logmvir_bin, c, sample_logM_star, 10**(sample_logR)])
                        
                        #> pix_arc
                        for ipixarc, pix_arc_bin in enumerate(lookUpBins['pix_arc']['grid']):
                            
                            #> setting pix_arc
                            paramRanges['pix_arc'] = pix_arc_bin
                            
                            #> for checking
                            if izl==izs==imvir==imstar==ire==0 and False: 
                                print(z_l_bin, z_s_bin, logmvir_bin, logmstar_bin, re_bin)
                                print(logM_star, sample_logM_star, sigma_logM, sigma_logM*logmstar_bin)
                                print(mean_logR, sample_logR, sigma_logR, sigma_logR*re_bin)
                                print(paramRanges)
                                
                            #> getting bprofiles
                            batch_profiles = params.bprofiles(numgals=1, 
                                                              ranges=paramRanges, 
                                                              priorDict=priorDict,
                                                              verbose=False)
                            bprofiles.append(batch_profiles[0])
                            
                        # re
                    # mstar
                # mvir
            # zs
        # zl
        
        #> generating galaxy population at given zl
        lenses, bprofiles = generate.genGalPop(redshifts=[], bprofiles=bprofiles, nph=nph)
        
        #> getting image positions for \beta=0
        images, im_obs, sources, im_mags = generate.genQuadPop(lenses, bprofiles, warnings=False, source=(0.0,0.0))
        (xgrid, ygrid, delx, dely, lamt) = lenses
        
        #> measuring parameters
        dareas, oareas, ers = [], [], []
        num_pix_arc = lookUpBins['pix_arc']['bins']
        for i in range(len(delx)):
            
            #> measuring strong lensing cross section
            dcaustic, ocaustic = lensing.caustics(xgrid, 
                                                  ygrid,
                                                  delx[i], 
                                                  dely[i], 
                                                  bprofiles[i]['pix_arc'])
            
            #> plotting caustics to help visualize
            # plot.caustics(xgrid, ygrid, delx[i], dely[i], bprofiles[i]['pix_arc'], images=images[i])
            
            #> checking to see if caustics are coherent (i.e., only one linestring)
            #> and if there is a caustic
            
            #> getting caustic areas
            darea, oarea = lensing.causticArea(dcaustic, ocaustic)
            
            #> getting einstein radius
            observables = ['d01', 'd02', 'd03', 'd04']
            if len(images[i]) == 5: # if a quad
                
                #> getting distances
                lens_obs = geometry.lensObs(origin=(0,0), 
                                            images=images[i], 
                                            observables=observables)
                #> einstein radius
                ER = np.mean(lens_obs)
            
            else: # if not a quad
                ER = None
            
            #> appending
            dareas.append(darea)
            oareas.append(oarea)
            ers.append(ER)
            
            #> plotting
            # plot.ccurves(xgrid, ygrid, delx[i], dely[i], bprofiles[i]['pix_arc'])
            # plot.caustics(xgrid, ygrid, delx[i], dely[i], bprofiles[i]['pix_arc'])
        
        #> reshaping
        dareas = np.array(dareas).reshape(len(indices), num_pix_arc)
        oareas = np.array(oareas).reshape(len(indices), num_pix_arc)
        ers = np.array(ers).reshape(len(indices), num_pix_arc)
        
        #> collating / median / etc.
        darea = np.array([ np.median(x[x != None]) for x in dareas ]).reshape(len(indices),1)
        oarea = np.array([ np.median(x[x != None]) for x in oareas ]).reshape(len(indices),1)
        er    = np.array([ np.median(x[x != None]) for x in ers    ]).reshape(len(indices),1)        
        
        #> stacking
        columns = ['izs', 'ilogmvir', 'imstar_nsig', 'ire_nsig', 
                   'zs', 'logM_vir', 'c', 'logM_star', 'R_eff', 
                   'd_area', 'o_area', 'ER']
        all_model_info = np.hstack([indices, model_params, darea, oarea, er])
        df = pd.DataFrame(all_model_info, columns=columns)
        
        #> writing to file
        zl = z_l_bin
        loc = './priors/look_up_tables/'
        outFile = loc+f'model_zl_{zl:.2f}.parquet'
        df.to_parquet(outFile)

    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__),'\n')
    
    #> calling generate function
    function()
    
    # end
# thank