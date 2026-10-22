#> name: gentemp_parallel.py
#> author: John Miller Jr
#> descrp: template file to generate galaxy/quad populations from generate.py
#>         parallelized only over source-redshift (zs) bins

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from scipy.stats import norm

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> PARALLEL WORKERS ==============
================================== """

#> process-local storage
_worker = {}


#> initializes each worker once
def _init_zs_worker(scale_factor, nph, nfw, hern, mult, ex, hmf, cmr, shmr, msr):

    #> imports
    import generate
    import params
    import priors
    import lensing
    import modules.geometry as geometry
    from modules.units import u; u=u()

    #> galaxy profiles
    galProfs = generate.galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)

    #> priors
    priorDict = params.getPriorDict(hmf=hmf, cmr=cmr, shmr=shmr, msr=msr)

    #> bins
    lookUpBins = priors.look_up_bins(scale_factor=scale_factor)

    #> saving worker-local objects
    _worker['generate'] = generate
    _worker['params'] = params
    _worker['priors'] = priors
    _worker['lensing'] = lensing
    _worker['geometry'] = geometry
    _worker['u'] = u
    _worker['galProfs'] = galProfs
    _worker['priorDict'] = priorDict
    _worker['lookUpBins'] = lookUpBins
    _worker['nph'] = nph
    _worker['h_shuntov'] = 0.7

    return


#> median ignoring None values
def _median_valid(x):

    vals = [v for v in x if v is not None]
    return np.median(vals) if len(vals) else np.nan


#> generates all models for one zs bin at a fixed zl
def _generate_zs(job):

    #> unpacking
    izl, z_l_bin, izs, z_s_bin = job

    #> worker-local objects
    generate = _worker['generate']
    params = _worker['params']
    priors = _worker['priors']
    lensing = _worker['lensing']
    geometry = _worker['geometry']
    u = _worker['u']
    galProfs = _worker['galProfs']
    priorDict = _worker['priorDict']
    lookUpBins = _worker['lookUpBins']
    nph = _worker['nph']
    h_shuntov = _worker['h_shuntov']

    #> fresh parameter state for this zs task
    paramRanges = params.toggleParams(galProfs)
    paramRanges['zl'] = z_l_bin
    paramRanges['zs'] = z_s_bin

    #> declarations
    bprofiles = []
    indices, model_params = [], []

    #> mvir
    for imvir, logmvir_bin in enumerate(lookUpBins['logmvir']['grid']):

        #> declarations
        M_virs = np.array([10**logmvir_bin])
        zls = np.array([z_l_bin])

        #> setting log10(M_vir)
        paramRanges['nfw'][0]['logmass']['init'] = logmvir_bin

        #> setting concentration
        c = priors.mass_c_vir_rel(M_virs, zls, cosmo=u.cosmo)[0]
        paramRanges['nfw'][0]['concentration']['init'] = c

        #> mstar
        logM_star, sigma_logM = priors.stellar_h_mass_rel(M_virs, zls, return_sample=False)[0]
        for imstar, logmstar_bin in enumerate(lookUpBins['mstar_nsig']['grid']):

            #> sampling from prior based on sigma value
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
                sample_logR = mean_logR + (re_bin * sigma_logR)

                #> setting R_eff
                paramRanges['hern'][0]['effrad']['init'] = 10**sample_logR

                #> storing model information
                indices.append([izs, imvir, imstar, ire])
                model_params.append([z_s_bin, logmvir_bin, c, sample_logM_star, 10**sample_logR])

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

    #> generating galaxy population for this zs
    lenses, bprofiles = generate.genGalPop(redshifts=[], bprofiles=bprofiles, nph=nph)

    #> getting image positions for beta=0
    images, im_obs, sources, im_mags = generate.genQuadPop(lenses,
                                                            bprofiles,
                                                            warnings=False,
                                                            source=(0.0,0.0))
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

        #> getting caustic areas
        darea, oarea = lensing.causticArea(dcaustic, ocaustic)

        #> getting einstein radius
        observables = ['d01', 'd02', 'd03', 'd04']
        if len(images[i]) == 5:

            #> getting distances
            lens_obs = geometry.lensObs(origin=(0,0),
                                        images=images[i],
                                        observables=observables)

            #> einstein radius
            ER = np.mean(lens_obs)

        else:
            ER = None

        #> appending
        dareas.append(darea)
        oareas.append(oarea)
        ers.append(ER)

    #> reshaping
    dareas = np.array(dareas, dtype=object).reshape(len(indices), num_pix_arc)
    oareas = np.array(oareas, dtype=object).reshape(len(indices), num_pix_arc)
    ers = np.array(ers, dtype=object).reshape(len(indices), num_pix_arc)

    #> collating / median / etc.
    darea = np.array([_median_valid(x) for x in dareas]).reshape(len(indices),1)
    oarea = np.array([_median_valid(x) for x in oareas]).reshape(len(indices),1)
    er = np.array([_median_valid(x) for x in ers]).reshape(len(indices),1)

    #> stacking
    all_model_info = np.hstack([indices, model_params, darea, oarea, er])

    return all_model_info


""" #> GENERATE FN ===================
================================== """

#> generates a galaxy/ quad population based on the given params
def function(workers=None):

    #> imports  (CHANGE IF GENERATE IS IN DIFF DIR)
    import generate
    import params
    from modules.units import u; u=u()

    #> declarations
    numGals       = 1
    numSource_gal = 1

    #> output kwargs
    folder   = '+unsorted'
    suffix   = ''
    verbose  = True
    timeFlag = False
    saveFlag = generate.getSaveDict(False)
    plotFlag = generate.getPlotDict(True, kappa=False)

    #> grid kwargs
    scale_factor = 1
    pix_arc      = 60 * scale_factor
    nph          = 50 * scale_factor

    #> galaxy profiles kwargs
    nfw  = True
    hern = True
    mult = []
    ex   = False
    galProfs = generate.galProfiles(nfw=nfw, hern=hern, mult=mult, ex=ex)

    #> galaxy profile params kwargs
    lb  = None
    ub  = None
    mu  = None
    cov = None
    uniform = False

    #> priors
    hmf  = False
    cmr  = False
    shmr = False
    msr  = False
    priorDict = params.getPriorDict(hmf=hmf, cmr=cmr, shmr=shmr, msr=msr)

    #> redshifts
    zl = 0.5
    zs = 1.0
    redshifts = (zl, zs)
    print(redshifts)

    #> image properties kwargs
    jims = None
    mags = False
    observables = None

    #> bprofiles
    bprofiles = None

    #> imports
    import priors

    #> getting paramRanges
    paramRanges = params.toggleParams(galProfs)

    #> getting the bins
    lookUpBins = priors.look_up_bins(scale_factor=scale_factor)

    #>>># SETTING UP BPROFILES

    #> setting worker count
    if workers is None:
        workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

    workers = max(1, min(workers, len(lookUpBins['zs']['grid'])))
    print(f'> zs workers: {workers}')

    #> worker initialization values
    worker_args = (scale_factor, nph, nfw, hern, mult, ex, hmf, cmr, shmr, msr)

    #> using spawn so each process has independent python/module state
    mp_context = get_context('spawn')

    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp_context,
                             initializer=_init_zs_worker,
                             initargs=worker_args) as pool:

        #> zl stays serial
        for izl, z_l_bin in enumerate(lookUpBins['zl']['grid']):

            print(f'> zl = {z_l_bin:.4f}')

            #> one independent task per zs
            jobs = [(izl, z_l_bin, izs, z_s_bin)
                    for izs, z_s_bin in enumerate(lookUpBins['zs']['grid'])]

            #> parallel only over zs; waits for every zs before next zl
            results = list(pool.map(_generate_zs, jobs))

            #> combining zs chunks in original zs order
            all_model_info = np.vstack(results)

            #> dataframe
            columns = ['izs', 'ilogmvir', 'imstar_nsig', 'ire_nsig',
                       'zs', 'logM_vir', 'c', 'logM_star', 'R_eff',
                       'd_area', 'o_area', 'ER']
            df = pd.DataFrame(all_model_info, columns=columns)

            #> writing to file
            zl = z_l_bin
            loc = '../priors/look_up_tables/'
            os.makedirs(loc, exist_ok=True)
            outFile = loc+f'model_zl_{zl:.2f}.parquet'
            df.to_parquet(outFile)

            print(f'> saved {outFile}')

    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None,
                        help='number of parallel zs workers; defaults to SLURM_CPUS_PER_TASK or os.cpu_count()')
    args = parser.parse_args()

    #> name
    print('> '+os.path.basename(__file__),'\n')

    #> calling generate function
    function(workers=args.workers)

    # end
# thank
