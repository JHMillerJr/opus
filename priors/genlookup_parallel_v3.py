#> name: genlookup_parallel_v3.py
#> author: John Miller Jr
#> descrp: generates high-resolution adaptive strong-lensing look-up tables
#>         v2 tables are used only to estimate the initial grid scale

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import json
import argparse
import inspect
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> GRID DEFINITIONS ==============
================================== """

#> returns v3 lookup bins
def look_up_bins_v3():

    #> declarations
    nzl = 30
    nzs = 30
    nmvir = 30

    #> denser at low lens redshift
    t = np.linspace(0,1,nzl)
    zl_grid = 0.01 + (1.8-0.01)*t**1.7

    #> denser at low source redshift
    t = np.linspace(0,1,nzs)
    zs_grid = 0.10 + (5.0-0.10)*t**1.3

    #> halo mass
    logmvir_grid = np.linspace(11.5,14.5,nmvir)

    #> scatter grids
    nsig_grid = np.array([-3.0,-1.75,-0.75,0.0,0.75,1.75,3.0])

    #> rounding
    zl_grid = np.round(zl_grid,8)
    zs_grid = np.round(zs_grid,8)
    logmvir_grid = np.round(logmvir_grid,8)

    return {'zl':         {'bins': len(zl_grid),      'grid': zl_grid},
            'zs':         {'bins': len(zs_grid),      'grid': zs_grid},
            'logmvir':    {'bins': len(logmvir_grid), 'grid': logmvir_grid},
            'mstar_nsig': {'bins': len(nsig_grid),    'grid': nsig_grid},
            're_nsig':    {'bins': len(nsig_grid),    'grid': nsig_grid}}


#> checks lookup grid
def _check_grid(bins):

    #> checking monotonic grids
    for key in ['zl','zs','logmvir','mstar_nsig','re_nsig']:
        grid = np.asarray(bins[key]['grid'])
        if np.any(np.diff(grid) <= 0):
            raise ValueError(f'{key} lookup grid is not strictly increasing')

    #> checking rounded file names
    names = [f'{zl:.2f}' for zl in bins['zl']['grid']]
    if len(names) != len(set(names)):
        raise ValueError('zl grid has duplicate file names after rounding to 2 decimals')

    return


""" #> PARALLEL WORKERS ==============
================================== """

#> process-local storage
_worker = {}


#> initializes each worker once
def _init_worker(config):

    #> imports
    import generate
    import params
    import priors
    import lensing
    import modules.geometry as geometry
    from skimage import measure
    from modules.units import u; u=u()

    #> galaxy profiles
    galProfs = generate.galProfiles(nfw=True,
                                    hern=True,
                                    mult=[],
                                    ex=False)

    #> no random priors in lookup construction
    priorDict = params.getPriorDict(hmf=False,
                                    cmr=False,
                                    shmr=False,
                                    msr=False)

    #> generate function signature
    first_arg = next(iter(inspect.signature(generate.genGalPop).parameters))

    #> worker-local objects
    _worker['generate'] = generate
    _worker['params'] = params
    _worker['priors'] = priors
    _worker['lensing'] = lensing
    _worker['geometry'] = geometry
    _worker['measure'] = measure
    _worker['u'] = u

    _worker['galProfs'] = galProfs
    _worker['priorDict'] = priorDict
    _worker['gen_numgals'] = first_arg == 'numGals'

    _worker['bins'] = look_up_bins_v3()
    _worker['config'] = config

    #> seed tables
    _worker['seed_bins'] = _load_seed_bins(config)
    _worker['seed_tables'] = OrderedDict()
    _worker['seed_blocks'] = OrderedDict()

    return


""" #> V2 LOOKUP SEED =================
================================== """

#> loads seed lookup bins
def _load_seed_bins(config):

    if not config['use_seed']:
        return None

    metaFile = os.path.join(config['seed_dir'], config['seed_meta'])
    if not os.path.exists(metaFile):
        return None

    with open(metaFile) as f:
        meta = json.load(f)

    return {'zl':         {'grid': np.asarray(meta['zl_grid'], dtype=float)},
            'zs':         {'grid': np.asarray(meta['zs_grid'], dtype=float)},
            'logmvir':    {'grid': np.asarray(meta['logmvir_grid'], dtype=float)},
            'mstar_nsig': {'grid': np.asarray(meta['mstar_nsig_grid'], dtype=float)},
            're_nsig':    {'grid': np.asarray(meta['re_nsig_grid'], dtype=float)}}


#> nearest bin
def _nearest(x, grid):
    return int(np.abs(np.asarray(grid)-x).argmin())


#> loads one seed zl table
def _load_seed_table(izl):

    cache = _worker['seed_tables']
    config = _worker['config']
    bins = _worker['seed_bins']

    if bins is None:
        return None

    #> cached table
    if izl in cache:
        cache.move_to_end(izl)
        return cache[izl]

    #> loading table
    zl = bins['zl']['grid'][izl]
    fileName = os.path.join(config['seed_dir'], f'model_zl_{zl:.2f}.parquet')

    if not os.path.exists(fileName):
        return None

    cols = ['logM_star','R_eff','ER']
    table = pd.read_parquet(fileName, columns=cols)

    #> limiting memory
    cache[izl] = table
    cache.move_to_end(izl)

    while len(cache) > config['seed_table_cache']:
        cache.popitem(last=False)

    return table


#> loads one seed mstar/re block
def _load_seed_block(izl, izs, imvir):

    key = (int(izl),int(izs),int(imvir))
    cache = _worker['seed_blocks']
    config = _worker['config']
    bins = _worker['seed_bins']

    #> cached block
    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    table = _load_seed_table(int(izl))
    if table is None:
        return None

    #> declarations
    nzs = len(bins['zs']['grid'])
    nmvir = len(bins['logmvir']['grid'])
    nmstar = len(bins['mstar_nsig']['grid'])
    nre = len(bins['re_nsig']['grid'])

    dims = (nzs,nmvir,nmstar,nre)
    start = np.ravel_multi_index((int(izs),int(imvir),0,0), dims)
    nblock = nmstar*nre

    block = table.iloc[start:start+nblock]
    if len(block) != nblock:
        return None

    #> physical coordinates
    mstar = block['logM_star'].to_numpy(float)
    reff = block['R_eff'].to_numpy(float)
    er = pd.to_numeric(block['ER'], errors='coerce').to_numpy(float)

    valid = np.isfinite(mstar) & np.isfinite(reff) & (reff > 0) & np.isfinite(er) & (er > 0)

    out = (mstar,
           np.log10(np.where(reff > 0, reff, np.nan)),
           er,
           valid)

    #> limiting memory
    cache[key] = out
    cache.move_to_end(key)

    while len(cache) > config['seed_block_cache']:
        cache.popitem(last=False)

    return out


#> v2 ER estimate for initial pixel scale
def _seed_er(zl, zs, logmvir, logmstar, reff):

    config = _worker['config']
    bins = _worker['seed_bins']

    if bins is None:
        return config['fallback_er']

    #> nearest seed cell
    izl = _nearest(zl, bins['zl']['grid'])
    izs = _nearest(zs, bins['zs']['grid'])
    imvir = _nearest(logmvir, bins['logmvir']['grid'])

    block = _load_seed_block(izl,izs,imvir)
    if block is None:
        return config['fallback_er']

    mstar, logre, er, valid = block
    if not np.any(valid):
        return config['fallback_er']

    #> nearest physical mstar/re model
    dm = (mstar[valid]-logmstar)/config['seed_mstar_scale']
    dr = (logre[valid]-np.log10(reff))/config['seed_re_scale']
    dist = dm**2 + dr**2

    seed = er[valid][np.argmin(dist)]

    return float(np.clip(seed,
                         config['seed_er_min'],
                         config['seed_er_max']))


""" #> PROFILE GENERATION =============
================================== """

#> generates lens maps
def _gen_lenses(bprofiles):

    generate = _worker['generate']
    config = _worker['config']

    if _worker['gen_numgals']:
        return generate.genGalPop(len(bprofiles),
                                  bprofiles=bprofiles,
                                  nph=config['nph'])

    return generate.genGalPop(redshifts=[],
                              bprofiles=bprofiles,
                              nph=config['nph'])


#> changes only pixel scale
def _set_pix_arc(bprofiles, pix_arc, inds=None):

    if inds is None:
        inds = np.arange(len(bprofiles))

    out = []
    for i, pa in zip(inds,pix_arc):
        prof = deepcopy(bprofiles[int(i)])
        prof['pix_arc'] = float(pa)
        out.append(prof)

    return out


#> builds physical models for one zl/zs cell
def _build_models(z_l_bin, z_s_bin):

    #> worker objects
    params = _worker['params']
    priors = _worker['priors']
    u = _worker['u']

    bins = _worker['bins']
    config = _worker['config']

    galProfs = _worker['galProfs']
    priorDict = _worker['priorDict']

    #> declarations
    h_shuntov = 0.7
    physical = z_s_bin > z_l_bin

    #> fresh parameter state
    paramRanges = params.toggleParams(galProfs)
    paramRanges['zl'] = z_l_bin
    paramRanges['zs'] = z_s_bin

    #> output arrays
    indices = []
    model_params = []
    bprofiles = []
    seed_ers = []

    #> mvir
    for imvir, logmvir_bin in enumerate(bins['logmvir']['grid']):

        M_virs = np.array([10**logmvir_bin])
        zls = np.array([z_l_bin])

        #> halo mass
        paramRanges['nfw'][0]['logmass']['init'] = logmvir_bin

        #> concentration
        c = priors.mass_c_vir_rel(M_virs,
                                  zls,
                                  cosmo=u.cosmo)[0]

        paramRanges['nfw'][0]['concentration']['init'] = c

        #> stellar mass relation
        logM_star, sigma_logM = priors.stellar_h_mass_rel(M_virs,
                                                           zls,
                                                           return_sample=False)[0]

        #> mstar
        for imstar, mstar_nsig in enumerate(bins['mstar_nsig']['grid']):

            sample_logM_star_h2 = logM_star + mstar_nsig*sigma_logM
            sample_logM_star = np.log10(10**sample_logM_star_h2/h_shuntov**2)

            paramRanges['hern'][0]['logmass']['init'] = sample_logM_star

            M_stars = np.array([10**sample_logM_star])

            #> mass-size relation
            mean_logR, sigma_logR = priors.mass_r_relation_18mowla(M_stars,
                                                                    zls,
                                                                    return_sample=False)[0]

            #> re
            for ire, re_nsig in enumerate(bins['re_nsig']['grid']):

                sample_logR = mean_logR + re_nsig*sigma_logR
                reff = 10**sample_logR

                paramRanges['hern'][0]['effrad']['init'] = reff

                #> seed ER
                if physical:
                    seed_er = _seed_er(z_l_bin,
                                       z_s_bin,
                                       logmvir_bin,
                                       sample_logM_star,
                                       reff)
                else:
                    seed_er = config['fallback_er']

                #> initial wide scale
                pix_arc = config['seed_pix_wide']/seed_er
                pix_arc = np.clip(pix_arc,
                                  config['pix_arc_min'],
                                  config['pix_arc_max'])

                paramRanges['pix_arc'] = pix_arc

                #> physical profile
                if physical:
                    batch = params.bprofiles(numgals=1,
                                             ranges=paramRanges,
                                             priorDict=priorDict,
                                             verbose=False)
                    bprofiles.append(batch[0])
                else:
                    bprofiles.append(None)

                #> saving model
                indices.append([imvir,imstar,ire])
                model_params.append([logmvir_bin,
                                     c,
                                     sample_logM_star,
                                     reff,
                                     mstar_nsig,
                                     re_nsig])
                seed_ers.append(seed_er)

    return (np.asarray(indices, dtype=int),
            np.asarray(model_params, dtype=float),
            bprofiles,
            np.asarray(seed_ers, dtype=float))


""" #> LENS MEASUREMENTS ==============
================================== """

#> checks tangential critical curve
def _critical_curve_status(lamt):

    measure = _worker['measure']
    config = _worker['config']

    #> finding lamt=0 contours
    contours = measure.find_contours(np.asarray(lamt).T, 0.0)
    ncurve = len(contours)

    if ncurve == 0:
        return 0, False

    #> checking if curve touches grid edge
    ny, nx = np.asarray(lamt).shape
    pad = config['curve_edge_pad']

    touches = False
    for contour in contours:
        y = contour[:,0]
        x = contour[:,1]

        if (np.any(x <= pad) or np.any(x >= nx-1-pad) or
            np.any(y <= pad) or np.any(y >= ny-1-pad)):
            touches = True
            break

    return ncurve, touches


#> centered-source ER
def _measure_er(xgrid, ygrid, delx, dely, pix_arc):

    lensing = _worker['lensing']
    geometry = _worker['geometry']
    u = _worker['u']
    nph = _worker['config']['nph']

    #> lensing args
    args = (xgrid/pix_arc/u.arc_rad,
            ygrid/pix_arc/u.arc_rad,
            delx,
            dely,
            0.0,
            0.0,
            pix_arc,
            nph)

    #> image positions
    try:
        ims = lensing.lfims(*args, ordered=True)
    except TypeError:
        ims = lensing.lfims(*args)

    if len(ims) != 5:
        return np.nan

    #> mean image radius
    obs = geometry.lensObs(origin=(0,0),
                           images=ims,
                           observables=['d01','d02','d03','d04'])

    return float(np.mean(obs))


#> empty measurement arrays
def _empty_measurement(n):

    return {'ER': np.full(n,np.nan),
            'd_area': np.full(n,np.nan),
            'o_area': np.full(n,np.nan),
            'pix_arc': np.full(n,np.nan),
            'window_bad': np.zeros(n,dtype=bool),
            'n_tcurve': np.zeros(n,dtype=int),
            'multi_curve': np.zeros(n,dtype=bool),
            'fatal': np.zeros(n,dtype=bool)}


#> measures one deflection batch
def _measure_chunk(bprofiles):

    lensing = _worker['lensing']

    #> lens maps
    lenses, bprofiles = _gen_lenses(bprofiles)
    xgrid, ygrid, delx, dely, lamt = lenses

    n = len(bprofiles)
    out = _empty_measurement(n)

    #> each lens
    for i in range(n):

        pa = float(bprofiles[i]['pix_arc'])
        out['pix_arc'][i] = pa

        #> critical curve status
        ncurve, window_bad = _critical_curve_status(lamt[i])

        out['n_tcurve'][i] = ncurve
        out['multi_curve'][i] = ncurve > 1
        out['window_bad'][i] = window_bad

        #> curve leaves window --> rerun wider
        if window_bad:
            continue

        #> caustic areas
        try:
            dcaustic, ocaustic = lensing.caustics(xgrid,
                                                  ygrid,
                                                  delx[i],
                                                  dely[i],
                                                  pa)

            darea, oarea = lensing.causticArea(dcaustic, ocaustic)

            if darea is not None and np.isfinite(darea):
                out['d_area'][i] = float(darea)

            if oarea is not None and np.isfinite(oarea):
                out['o_area'][i] = float(oarea)

        except (Exception,SystemExit):
            pass

        #> einstein radius
        try:
            out['ER'][i] = _measure_er(xgrid,
                                       ygrid,
                                       delx[i],
                                       dely[i],
                                       pa)

        except (Exception,SystemExit):
            out['ER'][i] = np.nan

    return out


#> combines measurement dictionaries
def _concat_measurements(parts):

    if len(parts) == 1:
        return parts[0]

    out = {}
    for key in parts[0]:
        out[key] = np.concatenate([part[key] for part in parts])

    return out


#> isolates models if a whole batch crashes
def _measure_safe(bprofiles):

    n = len(bprofiles)

    try:
        return _measure_chunk(bprofiles)

    except (Exception,SystemExit):

        #> single bad model
        if n == 1:
            out = _empty_measurement(1)
            out['pix_arc'][0] = float(bprofiles[0]['pix_arc'])
            out['fatal'][0] = True
            return out

        #> splitting failed batch
        mid = n//2
        left = _measure_safe(bprofiles[:mid])
        right = _measure_safe(bprofiles[mid:])

        return _concat_measurements([left,right])


#> measures profiles in memory-safe batches
def _measure(bprofiles):

    config = _worker['config']
    batch_size = config['measure_batch_size']

    parts = []

    for i in range(0,len(bprofiles),batch_size):
        parts.append(_measure_safe(bprofiles[i:i+batch_size]))

    return _concat_measurements(parts)


#> measures one scale and widens failed windows
def _measure_with_recovery(bprofiles, pix_arc):

    config = _worker['config']

    n = len(bprofiles)
    current = np.clip(np.asarray(pix_arc,dtype=float),
                      config['pix_arc_min'],
                      config['pix_arc_max'])

    final = _empty_measurement(n)
    retries = np.zeros(n,dtype=int)
    window_failed = np.zeros(n,dtype=bool)

    active = np.arange(n)

    #> widening retries
    for attempt in range(config['window_retries']+1):

        if len(active) == 0:
            break

        #> current trial
        profiles = _set_pix_arc(bprofiles,
                                 current[active],
                                 inds=active)
        trial = _measure(profiles)

        #> ER size in pixels
        er_pix = trial['ER']*trial['pix_arc']
        too_large = (np.isfinite(er_pix) &
                     (er_pix > config['max_er_frac']*config['nph']))

        #> window/fatal failures get wider field
        bad = trial['window_bad'] | too_large | trial['fatal']
        good = ~bad

        #> saving good models
        if np.any(good):
            ginds = active[good]

            for key in final:
                final[key][ginds] = trial[key][good]

        #> failed models
        binds = active[bad]
        if len(binds) == 0:
            active = np.array([],dtype=int)
            break

        retries[binds] += 1

        #> no retries left
        if attempt == config['window_retries']:

            for key in final:
                final[key][binds] = trial[key][bad]

            window_failed[binds] = True
            active = np.array([],dtype=int)
            break

        #> widen angular field
        old = current[binds].copy()
        current[binds] = np.maximum(config['pix_arc_min'],
                                    old*config['window_scale'])

        #> cannot widen further
        stuck = current[binds] >= old
        if np.any(stuck):

            sinds = binds[stuck]
            trial_inds = np.where(bad)[0][stuck]

            for key in final:
                final[key][sinds] = trial[key][trial_inds]

            window_failed[sinds] = True

        active = binds[~stuck]

    final['window_retries'] = retries
    final['window_failed'] = window_failed

    return final


#> relative difference
def _reldiff(a, b, floor=1e-8):
    return np.abs(a-b)/np.maximum(np.maximum(np.abs(a),np.abs(b)),floor)


#> valid ER measurement
def _valid_er(result):
    return (np.isfinite(result['ER']) &
            (result['ER'] > 0) &
            ~result['window_failed'])


#> picks best initial scale
def _pick_bracket(wide, fine):

    config = _worker['config']
    target = config['target_pix']

    n = len(wide['ER'])
    choice = np.zeros(n,dtype=int)

    wide_valid = _valid_er(wide)
    fine_valid = _valid_er(fine)

    #> only fine works
    choice[~wide_valid & fine_valid] = 1

    #> both work
    both = wide_valid & fine_valid
    if np.any(both):

        wide_pix = wide['ER'][both]*wide['pix_arc'][both]
        fine_pix = fine['ER'][both]*fine['pix_arc'][both]

        use_fine = np.abs(fine_pix-target) < np.abs(wide_pix-target)
        choice[np.where(both)[0][use_fine]] = 1

    #> selecting results
    out = {}
    for key in wide:
        out[key] = np.where(choice == 0, wide[key], fine[key])

    out['valid'] = wide_valid | fine_valid
    out['window_retries'] = wide['window_retries'] + fine['window_retries']

    return out


""" #> ADAPTIVE SOLVER ================
================================== """

#> solves one physical model batch
def _adaptive_measure(bprofiles, seed_ers):

    config = _worker['config']
    n = len(bprofiles)

    #> scales around seed ER
    pix_wide = np.clip(config['seed_pix_wide']/seed_ers,
                       config['pix_arc_min'],
                       config['pix_arc_max'])

    pix_fine = np.clip(config['seed_pix_fine']/seed_ers,
                       config['pix_arc_min'],
                       config['pix_arc_max'])

    #> initial bracket
    wide = _measure_with_recovery(bprofiles,pix_wide)
    fine = _measure_with_recovery(bprofiles,pix_fine)
    prev = _pick_bracket(wide,fine)

    #> final values
    final = {key: np.array(prev[key],copy=True)
             for key in ['ER','d_area','o_area','pix_arc',
                         'n_tcurve','multi_curve','window_failed']}

    #> diagnostics
    n_pass = np.full(n,2,dtype=int)
    er_err = np.full(n,np.nan)
    da_err = np.full(n,np.nan)

    er_converged = np.zeros(n,dtype=bool)
    da_converged = np.zeros(n,dtype=bool)

    window_retries = np.array(prev['window_retries'],copy=True)

    #> adaptive passes
    for ipass in range(3,config['max_passes']+1):

        active = np.where(prev['valid'] & ~(er_converged & da_converged))[0]
        if len(active) == 0:
            break

        #> target ER pixel radius
        pix = config['target_pix']/prev['ER'][active]
        pix = np.clip(pix,
                      config['pix_arc_min'],
                      config['pix_arc_max'])

        trial = _measure_with_recovery([bprofiles[i] for i in active],pix)
        valid = _valid_er(trial)

        #> convergence errors
        er_rel = np.full(len(active),np.inf)
        da_rel = np.full(len(active),np.inf)

        er_rel[valid] = _reldiff(trial['ER'][valid],
                                 prev['ER'][active][valid])

        da_valid = (valid &
                    np.isfinite(trial['d_area']) &
                    np.isfinite(prev['d_area'][active]))

        da_rel[da_valid] = _reldiff(trial['d_area'][da_valid],
                                    prev['d_area'][active][da_valid],
                                    floor=config['darea_floor'])

        #> small cross sections
        tiny_da = (da_valid &
                   (np.maximum(trial['d_area'],
                               prev['d_area'][active]) < config['darea_floor']))

        er_ok = valid & (er_rel <= config['er_tol'])
        da_ok = da_valid & ((da_rel <= config['darea_tol']) | tiny_da)

        #> final ER pixel radius
        er_pix = trial['ER']*trial['pix_arc']
        pix_ok = (valid &
                  (er_pix >= config['min_er_pix']) &
                  (er_pix <= config['max_er_frac']*config['nph']))

        er_done = er_ok & pix_ok
        da_done = da_ok & pix_ok

        #> updating valid models
        good = np.where(valid)[0]
        if len(good):

            ginds = active[good]

            #> ER values
            final['ER'][ginds] = trial['ER'][good]
            final['pix_arc'][ginds] = trial['pix_arc'][good]
            final['n_tcurve'][ginds] = trial['n_tcurve'][good]
            final['multi_curve'][ginds] = trial['multi_curve'][good]
            final['window_failed'][ginds] = trial['window_failed'][good]

            #> area values
            da_good = np.isfinite(trial['d_area'][good])
            oa_good = np.isfinite(trial['o_area'][good])

            final['d_area'][ginds[da_good]] = trial['d_area'][good][da_good]
            final['o_area'][ginds[oa_good]] = trial['o_area'][good][oa_good]

            #> diagnostics
            er_err[ginds] = er_rel[good]
            da_err[ginds] = da_rel[good]
            n_pass[ginds] = ipass

            #> next reference
            prev['ER'][ginds] = trial['ER'][good]
            prev['pix_arc'][ginds] = trial['pix_arc'][good]

            prev_da_good = np.isfinite(trial['d_area'][good])
            prev['d_area'][ginds[prev_da_good]] = trial['d_area'][good][prev_da_good]

        #> convergence state
        er_converged[active] = er_done
        da_converged[active] = da_done

        #> retries
        window_retries[active] += trial['window_retries']

        #> invalid target run --> keep best bracket but stop repeating
        failed = active[~valid]
        if len(failed):
            prev['valid'][failed] = False

    #> invalid models
    invalid = ~np.isfinite(final['ER']) | (final['ER'] <= 0)
    final['ER'][invalid] = np.nan

    #> final diagnostics
    final['ER_pix'] = final['ER']*final['pix_arc']
    final['ER_relerr'] = er_err
    final['d_area_relerr'] = da_err
    final['n_pass'] = n_pass
    final['ER_converged'] = er_converged
    final['d_area_converged'] = da_converged
    final['converged'] = er_converged & da_converged
    final['window_retries'] = window_retries

    return final


""" #> ZS WORKER ======================
================================== """

#> empty result for unphysical redshifts
def _empty_result(n):

    return {'ER': np.full(n,np.nan),
            'd_area': np.full(n,np.nan),
            'o_area': np.full(n,np.nan),
            'pix_arc': np.full(n,np.nan),
            'ER_pix': np.full(n,np.nan),
            'ER_relerr': np.full(n,np.nan),
            'd_area_relerr': np.full(n,np.nan),
            'n_pass': np.zeros(n,dtype=int),
            'ER_converged': np.zeros(n,dtype=bool),
            'd_area_converged': np.zeros(n,dtype=bool),
            'converged': np.zeros(n,dtype=bool),
            'window_retries': np.zeros(n,dtype=int),
            'window_failed': np.zeros(n,dtype=bool),
            'n_tcurve': np.zeros(n,dtype=int),
            'multi_curve': np.zeros(n,dtype=bool)}


#> generates one zs bin at fixed zl
def _generate_zs(job):

    #> unpacking
    izl, z_l_bin, izs, z_s_bin = job

    #> physical models
    indices, model_params, bprofiles, seed_ers = _build_models(z_l_bin,z_s_bin)
    n = len(indices)

    #> lensing
    if z_s_bin > z_l_bin:
        result = _adaptive_measure(bprofiles,seed_ers)
    else:
        result = _empty_result(n)

    #> stacking
    data = np.column_stack([
        np.full(n,izs),
        indices,
        np.full(n,z_l_bin),
        np.full(n,z_s_bin),
        model_params,
        result['d_area'],
        result['o_area'],
        result['ER'],
        result['pix_arc'],
        result['ER_pix'],
        result['ER_relerr'],
        result['d_area_relerr'],
        result['n_pass'],
        result['ER_converged'].astype(int),
        result['d_area_converged'].astype(int),
        result['converged'].astype(int),
        result['window_retries'],
        result['window_failed'].astype(int),
        result['n_tcurve'],
        result['multi_curve'].astype(int),
        seed_ers
    ])

    return data


""" #> OUTPUT =========================
================================== """

#> saves exact grid/config
def _save_metadata(out_dir, config):

    bins = look_up_bins_v3()

    metadata = {'version': 3,
                'zl_grid': bins['zl']['grid'].tolist(),
                'zs_grid': bins['zs']['grid'].tolist(),
                'logmvir_grid': bins['logmvir']['grid'].tolist(),
                'mstar_nsig_grid': bins['mstar_nsig']['grid'].tolist(),
                're_nsig_grid': bins['re_nsig']['grid'].tolist(),
                'seed_dir': config['seed_dir'],
                'seed_meta': config['seed_meta'],
                'nph': config['nph'],
                'seed_pix_wide': config['seed_pix_wide'],
                'seed_pix_fine': config['seed_pix_fine'],
                'target_pix': config['target_pix'],
                'max_passes': config['max_passes'],
                'er_tol': config['er_tol'],
                'darea_tol': config['darea_tol'],
                'min_er_pix': config['min_er_pix'],
                'max_er_frac': config['max_er_frac'],
                'pix_arc_min': config['pix_arc_min'],
                'pix_arc_max': config['pix_arc_max'],
                'window_retries': config['window_retries'],
                'window_scale': config['window_scale'],
                'curve_edge_pad': config['curve_edge_pad'],
                'measure_batch_size': config['measure_batch_size']}

    os.makedirs(out_dir,exist_ok=True)

    with open(os.path.join(out_dir,'lookup_grid_v3.json'),'w') as f:
        json.dump(metadata,f,indent=2)

    return


#> adds cross-section normalization
def _finalize_table(df, max_er=2.4):

    df['d_area_max'] = 0.0
    df['p_quad'] = 0.0

    #> each source redshift
    for izs in df['izs'].unique():

        mask_zs = df['izs'] == izs

        #> valid normalization models
        base = (mask_zs &
                np.isfinite(df['ER']) &
                (df['ER'] <= max_er) &
                np.isfinite(df['d_area']))

        good = base & df['converged']
        mask_max = good if np.any(good) else base

        if np.any(mask_max):
            dmax = df.loc[mask_max,'d_area'].max()
        else:
            dmax = 0.0

        if not np.isfinite(dmax):
            dmax = 0.0

        df.loc[mask_zs,'d_area_max'] = dmax

        #> quad probability
        if dmax > 0:
            p = df.loc[mask_zs,'d_area']/dmax
            df.loc[mask_zs,'p_quad'] = p.clip(0,1).fillna(0)

    return df


#> dataframe from worker output
def _to_dataframe(data):

    columns = ['izs',
               'ilogmvir',
               'imstar_nsig',
               'ire_nsig',
               'zl',
               'zs',
               'logM_vir',
               'c',
               'logM_star',
               'R_eff',
               'mstar_nsig',
               're_nsig',
               'd_area',
               'o_area',
               'ER',
               'pix_arc',
               'ER_pix',
               'ER_relerr',
               'd_area_relerr',
               'n_pass',
               'ER_converged',
               'd_area_converged',
               'converged',
               'window_retries',
               'window_failed',
               'n_tcurve',
               'multi_curve',
               'seed_ER']

    df = pd.DataFrame(data,columns=columns)

    #> integer columns
    int_cols = ['izs','ilogmvir','imstar_nsig','ire_nsig',
                'n_pass','window_retries','n_tcurve']
    df[int_cols] = df[int_cols].astype(int)

    #> boolean columns
    bool_cols = ['ER_converged','d_area_converged','converged',
                 'window_failed','multi_curve']
    df[bool_cols] = df[bool_cols].astype(bool)

    return df


""" #> GENERATE =======================
================================== """

#> generates v3 lookup tables
def function(workers=None, **kwargs):

    #> paths
    seed_dir = kwargs.get('seed_dir','../priors/look_up_tables_v2/')
    out_dir = kwargs.get('out_dir','../priors/look_up_tables_v3/')

    #> configuration
    config = {
        'seed_dir': seed_dir,
        'out_dir': out_dir,
        'seed_meta': kwargs.get('seed_meta','lookup_grid_v2.json'),
        'use_seed': kwargs.get('use_seed',True),

        #> grid
        'nph': kwargs.get('nph',80),
        'measure_batch_size': kwargs.get('measure_batch_size',32),

        #> seed ER
        'seed_pix_wide': kwargs.get('seed_pix_wide',8.0),
        'seed_pix_fine': kwargs.get('seed_pix_fine',32.0),
        'fallback_er': kwargs.get('fallback_er',0.8),
        'seed_er_min': kwargs.get('seed_er_min',0.05),
        'seed_er_max': kwargs.get('seed_er_max',8.0),
        'seed_mstar_scale': kwargs.get('seed_mstar_scale',0.25),
        'seed_re_scale': kwargs.get('seed_re_scale',0.25),
        'seed_table_cache': kwargs.get('seed_table_cache',2),
        'seed_block_cache': kwargs.get('seed_block_cache',512),

        #> adaptive solution
        'target_pix': kwargs.get('target_pix',25.0),
        'max_passes': kwargs.get('max_passes',4),
        'er_tol': kwargs.get('er_tol',0.02),
        'darea_tol': kwargs.get('darea_tol',0.07),
        'darea_floor': kwargs.get('darea_floor',1e-4),

        #> valid scale
        'min_er_pix': kwargs.get('min_er_pix',14.0),
        'max_er_frac': kwargs.get('max_er_frac',0.65),
        'pix_arc_min': kwargs.get('pix_arc_min',0.5),
        'pix_arc_max': kwargs.get('pix_arc_max',250.0),

        #> window recovery
        'window_retries': kwargs.get('window_retries',4),
        'window_scale': kwargs.get('window_scale',0.60),
        'curve_edge_pad': kwargs.get('curve_edge_pad',2.0)
    }

    #> checking configuration
    if config['measure_batch_size'] < 1:
        raise ValueError('measure_batch_size must be >= 1')

    if not 0 < config['window_scale'] < 1:
        raise ValueError('window_scale must be between 0 and 1')

    if config['seed_pix_wide'] >= config['seed_pix_fine']:
        raise ValueError('seed_pix_wide must be smaller than seed_pix_fine')

    #> bins
    bins = look_up_bins_v3()
    _check_grid(bins)

    #> checking seed tables
    seedMeta = os.path.join(seed_dir,config['seed_meta'])
    if config['use_seed'] and not os.path.exists(seedMeta):
        print(f'> seed metadata not found: {seedMeta}')
        print('> using fallback ER for initial scales')
        config['use_seed'] = False

    #> output
    os.makedirs(out_dir,exist_ok=True)
    _save_metadata(out_dir,config)

    #> workers
    if workers is None:
        workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                     os.cpu_count() or 1))

    workers = max(1,min(workers,bins['zs']['bins']))

    print(f'> zs workers: {workers}')
    print(f'> nph: {config["nph"]}')
    print(f'> target ER pixels: {config["target_pix"]}')
    print(f'> batch size: {config["measure_batch_size"]}')
    print(f'> seed tables: {seed_dir}')
    print(f'> new tables: {out_dir}')

    #> multiprocessing
    mp_context = get_context('spawn')

    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp_context,
                             initializer=_init_worker,
                             initargs=(config,)) as pool:

        #> zl stays serial
        for izl, z_l_bin in enumerate(bins['zl']['grid']):

            print(f'\n> zl = {z_l_bin:.4f}')

            #> zs jobs
            jobs = [(izl,z_l_bin,izs,z_s_bin)
                    for izs,z_s_bin in enumerate(bins['zs']['grid'])]

            results = list(pool.map(_generate_zs,jobs))

            #> full zl table
            data = np.vstack(results)
            df = _to_dataframe(data)

            #> d_area normalization
            df = _finalize_table(df,
                                 max_er=kwargs.get('max_er',2.4))

            #> saving
            outFile = os.path.join(out_dir,
                                   f'model_zl_{z_l_bin:.2f}.parquet')
            df.to_parquet(outFile,index=False)

            #> summary
            valid = np.isfinite(df['ER']).mean()
            er_conv = df['ER_converged'].mean()
            conv = df['converged'].mean()
            wfail = df['window_failed'].mean()

            print(f'> valid ER: {valid:.3f}')
            print(f'> ER converged: {er_conv:.3f}')
            print(f'> fully converged: {conv:.3f}')
            print(f'> window failed: {wfail:.3f}')
            print(f'> saved {outFile}')

    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers',type=int,default=None)
    parser.add_argument('--seed-dir',type=str,default='./priors/look_up_tables_v2/')
    parser.add_argument('--out-dir',type=str,default='./priors/look_up_tables_v3/')
    parser.add_argument('--nph',type=int,default=80)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--no-seed',action='store_true')

    args = parser.parse_args()

    #> name
    print('> '+os.path.basename(__file__),'\n')

    #> generating tables
    function(workers=args.workers,
             seed_dir=args.seed_dir,
             out_dir=args.out_dir,
             nph=args.nph,
             measure_batch_size=args.batch_size,
             use_seed=not args.no_seed)

# end
# thank
