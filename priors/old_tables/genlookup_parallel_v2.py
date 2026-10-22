#> name: genlookup_parallel_v2.py
#> author: John Miller Jr
#> descrp: generates adaptive strong-lensing look-up tables
#>         old tables are used only to estimate the initial grid scale

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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> GRID DEFINITIONS ==============
================================== """

#> new lookup grid
def look_up_bins_v2():

    #> more resolution at low redshift
    t = np.linspace(0,1,20)

    zl_grid = 0.01 + (1.8-0.01) * t**1.7
    zs_grid = 0.10 + (5.0-0.10) * t**1.3

    #> physical mass grid
    logmvir_grid = np.linspace(11.5,14.5,20)

    #> denser near the center of the scatter distributions
    nsig_grid = np.array([-3.0,-1.75,-0.75,0.0,0.75,1.75,3.0])

    return {'zl':         {'bins': len(zl_grid),      'grid': zl_grid},
            'zs':         {'bins': len(zs_grid),      'grid': zs_grid},
            'logmvir':    {'bins': len(logmvir_grid), 'grid': logmvir_grid},
            'mstar_nsig': {'bins': len(nsig_grid),    'grid': nsig_grid},
            're_nsig':    {'bins': len(nsig_grid),    'grid': nsig_grid}}


#> old lookup grid
def old_look_up_bins():

    bins = {'zl':         {'min': 0.01, 'max':  1.8, 'bins': 20},
            'zs':         {'min':  0.1, 'max':  5.0, 'bins': 20},
            'logmvir':    {'min': 11.5, 'max': 14.5, 'bins': 20},
            'mstar_nsig': {'min':   -3, 'max':    3, 'bins': 7},
            're_nsig':    {'min':   -3, 'max':    3, 'bins': 7}}

    for key in bins:

        if key in ['mstar_nsig','re_nsig']:
            grid = np.linspace(bins[key]['min'], bins[key]['max'], bins[key]['bins'])
        else:
            edges = np.linspace(bins[key]['min'], bins[key]['max'], bins[key]['bins']+1)
            grid = (edges[:-1]+edges[1:])/2

        bins[key]['grid'] = np.round(grid,6)

    return bins


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
    from modules.units import u; u=u()

    #> galaxy profiles
    galProfs = generate.galProfiles(nfw=True,
                                    hern=True,
                                    mult=[],
                                    ex=False)

    #> no random priors in the lookup construction
    priorDict = params.getPriorDict(hmf=False,
                                    cmr=False,
                                    shmr=False,
                                    msr=False)

    #> worker-local objects
    _worker['generate'] = generate
    _worker['params'] = params
    _worker['priors'] = priors
    _worker['lensing'] = lensing
    _worker['geometry'] = geometry
    _worker['u'] = u

    _worker['galProfs'] = galProfs
    _worker['priorDict'] = priorDict

    _worker['bins'] = look_up_bins_v2()
    _worker['old_bins'] = old_look_up_bins()

    _worker['config'] = config
    _worker['old_tables'] = {}

    return


""" #> OLD LOOKUP SEED ===============
================================== """

#> nearest bin
def _nearest(x, grid):
    return int(np.abs(np.asarray(grid)-x).argmin())


#> loads old zl table once per worker
def _load_old_table(izl):

    cache = _worker['old_tables']
    if izl in cache:
        return cache[izl]

    config = _worker['config']
    bins = _worker['old_bins']

    zl = bins['zl']['grid'][izl]
    fileName = os.path.join(config['old_dir'], f'model_zl_{zl:.2f}.parquet')

    if not os.path.exists(fileName):
        cache[izl] = None
        return None

    df = pd.read_parquet(fileName)
    cache[izl] = df

    return df


#> old ER estimate from nearest physical Mstar/Re model
def _old_seed_er(zl, zs, logmvir, logmstar, reff):

    config = _worker['config']
    if not config['use_old']:
        return config['fallback_er']

    bins = _worker['old_bins']

    #> nearest old large-scale cell
    izl = _nearest(zl, bins['zl']['grid'])
    izs = _nearest(zs, bins['zs']['grid'])
    imvir = _nearest(logmvir, bins['logmvir']['grid'])

    table = _load_old_table(izl)
    if table is None:
        return config['fallback_er']

    #> contiguous 7x7 physical block
    dims = (bins['zs']['bins'],
            bins['logmvir']['bins'],
            bins['mstar_nsig']['bins'],
            bins['re_nsig']['bins'])

    start = np.ravel_multi_index((izs,imvir,0,0), dims)
    nblock = bins['mstar_nsig']['bins'] * bins['re_nsig']['bins']

    block = table.iloc[start:start+nblock]

    #> nearest physical Mstar/Re
    m = block['logM_star'].to_numpy(float)
    r = np.log10(block['R_eff'].to_numpy(float))
    er = pd.to_numeric(block['ER'], errors='coerce').to_numpy(float)

    valid = np.isfinite(m) & np.isfinite(r) & np.isfinite(er) & (er > 0)
    if not np.any(valid):
        return config['fallback_er']

    dm = (m[valid]-logmstar) / 0.25
    dr = (r[valid]-np.log10(reff)) / 0.25
    dist = dm**2 + dr**2

    seed = er[valid][np.argmin(dist)]

    return float(np.clip(seed,
                         config['seed_er_min'],
                         config['seed_er_max']))


""" #> PROFILE GENERATION =============
================================== """

#> supports old/new genGalPop signatures
def _gen_lenses(bprofiles):

    generate = _worker['generate']
    nph = _worker['config']['nph']

    first_arg = next(iter(inspect.signature(generate.genGalPop).parameters))

    if first_arg == 'numGals':
        return generate.genGalPop(len(bprofiles),
                                  bprofiles=bprofiles,
                                  nph=nph)

    return generate.genGalPop(redshifts=[],
                              bprofiles=bprofiles,
                              nph=nph)


#> changes only the angular grid scale
def _set_pix_arc(bprofiles, pix_arc, inds=None):

    if inds is None:
        inds = np.arange(len(bprofiles))

    out = []
    for i, p in zip(inds, pix_arc):
        prof = deepcopy(bprofiles[int(i)])
        prof['pix_arc'] = float(p)
        out.append(prof)

    return out


#> builds all physical models for one zl/zs cell
def _build_models(z_l_bin, z_s_bin):

    params = _worker['params']
    priors = _worker['priors']
    u = _worker['u']

    bins = _worker['bins']
    config = _worker['config']

    galProfs = _worker['galProfs']
    priorDict = _worker['priorDict']

    h_shuntov = 0.7

    #> fresh state
    paramRanges = params.toggleParams(galProfs)
    paramRanges['zl'] = z_l_bin
    paramRanges['zs'] = z_s_bin

    #> declarations
    physical = z_s_bin > z_l_bin

    indices = []
    model_params = []
    bprofiles = []
    seed_ers = []

    #> mvir
    for imvir, logmvir_bin in enumerate(bins['logmvir']['grid']):

        M_virs = np.array([10**logmvir_bin])
        zls = np.array([z_l_bin])

        #> halo
        paramRanges['nfw'][0]['logmass']['init'] = logmvir_bin

        c = priors.mass_c_vir_rel(M_virs,
                                  zls,
                                  cosmo=u.cosmo)[0]

        paramRanges['nfw'][0]['concentration']['init'] = c

        #> stellar mass
        logM_star, sigma_logM = priors.stellar_h_mass_rel(M_virs,
                                                           zls,
                                                           return_sample=False)[0]

        for imstar, mstar_nsig in enumerate(bins['mstar_nsig']['grid']):

            sample_logM_star_h2 = logM_star + mstar_nsig*sigma_logM
            sample_logM_star = np.log10(10**sample_logM_star_h2 / h_shuntov**2)

            paramRanges['hern'][0]['logmass']['init'] = sample_logM_star

            M_stars = np.array([10**sample_logM_star])

            #> effective radius
            mean_logR, sigma_logR = priors.mass_r_relation_18mowla(M_stars,
                                                                    zls,
                                                                    return_sample=False)[0]

            for ire, re_nsig in enumerate(bins['re_nsig']['grid']):

                sample_logR = mean_logR + re_nsig*sigma_logR
                reff = 10**sample_logR

                paramRanges['hern'][0]['effrad']['init'] = reff

                #> old ER estimate
                seed_er = _old_seed_er(z_l_bin,
                                       z_s_bin,
                                       logmvir_bin,
                                       sample_logM_star,
                                       reff)

                #> broad initial scale
                pix_arc = config['seed_pix_low'] / seed_er
                pix_arc = np.clip(pix_arc,
                                  config['pix_arc_min'],
                                  config['pix_arc_max'])

                paramRanges['pix_arc'] = pix_arc

                #> physical model
                if physical:
                    batch = params.bprofiles(numgals=1,
                                             ranges=paramRanges,
                                             priorDict=priorDict,
                                             verbose=False)

                    bprofiles.append(batch[0])
                else:
                    bprofiles.append(None)

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

#> centered-source ER
def _measure_er(xgrid, ygrid, delx, dely, pix_arc):

    lensing = _worker['lensing']
    geometry = _worker['geometry']
    u = _worker['u']
    nph = _worker['config']['nph']

    #> images
    args = (xgrid/pix_arc/u.arc_rad,
            ygrid/pix_arc/u.arc_rad,
            delx,
            dely,
            0.0,
            0.0,
            pix_arc,
            nph)

    try:
        ims = lensing.lfims(*args, ordered=True)
    except TypeError:
        ims = lensing.lfims(*args)

    if len(ims) != 5:
        return np.nan

    obs = geometry.lensObs(origin=(0,0),
                           images=ims,
                           observables=['d01','d02','d03','d04'])

    return float(np.mean(obs))


#> all lookup measurements for a batch
def _measure(bprofiles):

    lensing = _worker['lensing']

    #> lens maps
    lenses, bprofiles = _gen_lenses(bprofiles)
    xgrid, ygrid, delx, dely, lamt = lenses

    n = len(bprofiles)

    ER = np.full(n, np.nan)
    darea = np.zeros(n)
    oarea = np.zeros(n)
    pix_arc = np.empty(n)

    #> measurements
    for i in range(n):

        pa = float(bprofiles[i]['pix_arc'])
        pix_arc[i] = pa

        #> caustic areas
        try:
            dcaustic, ocaustic = lensing.caustics(xgrid,
                                                  ygrid,
                                                  delx[i],
                                                  dely[i],
                                                  pa)

            da, oa = lensing.causticArea(dcaustic, ocaustic)

            if np.isfinite(da):
                darea[i] = da

            if np.isfinite(oa):
                oarea[i] = oa

        except Exception:
            darea[i] = 0.0
            oarea[i] = 0.0

        #> einstein radius
        try:
            ER[i] = _measure_er(xgrid,
                                ygrid,
                                delx[i],
                                dely[i],
                                pa)
        except Exception:
            ER[i] = np.nan

    return {'ER': ER,
            'd_area': darea,
            'o_area': oarea,
            'pix_arc': pix_arc}


#> relative difference
def _reldiff(a, b, floor=1e-8):
    return np.abs(a-b) / np.maximum(np.maximum(np.abs(a),np.abs(b)),floor)


#> picks best bracket scale for each model
def _pick_bracket(low, high):

    config = _worker['config']
    target = config['target_pix']

    n = len(low['ER'])

    choice = np.zeros(n, dtype=int)

    low_valid = np.isfinite(low['ER']) & (low['ER'] > 0)
    high_valid = np.isfinite(high['ER']) & (high['ER'] > 0)

    #> if only one works
    choice[~low_valid & high_valid] = 1

    #> if both work, choose radius closest to target pixels
    both = low_valid & high_valid
    if np.any(both):

        low_pix = low['ER'][both] * low['pix_arc'][both]
        high_pix = high['ER'][both] * high['pix_arc'][both]

        use_high = np.abs(high_pix-target) < np.abs(low_pix-target)
        choice[np.where(both)[0][use_high]] = 1

    #> selected results
    out = {}
    for key in low:
        out[key] = np.where(choice == 0, low[key], high[key])

    out['valid'] = low_valid | high_valid

    return out


""" #> ADAPTIVE SOLVER ================
================================== """

#> solves one physical model batch
def _adaptive_measure(bprofiles, seed_ers):

    config = _worker['config']
    n = len(bprofiles)

    #> bracket around old estimate
    pix_low = np.clip(config['seed_pix_low']/seed_ers,
                      config['pix_arc_min'],
                      config['pix_arc_max'])

    pix_high = np.clip(config['seed_pix_high']/seed_ers,
                       config['pix_arc_min'],
                       config['pix_arc_max'])

    low = _measure(_set_pix_arc(bprofiles, pix_low))
    high = _measure(_set_pix_arc(bprofiles, pix_high))

    prev = _pick_bracket(low, high)

    #> output state
    final = {key: np.array(prev[key], copy=True)
             for key in ['ER','d_area','o_area','pix_arc']}

    n_pass = np.full(n, 2, dtype=int)
    converged = np.zeros(n, dtype=bool)

    er_err = np.full(n, np.nan)
    da_err = np.full(n, np.nan)

    #> adaptive passes
    for ipass in range(3, config['max_passes']+1):

        active = np.where(prev['valid'] & ~converged)[0]
        if len(active) == 0:
            break

        #> place current ER at target pixel radius
        pix = config['target_pix'] / prev['ER'][active]
        pix = np.clip(pix,
                      config['pix_arc_min'],
                      config['pix_arc_max'])

        trial_profiles = _set_pix_arc(bprofiles, pix, inds=active)
        trial = _measure(trial_profiles)

        #> current errors
        valid = np.isfinite(trial['ER']) & (trial['ER'] > 0)

        er_rel = np.full(len(active), np.inf)
        da_rel = np.full(len(active), np.inf)

        er_rel[valid] = _reldiff(trial['ER'][valid],
                                 prev['ER'][active][valid])

        da_rel[valid] = _reldiff(trial['d_area'][valid],
                                 prev['d_area'][active][valid],
                                 floor=config['darea_floor'])

        #> tiny cross sections do not force extra passes
        tiny_da = (np.maximum(trial['d_area'],
                              prev['d_area'][active]) < config['darea_floor'])

        da_ok = (da_rel <= config['darea_tol']) | tiny_da
        er_ok = er_rel <= config['er_tol']

        #> ensure useful pixel sampling / FOV
        er_pix = trial['ER'] * trial['pix_arc']
        pix_ok = ((er_pix >= config['min_er_pix']) &
                  (er_pix <= config['max_er_frac']*config['nph']))

        done = valid & er_ok & da_ok & pix_ok

        #> do not repeat the same failed adaptive scale
        failed = active[~valid]
        if len(failed):
            prev['valid'][failed] = False

        #> update final values whenever trial is valid
        good = np.where(valid)[0]
        if len(good):

            ginds = active[good]

            for key in final:
                final[key][ginds] = trial[key][good]

            er_err[ginds] = er_rel[good]
            da_err[ginds] = da_rel[good]
            n_pass[ginds] = ipass

            #> current result becomes next reference
            for key in ['ER','d_area','o_area','pix_arc']:
                prev[key][ginds] = trial[key][good]

            prev['valid'][ginds] = True

        converged[active[done]] = True

    #> models that never returned a valid centered quad
    invalid = ~np.isfinite(final['ER']) | (final['ER'] <= 0)

    final['ER'][invalid] = np.nan
    final['d_area'][invalid] = np.maximum(final['d_area'][invalid],0)
    final['o_area'][invalid] = np.maximum(final['o_area'][invalid],0)

    #> diagnostics
    final['ER_pix'] = final['ER'] * final['pix_arc']
    final['ER_relerr'] = er_err
    final['d_area_relerr'] = da_err
    final['n_pass'] = n_pass
    final['converged'] = converged

    return final


""" #> ZS WORKER ======================
================================== """

#> generates one zs bin at fixed zl
def _generate_zs(job):

    #> unpacking
    izl, z_l_bin, izs, z_s_bin = job

    config = _worker['config']

    #> model definitions
    indices, model_params, bprofiles, seed_ers = _build_models(z_l_bin,
                                                                z_s_bin)

    n = len(indices)

    #> source must be behind lens
    if z_s_bin <= z_l_bin:

        result = {'ER': np.full(n,np.nan),
                  'd_area': np.zeros(n),
                  'o_area': np.zeros(n),
                  'pix_arc': np.full(n,np.nan),
                  'ER_pix': np.full(n,np.nan),
                  'ER_relerr': np.full(n,np.nan),
                  'd_area_relerr': np.full(n,np.nan),
                  'n_pass': np.zeros(n,dtype=int),
                  'converged': np.zeros(n,dtype=bool)}

    else:
        result = _adaptive_measure(bprofiles, seed_ers)

    #> stacking
    izs_col = np.full((n,1), izs)
    zl_col = np.full((n,1), z_l_bin)
    zs_col = np.full((n,1), z_s_bin)

    data = np.column_stack([
        izs_col,
        indices,
        zl_col,
        zs_col,
        model_params,
        result['d_area'],
        result['o_area'],
        result['ER'],
        result['pix_arc'],
        result['ER_pix'],
        result['ER_relerr'],
        result['d_area_relerr'],
        result['n_pass'],
        result['converged'].astype(int),
        seed_ers
    ])

    return data


""" #> OUTPUT =========================
================================== """

#> saves exact grids/config used by generator
def _save_metadata(out_dir, config):

    bins = look_up_bins_v2()

    metadata = {
        'version': 2,
        'zl_grid': bins['zl']['grid'].tolist(),
        'zs_grid': bins['zs']['grid'].tolist(),
        'logmvir_grid': bins['logmvir']['grid'].tolist(),
        'mstar_nsig_grid': bins['mstar_nsig']['grid'].tolist(),
        're_nsig_grid': bins['re_nsig']['grid'].tolist(),
        'nph': config['nph'],
        'seed_pix_low': config['seed_pix_low'],
        'seed_pix_high': config['seed_pix_high'],
        'target_pix': config['target_pix'],
        'max_passes': config['max_passes'],
        'er_tol': config['er_tol'],
        'darea_tol': config['darea_tol'],
        'pix_arc_min': config['pix_arc_min'],
        'pix_arc_max': config['pix_arc_max']
    }

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir,'lookup_grid_v2.json'),'w') as f:
        json.dump(metadata, f, indent=2)

    return


#> adds d_area normalization / quad probability
def _finalize_table(df, max_er=2.4):

    df['d_area_max'] = 0.0
    df['p_quad'] = 0.0

    for izs in df['izs'].unique():

        mask_zs = df['izs'] == izs
        mask_max = (mask_zs &
                    np.isfinite(df['ER']) &
                    (df['ER'] <= max_er))

        if np.any(mask_max):
            dmax = df.loc[mask_max,'d_area'].max()
        else:
            dmax = 0.0

        if not np.isfinite(dmax):
            dmax = 0.0

        df.loc[mask_zs,'d_area_max'] = dmax

        if dmax > 0:
            df.loc[mask_zs,'p_quad'] = np.clip(df.loc[mask_zs,'d_area']/dmax,
                                               0,
                                               1)

    return df


""" #> GENERATE =======================
================================== """

#> generates lookup tables
def function(workers=None, **kwargs):

    #> paths
    old_dir = kwargs.get('old_dir', './priors/look_up_tables/')
    out_dir = kwargs.get('out_dir', './priors/look_up_tables_v2/')

    #> configuration
    config = {
        'old_dir': old_dir,
        'out_dir': out_dir,
        'use_old': kwargs.get('use_old', True),

        #> grid
        'nph': kwargs.get('nph', 50),

        #> old ER bracketing
        'seed_pix_low': kwargs.get('seed_pix_low', 6.0),
        'seed_pix_high': kwargs.get('seed_pix_high', 24.0),
        'fallback_er': kwargs.get('fallback_er', 0.8),
        'seed_er_min': kwargs.get('seed_er_min', 0.10),
        'seed_er_max': kwargs.get('seed_er_max', 5.00),

        #> adaptive solution
        'target_pix': kwargs.get('target_pix', 20.0),
        'max_passes': kwargs.get('max_passes', 4),
        'er_tol': kwargs.get('er_tol', 0.03),
        'darea_tol': kwargs.get('darea_tol', 0.10),
        'darea_floor': kwargs.get('darea_floor', 1e-4),

        #> valid final scale
        'min_er_pix': kwargs.get('min_er_pix', 10.0),
        'max_er_frac': kwargs.get('max_er_frac', 0.65),
        'pix_arc_min': kwargs.get('pix_arc_min', 2.0),
        'pix_arc_max': kwargs.get('pix_arc_max', 200.0)
    }

    #> bins
    bins = look_up_bins_v2()

    #> output
    os.makedirs(out_dir, exist_ok=True)
    _save_metadata(out_dir, config)

    #> workers
    if workers is None:
        workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                     os.cpu_count() or 1))

    workers = max(1, min(workers, bins['zs']['bins']))
    print(f'> zs workers: {workers}')
    print(f'> nph: {config["nph"]}')
    print(f'> old tables: {old_dir}')
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

            results = list(pool.map(_generate_zs, jobs))

            #> full zl table
            data = np.vstack(results)

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
                       'converged',
                       'seed_ER']

            df = pd.DataFrame(data, columns=columns)

            #> integer columns
            int_cols = ['izs','ilogmvir','imstar_nsig','ire_nsig','n_pass']
            df[int_cols] = df[int_cols].astype(int)
            df['converged'] = df['converged'].astype(bool)

            #> d_area normalization
            df = _finalize_table(df,
                                 max_er=kwargs.get('max_er',2.4))

            #> save
            outFile = os.path.join(out_dir,
                                   f'model_zl_{z_l_bin:.2f}.parquet')

            df.to_parquet(outFile, index=False)

            frac = df['converged'].mean()
            valid = np.isfinite(df['ER']).mean()

            print(f'> valid ER: {valid:.3f}')
            print(f'> converged: {frac:.3f}')
            print(f'> saved {outFile}')

    return


""" #> MAIN ==========================
================================== """

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=None)

    parser.add_argument('--old-dir',
                        type=str,
                        default='./priors/look_up_tables/')

    parser.add_argument('--out-dir',
                        type=str,
                        default='./priors/look_up_tables_v2/')

    parser.add_argument('--nph',
                        type=int,
                        default=50)

    parser.add_argument('--no-old',
                        action='store_true')

    args = parser.parse_args()

    print('> '+os.path.basename(__file__),'\n')

    function(workers=args.workers,
             old_dir=args.old_dir,
             out_dir=args.out_dir,
             nph=args.nph,
             use_old=not args.no_old)

# end
# thank
