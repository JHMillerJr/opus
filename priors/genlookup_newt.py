#> name: genlookup_newt.py
#> author: John Miller Jr
#> descrp: generates the strong-lensing look-up tables (ER, caustic areas, quad probability) with the newton code
#>         (profiles + newtlens) for one halo mass definition: mdef='vir' --> newt_tables_cvir, mdef='200c' --> newt_tables_c200
#>         the old v3 tables are only used to estimate each model's starting pixel scale

""" #> IMPORTS =======================
================================== """

#> standard imports
import io
import os
import sys
import json
import time
import argparse
import contextlib

#> one thread per worker (set before numpy, avoids mkl oversubscription w/ multiprocessing)
for key in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


""" #> DECLARATIONS ==================
================================== """

#> directories + files
outDirs     = {'vir':  './priors/newt_tables_cvir/',    # delta_vir(z), c_vir relation, M_vir axis
               '200c': './priors/newt_tables_c200/'}    # delta = 200,  c_200 relation, M_200 axis
validateDir = './priors/newt_tables_validate/'          # validation tables (old physics, v3 grid)
seedDir     = './priors/old_tables/look_up_tables_v3/'  # starting pixel scales (v3 tables)
v3File      = 'lookup_grid_v3.json'                     # v3 grid metadata (seed + validation)
gridFile    = 'lookup_grid_newt.json'                   # newt grid metadata
gridKeys    = ['zl', 'zs', 'logmvir', 'mstar_nsig', 're_nsig']

#> physics
h_shuntov = 0.7                                         # reduced hubble param of 2022 Shuntov

#> columns of the tables
columns = ['izs', 'ilogmvir', 'imstar_nsig', 'ire_nsig',                              # grid indices
           'zl', 'zs', 'logM_vir', 'c', 'logM_star', 'R_eff', 'mstar_nsig', 're_nsig', # model (logM_vir/c: mdef halo mass/conc.)
           't_area', 'r_area', 'nc_area', 'q_area', 'ER', 'pix_arc', 'ER_pix',          # measurements [arcsec^2, arcsec, pix/arcsec, pix]
           'ER_relerr', 'q_area_relerr', 'n_pass', 'ER_converged', 'q_area_converged',  # convergence
           'converged', 'window_retries', 'window_failed', 'n_tcurve', 'n_rcurve',      # diagnostics
           'multi_curve', 'nims', 'seed_ER']
int_cols  = ['izs', 'ilogmvir', 'imstar_nsig', 'ire_nsig', 'n_pass', 'window_retries', 'n_tcurve', 'n_rcurve', 'nims']
bool_cols = ['ER_converged', 'q_area_converged', 'converged', 'window_failed', 'multi_curve']

#> result of an unphysical model (zs <= zl)
unphysical = {'t_area':           np.nan,
              'r_area':           np.nan,
              'nc_area':          np.nan,
              'q_area':           np.nan,
              'ER':               np.nan,
              'pix_arc':          np.nan,
              'ER_relerr':        np.nan,
              'q_area_relerr':    np.nan,
              'n_pass':           0,
              'ER_converged':     False,
              'q_area_converged': False,
              'window_retries':   0,
              'window_failed':    False,
              'n_tcurve':         0,
              'n_rcurve':         0,
              'nims':             0}


""" #> GRID ==========================
================================== """

#> look-up grid (zl, zs denser at low redshift; halo mass; sigma offsets from the shmr & msr)
def look_up_bins(nzl=30, nzs=30, nmvir=60, nsig=13):

    #> lens redshift, denser at low zl
    t = np.linspace(0, 1, nzl)
    zl_grid = 0.01 + (1.8 - 0.01)*t**1.7

    #> source redshift, denser at low zs
    t = np.linspace(0, 1, nzs)
    zs_grid = 0.10 + (5.0 - 0.10)*t**1.3

    #> halo mass [log10(solMass)] (M_vir or M_200, see mdef)
    logmvir_grid = np.linspace(11.5, 14.5, nmvir)

    #> sigma offsets from the shmr (mstar) & msr (re)
    nsig_grid = np.linspace(-3.0, 3.0, nsig)

    return {'zl':         np.round(zl_grid, 8),
            'zs':         np.round(zs_grid, 8),
            'logmvir':    np.round(logmvir_grid, 8),
            'mstar_nsig': np.round(nsig_grid, 8),
            're_nsig':    np.round(nsig_grid, 8)}


#> the old v3 grid
def look_up_bins_v3(loc=seedDir):

    #> metadata
    with open(os.path.join(loc, v3File)) as f: meta = json.load(f)

    return {key: np.asarray(meta[f'{key}_grid'], dtype=float) for key in gridKeys}


#> checks the grid
def check_grid(bins):

    #> checking increasing
    for key in bins:
        if np.any(np.diff(bins[key]) <= 0):
            raise ValueError(f'{key} look-up grid is not strictly increasing')
    names = [f'{zl:.2f}' for zl in bins['zl']]

    #> ensuring no duplicate names
    if len(names) != len(set(names)):
        raise ValueError('zl grid has duplicate file names after rounding to 2 decimals')
    return


""" #> CONFIG ========================
================================== """

#> generator settings for one halo mass definition
def make_config(**kwargs):

    #> kwargs
    mdef     = kwargs.get('mdef', 'vir')            # halo mass definition: 'vir' or '200c'
    validate = kwargs.get('validate', False)        # validation run

    return {'mdef':     '200c' if (mdef == '200c' or validate) else 'vir',         # density contrast (params.bprofiles; validation: delta = 200 like v3)
            'cmr':      'vir' if (mdef == 'vir' or validate) else '200c',          # concentration relation
            'out_dir':  kwargs.get('out_dir', validateDir if validate else outDirs[mdef]),
            'seed_dir': kwargs.get('seed_dir', seedDir),
            'use_seed': kwargs.get('use_seed', True),

            #> grid
            'nph': kwargs.get('nph', 50),                        # grid half-width [pix]

            #> seed ER
            'seed_pix_wide': kwargs.get('seed_pix_wide', 8.0),   # starting ER [pix] (wide field)
            'seed_pix_fine': kwargs.get('seed_pix_fine', 24.0),  # starting ER [pix] (fine field)
            'fallback_er': kwargs.get('fallback_er', 0.8),       # arcsec
            'seed_er_min': kwargs.get('seed_er_min', 0.05),      # arcsec
            'seed_er_max': kwargs.get('seed_er_max', 8.0),       # arcsec
            'seed_mstar_scale': kwargs.get('seed_mstar_scale', 0.25),
            'seed_re_scale': kwargs.get('seed_re_scale', 0.25),
            'seed_table_cache': kwargs.get('seed_table_cache', 2),

            #> adaptive solution
            'target_pix': kwargs.get('target_pix', 18.0),        # target ER [pix]
            'max_passes': kwargs.get('max_passes', 4),
            'er_tol': kwargs.get('er_tol', 0.02),                # ER convergence (relative)
            'area_tol': kwargs.get('area_tol', 0.07),            # q_area convergence (relative)
            'area_floor': kwargs.get('area_floor', 1e-4),        # q_area below this counts as converged [arcsec^2]

            #> valid scale
            'min_er_pix': kwargs.get('min_er_pix', 10.0),        # min ER [pix]
            'max_er_frac': kwargs.get('max_er_frac', 0.65),      # max ER [fraction of nph]
            'pix_arc_min': kwargs.get('pix_arc_min', 0.5),       # pix/arcsec
            'pix_arc_max': kwargs.get('pix_arc_max', 250.0),     # pix/arcsec

            #> window recovery
            'window_retries': kwargs.get('window_retries', 4),
            'window_scale': kwargs.get('window_scale', 0.60),
            'curve_edge_pad': kwargs.get('curve_edge_pad', 2.0)} # pix


""" #> WORKERS =======================
================================== """

#> function to initialize worker
_worker = {}
def init_worker(config, bins):

    #> imports (each worker)
    import params
    import priors
    import profiles
    import newtlens
    import generate
    from modules.units import u; u=u()

    #> worker-local objects
    _worker.update({'params':      params,
                    'priors':      priors,
                    'profiles':    profiles,
                    'newtlens':    newtlens,
                    'u':           u,
                    'galProfs':    generate.galProfiles(nfw=True, hern=True, mult=[], ex=False), # nfw + hernquist only
                    'priorDict':   params.getPriorDict(all=False),                               # no random draws
                    'config':      config,
                    'bins':        bins,
                    'seed_bins':   load_seed_bins(config),
                    'seed_tables': OrderedDict()})
    return


""" #> SEED (V3 TABLES) ==============
================================== """

#> v3 grid for the seed tables (None if not used / not found)
def load_seed_bins(config):
    if not config['use_seed']: return None
    if not os.path.exists(os.path.join(config['seed_dir'], v3File)): return None
    return look_up_bins_v3(config['seed_dir'])


#> one seed zl table (logM_star, R_eff, ER), small cache
def load_seed_table(izl):

    #> declarations
    cache  = _worker['seed_tables']
    config = _worker['config']
    bins   = _worker['seed_bins']

    #> cached
    if izl in cache:
        cache.move_to_end(izl)
        return cache[izl]

    #> loading
    fileName = os.path.join(config['seed_dir'], f"model_zl_{bins['zl'][izl]:.2f}.parquet")
    if not os.path.exists(fileName): return None
    cache[izl] = pd.read_parquet(fileName, columns=['logM_star', 'R_eff', 'ER'])

    #> dropping the oldest
    while len(cache) > config['seed_table_cache']: cache.popitem(last=False)

    return cache[izl]


#> (mstar_nsig, re_nsig) block of one seed cell: log10 M_star, log10 R_eff, ER, valid
def seed_block(table, bins, izs, ih):

    #> rows of the block
    nm, nr = len(bins['mstar_nsig']), len(bins['re_nsig'])
    dims   = (len(bins['zs']), len(bins['logmvir']), nm, nr)
    start  = np.ravel_multi_index((izs, ih, 0, 0), dims)
    block  = table.iloc[start:start + nm*nr]

    #> values
    mstar = block['logM_star'].to_numpy(float)                                # log10(solMass)
    reff  = block['R_eff'].to_numpy(float)                                    # kpc
    logre = np.log10(np.where(reff > 0, reff, np.nan))                        # log10(kpc)
    er    = pd.to_numeric(block['ER'], errors='coerce').to_numpy(float)       # arcsec
    valid = np.isfinite(mstar) & np.isfinite(logre) & np.isfinite(er) & (er > 0)

    return mstar, logre, er, valid


#> starting einstein radius [arcsec] from the nearest v3 model (fallback_er if none)
def seed_er(zl, zs, logmvir, logmstar, reff):

    #> declarations
    config, bins = _worker['config'], _worker['seed_bins']
    if bins is None: return config['fallback_er']

    #> nearest seed cell
    near  = lambda x, g: int(np.abs(g - x).argmin())
    izl   = near(zl, bins['zl'])
    izs   = near(zs, bins['zs'])
    ih    = near(logmvir, bins['logmvir'])
    table = load_seed_table(izl)
    if table is None: return config['fallback_er']

    #> block of (mstar_nsig, re_nsig) models in that cell
    mstar, logre, er, valid = seed_block(table, bins, izs, ih)
    if not np.any(valid): return config['fallback_er']

    #> nearest physical (mstar, re) model
    dm   = (mstar[valid] - logmstar)/config['seed_mstar_scale']
    dr   = (logre[valid] - np.log10(reff))/config['seed_re_scale']
    best = er[valid][np.argmin(dm**2 + dr**2)]                                # arcsec

    return float(np.clip(best, config['seed_er_min'], config['seed_er_max']))


""" #> MODELS ========================
================================== """

#> concentration (mean of the cmr) of a halo [log10(solMass)] at zl
def concentration(logmvir, zl):

    #> worker objects
    priors, u, config = _worker['priors'], _worker['u'], _worker['config']

    #> declarations
    M_halos = np.array([10**logmvir])                                         # solMass
    zls     = np.array([zl])

    #> cmr
    if config['cmr'] == 'vir': c = priors.mass_c_vir_rel(M_halos, zls, cosmo=u.cosmo)
    else: c = priors.mass_c_200_rel(M_halos, zls, cosmo=u.cosmo)

    return float(np.atleast_1d(c)[0])


#> stellar-to-halo mass relation (mean + scatter) [log10(solMass/h^2)]
def shmr(logmvir, zl):
    M_halos = np.array([10**logmvir])                                         # solMass
    return _worker['priors'].stellar_h_mass_rel(M_halos, np.array([zl]), return_sample=False)[0]


#> stellar mass [log10(solMass)] mstar_nsig sigma from the shmr mean
def stellar_mass(logM_star_h2, sigma_logM, mstar_nsig):
    return np.log10(10**(logM_star_h2 + mstar_nsig*sigma_logM) / h_shuntov**2)


#> mass-size relation (mean + scatter) [log10(kpc)]
def msr(logM_star, zl):
    M_stars = np.array([10**logM_star])                                       # solMass
    return _worker['priors'].mass_r_relation_18mowla(M_stars, np.array([zl]), return_sample=False)[0]


#> galaxy profile of the current parameter state
def galaxy_profile(ranges, zl, zs):

    #> worker objects
    params, u, config = _worker['params'], _worker['u'], _worker['config']

    return params.bprofiles(1, ranges=ranges, priorDict=_worker['priorDict'],
                            redshifts=np.array([[zl, zs]]), mdef=config['mdef'],
                            cosmo=u.cosmo)[0]


#> galaxy models of one (zl, zs) cell: nfw (mvir, c from the cmr) + hernquist (mstar, re from the shmr & msr)
def build_models(zl, zs):

    #> worker objects
    bins = _worker['bins']

    #> fresh parameter state
    ranges = _worker['params'].toggleParams(_worker['galProfs'])
    models, bprofiles, seeds = [], [], []

    #> halo mass
    for ih, logmvir in enumerate(bins['logmvir']):

        #> nfw: halo mass + concentration
        c = concentration(logmvir, zl)
        ranges['nfw'][0]['logmass']['init'] = logmvir
        ranges['nfw'][0]['concentration']['init'] = c

        #> shmr at this halo mass
        logM_star_h2, sigma_logM = shmr(logmvir, zl)

        #> stellar mass
        for im, mstar_nsig in enumerate(bins['mstar_nsig']):

            logM_star = stellar_mass(logM_star_h2, sigma_logM, mstar_nsig)  # log10(solMass)
            ranges['hern'][0]['logmass']['init'] = logM_star

            #> msr at this stellar mass
            mean_logR, sigma_logR = msr(logM_star, zl)

            #> effective radius
            for ir, re_nsig in enumerate(bins['re_nsig']):

                reff = 10**(mean_logR + re_nsig*sigma_logR)                   # kpc
                ranges['hern'][0]['effrad']['init'] = reff

                #> galaxy profile (only physical redshifts are lensed)
                if zs > zl:
                    bprofiles.append(galaxy_profile(ranges, zl, zs))
                    seeds.append(seed_er(zl, zs, logmvir, logM_star, reff))
                else:
                    bprofiles.append(None)
                    seeds.append(np.nan)

                #> model parameters
                models.append([ih, im, ir, logmvir, c, logM_star, reff, mstar_nsig, re_nsig])

    return np.asarray(models, dtype=float), bprofiles, np.asarray(seeds, dtype=float)


""" #> MEASUREMENTS ==================
================================== """

#> lens-plane grid (same as generate.genQuadNewt) [arcsec]
def lens_grid(pix_arc):
    nph = _worker['config']['nph']
    w = np.arange(-nph, nph, 1, dtype=float) / pix_arc   # arcsec
    return np.meshgrid(w, w)                             # arcsec


#> caustic areas [arcsec^2], curve counts, and the critical-curve einstein radius (area = pi ER^2) [arcsec]
def caustic_measures(caus, pix_arc):

    #> declarations
    newtlens = _worker['newtlens']

    #> areas + curves
    m = {'t_area':   newtlens.t_area(caus),                 # arcsec^2
         'r_area':   newtlens.r_area(caus),                 # arcsec^2
         'nc_area':  newtlens.nc_area(caus),                # arcsec^2
         'q_area':   newtlens.q_area(caus),                 # arcsec^2
         'n_tcurve': caus['n_tcurve'],
         'n_rcurve': caus['n_rcurve'],
         'edge':     caus['edge'],
         'ER':       np.nan,
         'nims':     0,
         'pix_arc':  pix_arc}

    #> no tangential critical curve --> no quads
    if caus['tcrit'] is None: m |= {'t_area': 0.0, 'nc_area': 0.0, 'q_area': 0.0}

    #> einstein radius estimate from the tangential critical curve
    m['ER_cc'] = np.sqrt(newtlens.poly_area(caus['tcrit'])/np.pi) if caus['tcrit'] is not None else np.nan

    return m


#> einstein radius: mean radius of the 4 outer images of a source at the lens center [arcsec]
def einstein_radius(xg, yg, jac, bprof):

    #> images of beta = 0
    with contextlib.redirect_stdout(io.StringIO()):
        res = _worker['newtlens'].find_images(xg, yg, xg - jac[0], yg - jac[1], np.zeros((1,2)), bprof)
    nims = int(res['nims'][0])
    ims  = res['images'][0][:nims]                                          # arcsec

    #> outer 4 images
    if nims != 5: return np.nan, nims
    r = np.sort(np.hypot(ims[:,0], ims[:,1]))                               # arcsec

    return float(r[1:].mean()), nims


#> measures er w/ beta=0, and caustic areas
def measure(bprof, pix_arc):

    #> worker objects
    profiles, newtlens, config = _worker['profiles'], _worker['newtlens'], _worker['config']

    #> grid + deflections + jacobian
    xg, yg = lens_grid(pix_arc)
    jac = profiles.alpha_jac_points(xg, yg, bprof)

    #> critical curves + caustic areas
    caus = newtlens.crit_caustics(xg, yg, bprof, jac=jac, pad=config['curve_edge_pad'])
    m = caustic_measures(caus, pix_arc)
    if m['edge']: return m                                                  # curve leaves the window --> wider field

    #> einstein radius
    m['ER'], m['nims'] = einstein_radius(xg, yg, jac, bprof)

    return m


#> true if the einstein radius is too large in pixels for the window
def er_too_large(m, pix):
    config = _worker['config']
    return np.isfinite(m['ER']) and m['ER']*pix > config['max_er_frac']*config['nph']


#> measures at pix_arc, widening the field if a curve leaves the window or the ER is too large in pixels
def measure_with_recovery(bprof, pix_arc):

    #> declarations
    config = _worker['config']
    pix    = float(np.clip(pix_arc, config['pix_arc_min'], config['pix_arc_max']))

    #> widening retries
    for retry in range(config['window_retries'] + 1):
        m = measure(bprof, pix)
        if not (m['edge'] or er_too_large(m, pix)): return m | {'window_retries': retry, 'window_failed': False}

        #> wider field (fewer pix/arcsec)
        new = max(config['pix_arc_min'], pix*config['window_scale'])
        if new >= pix: break
        pix = new

    return m | {'window_retries': retry, 'window_failed': True}


#> valid einstein radius
def valid(m):
    return np.isfinite(m['ER']) and m['ER'] > 0 and not m['window_failed']


#> relative difference
def reldiff(a, b, floor=1e-8):
    return abs(a - b) / max(abs(a), abs(b), floor)


""" #> SOLVE =========================
================================== """

#> starting bracket (wide, fine) around the seed ER, keeps the one closest to the target
def start_bracket(bprof, seed):

    #> declarations
    config, target = _worker['config'], _worker['config']['target_pix']

    #> two scales
    trials = [measure_with_recovery(bprof, config[key]/seed) for key in ['seed_pix_wide', 'seed_pix_fine']]
    good   = [m for m in trials if valid(m)]
    prev   = min(good, key=lambda m: abs(m['ER']*m['pix_arc'] - target)) if good else trials[-1]

    #> starting output
    out = prev | {'n_pass':           2,
                  'ER_relerr':        np.nan,
                  'q_area_relerr':    np.nan,
                  'ER_converged':     False,
                  'q_area_converged': False,
                  'window_retries':   sum(m['window_retries'] for m in trials)}

    return trials, good, prev, out


#> no einstein radius at either scale (images unresolved): re-target from the critical-curve size
def retarget_from_cc(bprof, trials, out):

    #> critical-curve einstein radii
    cc = [m for m in trials if np.isfinite(m['ER_cc']) and m['ER_cc'] > 0]
    if not cc: return None, out                                             # no tangential curve --> not a strong lens

    #> measuring at the re-targeted scale
    m   = measure_with_recovery(bprof, _worker['config']['target_pix']/cc[-1]['ER_cc'])
    out = out | m | {'n_pass': 3, 'window_retries': out['window_retries'] + m['window_retries']}

    return m, out


#> convergence of one pass (ER + quad area) relative to the previous pass
def pass_status(m, prev):

    #> declarations
    config = _worker['config']
    floor  = config['area_floor']

    #> relative changes
    er_rel = reldiff(m['ER'], prev['ER'])
    finite = np.isfinite(m['q_area']) and np.isfinite(prev['q_area'])
    qa_rel = reldiff(m['q_area'], prev['q_area'], floor=floor) if finite else np.inf

    #> tiny quad area + ER pixel radius in range
    tiny   = np.isfinite(qa_rel) and max(m['q_area'], prev['q_area']) < floor
    pix_ok = config['min_er_pix'] <= m['ER']*m['pix_arc'] <= config['max_er_frac']*config['nph']

    return {'ER_relerr':        er_rel,
            'q_area_relerr':    qa_rel,
            'ER_converged':     bool(er_rel <= config['er_tol'] and pix_ok),
            'q_area_converged': bool((qa_rel <= config['area_tol'] or tiny) and pix_ok)}


#> adaptive pixel scale: starts at two scales around the seed ER, then re-targets ER --> target_pix until converged
def solve(bprof, seed):

    #> declarations
    config, target = _worker['config'], _worker['config']['target_pix']

    #> starting bracket
    trials, good, prev, out = start_bracket(bprof, seed)
    first = 3

    #> no einstein radius at either scale
    if not good:
        m, out = retarget_from_cc(bprof, trials, out)
        if m is None or not valid(m): return out
        prev, first = m, 4

    #> adaptive passes
    for ipass in range(first, config['max_passes'] + 1):

        #> re-targeting the ER pixel radius
        m = measure_with_recovery(bprof, target/prev['ER'])
        out['window_retries'] += m['window_retries']
        if not valid(m): break

        #> saving
        out |= m | {'n_pass': ipass} | pass_status(m, prev) | {'window_retries': out['window_retries']}
        prev = m
        if out['ER_converged'] and out['q_area_converged']: break

    return out


""" #> ONE (ZL, ZS) CELL =============
================================== """

#> one table row of a solved model
def table_row(izs, zl, zs, model, r, seed):

    #> derived
    ER_pix    = r['ER']*r['pix_arc']                                        # pix
    converged = r['ER_converged'] and r['q_area_converged']

    return [izs, *model[:3], zl, zs, *model[3:],                                                # indices + model
            r['t_area'], r['r_area'], r['nc_area'], r['q_area'], r['ER'], r['pix_arc'], ER_pix,  # measurements
            r['ER_relerr'], r['q_area_relerr'], r['n_pass'], r['ER_converged'], r['q_area_converged'],  # convergence
            converged, r['window_retries'], r['window_failed'], r['n_tcurve'], r['n_rcurve'],    # diagnostics
            r['n_tcurve'] > 1, r['nims'], seed]


#> measures every model of one (zl, zs) cell
def generate_cell(job):

    #> unpacking
    izl, zl, izs, zs = job
    models, bprofiles, seeds = build_models(zl, zs)

    #> each model (unphysical: zs <= zl)
    rows = []
    for model, bprof, seed in zip(models, bprofiles, seeds):
        r = unphysical if bprof is None else solve(bprof, seed)
        rows.append(table_row(izs, zl, zs, model, r, seed))

    return izl, izs, rows


""" #> ZL TABLE ======================
================================== """

#> dataframe of one zl table w/ integer + boolean columns
def to_dataframe(rows):
    df = pd.DataFrame(rows, columns=columns)
    df[int_cols], df[bool_cols] = df[int_cols].astype(int), df[bool_cols].astype(bool)
    return df


#> largest quad area of one zs (converged models w/ ER <= max_er, otherwise any finite model) [arcsec^2]
def quad_area_max(df, mask, max_er=2.4):

    #> models w/ a finite ER <= max_er
    base = mask & np.isfinite(df['ER']) & (df['ER'] <= max_er) & np.isfinite(df['q_area'])
    if not base.any(): return 0.0

    #> converged ones if any
    good = base & df['converged']
    qmax = df.loc[good if good.any() else base, 'q_area'].max()

    return qmax if np.isfinite(qmax) else 0.0


#> one zl table: dataframe + quad-area normalization per zs (q_area_max, p_quad)
def finalize_table(rows, max_er=2.4):

    #> dataframe
    df = to_dataframe(rows)
    df['q_area_max'], df['p_quad'] = 0.0, 0.0

    #> each zs
    for izs in df['izs'].unique():
        mask = df['izs'] == izs
        qmax = quad_area_max(df, mask, max_er)
        df.loc[mask, 'q_area_max'] = qmax
        if qmax > 0: df.loc[mask, 'p_quad'] = (df.loc[mask, 'q_area']/qmax).clip(0, 1).fillna(0)

    return df


#> one-line summary of a zl table (physical models)
def print_zl_summary(df, zl, outFile, srt):

    #> physical models
    phys = df['zs'] > df['zl']

    #> fractions
    er_ok  = np.isfinite(df['ER'][phys]).mean()
    conv   = df['converged'][phys].mean()
    w_fail = df['window_failed'][phys].mean()
    nc     = (df['nc_area'][phys] > 0).mean()

    print(f"> zl = {zl:.2f}: valid ER {er_ok:.3f}, converged {conv:.3f}, "
          f"window failed {w_fail:.3f}, naked cusps {nc:.3f}, "
          f"{(time.time() - srt)/60:.1f} min --> {outFile}")
    return


""" #> VALIDATION ====================
================================== """

#> compares a validation table (old physics, v3 grid) to the v3 table of the same zl
def compare_v3(df, zl, loc=seedDir):

    #> v3 table (same grid --> same row order)
    old = pd.read_parquet(os.path.join(loc, f'model_zl_{zl:.2f}.parquet'))
    if len(old) != len(df): raise ValueError(f'validation table has {len(df)} rows, v3 table has {len(old)}')

    #> rows valid in both
    ok = np.isfinite(df['ER']) & np.isfinite(old['ER']) & (old['ER'] > 0) & df['converged'] & old['converged']

    #> relative differences (ER, tangential caustic area vs the old d_area)
    er = (df['ER'] - old['ER'])[ok] / old['ER'][ok]
    da = ok & np.isfinite(old['d_area']) & (old['d_area'] > 0)
    ta = (df['t_area'] - old['d_area'])[da] / old['d_area'][ok & (old['d_area'] > 0)]

    #> summary
    stats = {'zl':                              zl,
             'n_both_valid':                    int(ok.sum()),
             'n_new_valid':                     int(np.isfinite(df['ER']).sum()),
             'n_old_valid':                     int(np.isfinite(old['ER']).sum()),
             'ER_median_rel':                   float(np.median(er)),
             'ER_median_abs_rel':               float(np.median(np.abs(er))),
             'ER_within_2pct':                  float(np.mean(np.abs(er) < 0.02)),
             'ER_within_5pct':                  float(np.mean(np.abs(er) < 0.05)),
             't_area_vs_d_area_median_rel':     float(np.median(ta)),
             't_area_vs_d_area_median_abs_rel': float(np.median(np.abs(ta))),
             't_area_within_7pct':              float(np.mean(np.abs(ta) < 0.07)),
             'nc_area_nonzero_frac':            float(np.mean(df['nc_area'][ok] > 0)),
             'no_rcurve_frac':                  float(np.mean(df['n_rcurve'][ok] == 0))}
    for k, v in stats.items(): print(f'>   {k:32s}: {v}')

    return stats


#> saves the validation summary
def save_validation(stats, config):
    fileName = os.path.join(config['out_dir'], 'validation_v3.csv')
    pd.DataFrame(stats).to_csv(fileName, index=False)
    print(f"> validation summary saved to {fileName}")
    return


""" #> SETUP =========================
================================== """

#> grid + zl bins to make (validation: the v3 grid + the zl bin nearest 0.5)
def make_grid(config, **kwargs):

    #> kwargs
    validate = kwargs.get('validate', False)        # validation run

    #> grid
    bins = look_up_bins_v3(config['seed_dir']) if validate else look_up_bins(**kwargs.get('grid', {}))
    check_grid(bins)

    #> zl bins
    default = [int(np.argmin(np.abs(bins['zl'] - 0.5)))] if validate else range(len(bins['zl']))
    zl_inds = kwargs.get('zl_inds', default)

    return bins, zl_inds


#> turns off the seed if the v3 tables are missing
def check_seed(config):
    if config['use_seed'] and not os.path.exists(os.path.join(config['seed_dir'], v3File)):
        print(f"> seed tables not found in {config['seed_dir']}, using fallback ER for the starting scales")
        config['use_seed'] = False
    return


#> writes the grid metadata (grids + config)
def write_metadata(config, bins, validate, cosmo):

    #> metadata
    meta = {'version':  'newt',
            'validate': validate,
            'cosmo':    cosmo,
            'columns':  columns + ['q_area_max', 'p_quad']}
    meta |= {f'{key}_grid': bins[key].tolist() for key in bins} | config

    #> saving
    os.makedirs(config['out_dir'], exist_ok=True)
    with open(os.path.join(config['out_dir'], gridFile), 'w') as f: json.dump(meta, f, indent=2)
    return


#> file of one zl table
def table_file(config, bins, izl):
    return os.path.join(config['out_dir'], f"model_zl_{bins['zl'][izl]:.2f}.parquet")


#> jobs: every (zl, zs) cell (skipping finished zl tables)
def make_jobs(config, bins, zl_inds, resume=True):
    todo = [izl for izl in zl_inds if not (resume and os.path.exists(table_file(config, bins, izl)))]
    jobs = [(izl, bins['zl'][izl], izs, bins['zs'][izs]) for izl in todo for izs in range(len(bins['zs']))]
    return todo, jobs


#> number of processes (slurm cpus or all cores, at most one per job)
def n_workers(workers, jobs):
    if workers is None: workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    return max(1, min(workers, max(len(jobs), 1)))


#> run header
def print_header(config, bins, zl_inds, todo, workers, validate):
    nmod = len(bins['logmvir'])*len(bins['mstar_nsig'])*len(bins['re_nsig'])
    print(f"> mdef: {config['mdef']} (cmr: c_{config['cmr']}){'  [VALIDATION]' if validate else ''}")
    print(f"> grid: {' x '.join(str(len(bins[k])) for k in bins)} ({nmod} models per cell)")
    print(f"> zl tables to make: {len(todo)} ({len(zl_inds) - len(todo)} already exist)")
    print(f"> workers: {workers}, nph: {config['nph']}, target ER: {config['target_pix']} pix")
    print(f"> tables: {config['out_dir']}\n")
    return


""" #> RUN ===========================
================================== """

#> multiprocessing over (zl, zs) cells; a zl table is written once all of its zs cells are done
def run_cells(jobs, config, bins, workers, validate=False, max_er=2.4):

    #> declarations
    done, stats, srt = {}, [], time.time()

    #> pool
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=init_worker, initargs=(config, bins)) as pool:

        futures = [pool.submit(generate_cell, job) for job in jobs]
        for fut in as_completed(futures):

            #> collecting
            izl, izs, rows = fut.result()
            done.setdefault(izl, {})[izs] = rows
            print(f">   zl {bins['zl'][izl]:.2f}, zs {bins['zs'][izs]:.2f}: {len(rows)} models, {(time.time() - srt)/60:.1f} min", flush=True)
            if len(done[izl]) < len(bins['zs']): continue

            #> full zl table (rows in izs order, then ilogmvir, imstar_nsig, ire_nsig)
            cells   = done.pop(izl)
            df      = finalize_table([row for i in sorted(cells) for row in cells[i]], max_er=max_er)
            outFile = table_file(config, bins, izl)
            df.to_parquet(outFile, index=False)
            print_zl_summary(df, bins['zl'][izl], outFile, srt)

            #> validation
            if validate: stats.append(compare_v3(df, bins['zl'][izl], config['seed_dir']))

    return stats, srt


""" #> GENERATE ======================
================================== """

#> generates the look-up tables for one halo mass definition (mdef = 'vir' or '200c')
def function(**kwargs):

    #> imports
    import modules.error as error
    from modules.units import u; u=u()

    #> kwargs
    mdef     = kwargs.get('mdef', 'vir')            # halo mass definition: 'vir' or '200c'
    validate = kwargs.get('validate', False)        # old physics (delta=200, c_vir) on the v3 grid, compared to v3
    workers  = kwargs.get('workers', None)          # number of processes
    resume   = kwargs.get('resume', True)           # skips zl tables that already exist
    max_er   = kwargs.get('max_er', 2.4)            # max ER for the quad-area normalization [arcsec]

    #> checking mdef
    if mdef not in outDirs: error.phrase(f"mdef={mdef} not understood. Please use 'vir' or '200c'")

    #> configuration
    config = make_config(**kwargs)

    #> shmr is calibrated on virial masses
    if config['mdef'] == '200c' and not validate:
        error.highlight('mdef=200c: the shmr (2022 Shuntov) is calibrated on VIRIAL halo masses, but is given M_200 here --> these tables may be inaccurate')

    #> grid + seed + metadata
    bins, zl_inds = make_grid(config, **kwargs)
    check_seed(config)
    write_metadata(config, bins, validate, u.cosmo)

    #> jobs + workers
    todo, jobs = make_jobs(config, bins, zl_inds, resume)
    workers = n_workers(workers, jobs)
    print_header(config, bins, zl_inds, todo, workers, validate)

    #> running
    stats, srt = run_cells(jobs, config, bins, workers, validate, max_er)

    #> validation summary
    if validate and stats: save_validation(stats, config)
    print(f'> done in {(time.time() - srt)/60:.1f} min')

    return


""" #> MAIN ==========================
================================== """

#> main function (run from the opus directory)
if __name__ == '__main__':

    #> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdef', type=str, default='vir', choices=['vir', '200c'])  # halo mass definition
    parser.add_argument('--validate', action='store_true')                          # old physics on the v3 grid vs the v3 tables
    parser.add_argument('--zl-inds', type=int, nargs='*', default=None)              # only these zl bins (index)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--nph', type=int, default=50)
    parser.add_argument('--target-pix', type=float, default=18.0)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--no-seed', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()

    #> name
    print('> '+os.path.basename(__file__),'\n')

    #> kwargs (only what was given)
    kwargs = {'mdef':       args.mdef,
              'validate':   args.validate,
              'workers':    args.workers,
              'nph':        args.nph,
              'target_pix': args.target_pix,
              'use_seed':   not args.no_seed,
              'resume':     not args.no_resume}
    if args.zl_inds is not None: kwargs['zl_inds'] = args.zl_inds
    if args.out_dir is not None: kwargs['out_dir'] = args.out_dir

    #> generating tables
    function(**kwargs)

    # end
# thank
