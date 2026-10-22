#> name: calib_qfactor.py
#> author: John Miller Jr
#> descrp: calibrates the axis-ratio correction f(q) = area(q) / area(q=0.85) for the newt look-up tables
#>         (tables are made at q=0.85; ER barely depends on q, the caustic areas do)

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import copy
import time
import json
import argparse

#> one thread per worker (set before numpy)
for key in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']: os.environ.setdefault(key, '1')

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

#> adding dir to sys paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#> generator
import genlookup_newt as gl

#> declarations
q_ref   = 0.85                                         # axis ratio of the tables []
q_grid  = np.round(np.arange(0.50, 0.951, 0.05), 2)    # calibration axis ratios []
cells   = [(0.2, 1.0), (0.2, 3.0), (0.5, 1.0), (0.5, 2.0),
           (0.5, 4.0), (1.0, 2.0), (1.0, 4.0), (1.5, 3.0)] # (zl, zs)
ids     = ['zl', 'zs', 'k']                            # model id columns
columns = ['zl', 'zs', 'k', 'logM_vir', 'c', 'logM_star', 'R_eff', 'mstar_nsig', 're_nsig',
           'q', 'ER', 't_area', 'r_area', 'nc_area', 'q_area', 'converged']

#> log10(rho) bin edges, rho = r_area/t_area at q_ref (small rho --> naked cusps at low q)
lrho_edges = np.array([-np.inf, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0, np.inf])


""" #> CONFIG ========================
================================== """

#> stops function() at the pool and returns its (config, bins)
class _Grab(Exception): pass
class _Capture:
    def __init__(self, *args, initializer=None, initargs=(), **kwargs): raise _Grab(initargs)


#> config + bins exactly as genlookup_newt.function() builds them
def get_config(**kwargs):

    #> swapping the pool for one that stops at its creation
    pool = gl.ProcessPoolExecutor
    gl.ProcessPoolExecutor = _Capture

    #> running until the pool
    try: gl.function(**(kwargs | {'workers': 1, 'resume': False}))
    except _Grab as e: return e.args[0]
    finally: gl.ProcessPoolExecutor = pool

    return None


""" #> WORKER ========================
================================== """

#> models of one cell (cached per worker)
_cells = {}
def cell_models(zl, zs):
    if (zl, zs) not in _cells: _cells[(zl, zs)] = gl.build_models(zl, zs)
    return _cells[(zl, zs)]


#> copy of a galaxy w/ axis ratio q (nfw + hernquist)
def set_axisrat(bprof, q):
    b = copy.deepcopy(bprof)
    for key in ('nfw', 'hern'): b[key][0]['axisrat'] = float(q)
    return b


#> one model at every q (q_ref first, its ER seeds the others)
def run_model(job):

    #> unpacking
    zl, zs, k = job
    models, bprofiles, seeds = cell_models(zl, zs)
    model, bprof = models[k], bprofiles[k]
    if bprof is None: return []

    #> declarations
    rows, seed = [], seeds[k]

    #> each axis ratio
    for q in [q_ref] + [q for q in q_grid if q != q_ref]:

        #> solving
        r = gl.solve(set_axisrat(bprof, q), seed)
        if q == q_ref: seed = r['ER']

        #> row
        converged = r['ER_converged'] and r['q_area_converged']
        rows.append([zl, zs, k, *model[3:], q,
                     r['ER'], r['t_area'], r['r_area'], r['nc_area'], r['q_area'], converged])

    return rows


""" #> CALIBRATION ===================
================================== """

#> jobs: nper random models in each cell (nearest grid cell)
def make_jobs(bins, nper=40, seed=1):

    #> declarations
    rng  = np.random.default_rng(seed)
    nmod = len(bins['logmvir'])*len(bins['mstar_nsig'])*len(bins['re_nsig'])
    near = lambda x, grid: grid[np.argmin(np.abs(grid - x))]

    #> each cell
    jobs = []
    for zl, zs in cells:
        zl, zs = near(zl, bins['zl']), near(zs, bins['zs'])
        jobs  += [(zl, zs, int(k)) for k in rng.choice(nmod, nper, replace=False)]

    return jobs


#> runs every job on a process pool
def run_jobs(jobs, config, bins, workers=None):

    #> workers
    if workers is None: workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    print(f'\n> {len(jobs)} models x {len(q_grid)} axis ratios, {workers} workers\n')

    #> running
    rows, srt = [], time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=gl.init_worker, initargs=(config, bins)) as pool:

        futures = [pool.submit(run_model, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures)):
            rows += fut.result()
            if (i+1) % 20 == 0: print(f'>   {i+1}/{len(jobs)} models, {(time.time() - srt)/60:.1f} min', flush=True)

    return rows


#> table + ratios to q_ref (same model)
def make_table(rows):

    #> table
    df = pd.DataFrame(rows, columns=columns)

    #> reference rows matched to every row
    ref = df[df['q'] == q_ref].set_index(ids)
    key = df.set_index(ids).index

    #> ratios
    for c in ['ER', 't_area', 'q_area']:
        df[f'f_{c}'] = df[c].to_numpy() / ref.loc[key, c].to_numpy()

    return df


#> reference models (q_ref) and the converged ones w/ a quad region
def reference(df, area_floor=1e-4):

    #> reference rows
    ref = df[df['q'] == q_ref].set_index(ids)

    #> good models
    good = ref.index[(ref['q_area'] > area_floor) & ref['converged'].astype(bool)]

    return ref, good


""" #> SUMMARY =======================
================================== """

#> median and 16-84% spread of f(q) over the good models
def print_summary(df, good):

    #> good models
    d = df.set_index(ids).loc[good].reset_index()

    #> header
    print(f'\n> f(q) = area(q)/area({q_ref}) over {len(good)} models  [median (16-84%)]\n')
    print(f"{'q':>5s} {'f_ER':>22s} {'f_t_area':>22s} {'f_q_area':>22s} {'nc > 0':>7s}")

    #> each q
    for q, g in d.groupby('q'):
        p    = {c: np.nanpercentile(g[c], [16, 50, 84]) for c in ['f_ER', 'f_t_area', 'f_q_area']}
        cols = ' '.join(f'{v[1]:8.3f} ({v[0]:.3f}-{v[2]:.3f})' for v in p.values())
        print(f"{q:5.2f} {cols} {(g['nc_area'] > 0).mean():7.2f}")

    return


#> naked cusps: f_q at the lowest q vs rho = r_area/t_area (at q_ref) terciles
def print_rho_check(df, ref, good):

    #> good models at the lowest q
    d   = df.set_index(ids).loc[good].reset_index()
    d   = d[d['q'] == q_grid[0]]
    rho = (ref['r_area'] / ref['t_area']).reindex(d.set_index(ids).index).to_numpy()

    #> terciles
    edges = np.nanpercentile(rho, [0, 33, 67, 100])
    print(f'\n> f_q_area at q = {q_grid[0]} vs rho = r_area/t_area (q = {q_ref}) terciles')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho <= hi)
        print(f">   rho {lo:.3f}-{hi:.3f}: f_q_area {np.nanmedian(d['f_q_area'].to_numpy()[m]):.3f}")

    return


""" #> FIT ===========================
================================== """

#> log10 rho of every row + its bin; bin centers = median log10 rho in each bin
def rho_bins(d, ref):

    #> log10 rho (at q_ref) for every row
    key       = d.set_index(ids).index
    d['lrho'] = np.log10((ref['r_area'] / ref['t_area']).reindex(key).to_numpy())

    #> bins
    d['b'] = np.digitize(d['lrho'], lrho_edges) - 1

    #> centers and counts (one row per model)
    d_ref = d[d['q'] == q_ref]
    lrho  = d_ref.groupby('b')['lrho'].median()
    nbin  = d_ref.groupby('b').size()

    return d, lrho, nbin


#> fit: medians of f_ER, f_q_area on (q, log10 rho) and of f_t_area on q (interpolated in lookup.q_factors)
def fit_qfactor(df, **kwargs):

    #> kwargs
    fitFile    = kwargs.get('fitFile', './priors/qfactor.json')
    area_floor = kwargs.get('area_floor', 1e-4)                  # min q_area at q_ref [arcsec^2]
    mdef       = kwargs.get('mdef', 'vir')

    #> good models + rho bins
    ref, good     = reference(df, area_floor)
    d             = df.set_index(ids).loc[good].reset_index()
    d, lrho, nbin = rho_bins(d, ref)

    #> medians
    f2d = {c: d.groupby(['q', 'b'])[c].median().unstack()[lrho.index] for c in ['f_ER', 'f_q_area']}
    ft  = d.groupby('q')['f_t_area'].median()

    #> fit dictionary
    fit = {'q_ref':        q_ref,
           'mdef':         mdef,
           'nmodels':      len(good),
           'q':            ft.index.tolist(),
           'lrho':         lrho.tolist(),
           'nmodels_lrho': nbin.tolist(),
           'f_t_area':     ft.tolist(),
           'f_ER':         f2d['f_ER'].to_numpy().tolist(),
           'f_q_area':     f2d['f_q_area'].to_numpy().tolist()}

    #> saving
    with open(fitFile, 'w') as f: json.dump(fit, f, indent=2)

    #> print
    print(f'\n> fit over {len(good)} models, log10 rho centers {np.round(lrho.to_numpy(), 2)} ({nbin.tolist()} models)')
    for c in f2d: print(f'\n> {c}(q, log10 rho)\n' + f2d[c].round(3).to_string())
    print(f'\n> saved {fitFile}')

    return fit


""" #> FUNCTION ======================
================================== """

#> runs the calibration
def function(**kwargs):

    #> kwargs
    mdef    = kwargs.get('mdef', 'vir')                         # halo mass definition
    nper    = kwargs.get('nper', 40)                            # models per cell
    workers = kwargs.get('workers', None)                       # number of processes
    outFile = kwargs.get('outFile', './priors/qfactor_calib.csv')
    fitFile = kwargs.get('fitFile', './priors/qfactor.json')

    #> config + bins
    config, bins = get_config(mdef=mdef, out_dir='./priors/_qcalib/')

    #> running
    srt  = time.time()
    jobs = make_jobs(bins, nper, kwargs.get('seed', 1))
    rows = run_jobs(jobs, config, bins, workers)

    #> table
    df = make_table(rows)
    df.to_csv(outFile, index=False)

    #> summary
    ref, good = reference(df, config['area_floor'])
    print_summary(df, good)
    print_rho_check(df, ref, good)

    #> fit
    fit_qfactor(df, fitFile=fitFile, area_floor=config['area_floor'], mdef=mdef)
    print(f'\n> saved {outFile} ({(time.time() - srt)/60:.1f} min)')

    return df


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> name
    print('> '+os.path.basename(__file__),'\n')

    #> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdef', type=str, default='vir', choices=['vir', '200c'])
    parser.add_argument('--nper', type=int, default=40)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--fit-only', action='store_true')                  # refits ./priors/qfactor_calib.csv
    args = parser.parse_args()

    #> refit or run
    if args.fit_only: fit_qfactor(pd.read_csv('./priors/qfactor_calib.csv'), mdef=args.mdef)
    else: function(mdef=args.mdef, nper=args.nper, workers=args.workers)

    # end
# thank
