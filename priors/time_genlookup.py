#> name: time_genlookup.py
#> author: John Miller Jr
#> descrp: times genlookup_newt.solve per model for one (zl, zs) cell, on one process, and prints the slowest models

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import argparse
import numpy as np

#> adding dir to sys paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#> generator
import genlookup_newt as gl


""" #> CONFIG ========================
================================== """

#> stops function() at the pool and returns its (config, bins)
class _Grab(Exception): pass
class _Capture:
    def __init__(self, *args, initializer=None, initargs=(), **kwargs): raise _Grab(initargs)

#> config + bins exactly as genlookup_newt.function() builds them
def get_config(**kwargs):
    gl.ProcessPoolExecutor = _Capture
    try: gl.function(**(kwargs | {'workers': 1, 'resume': False}))
    except _Grab as e: return e.args[0]
    return None


""" #> TIMING ========================
================================== """

#> times every step-th model of one cell
def function(**kwargs):

    #> kwargs
    validate = kwargs.get('validate', True)      # validation run (v3 grid)
    mdef     = kwargs.get('mdef', 'vir')         # halo mass definition
    zl       = kwargs.get('zl', 0.5)             # lens redshift (nearest bin)
    zs       = kwargs.get('zs', 2.0)             # source redshift (nearest bin)
    step     = kwargs.get('step', 7)             # every step-th model
    ntop     = kwargs.get('ntop', 15)            # slowest models printed

    #> config + worker
    config, bins = get_config(validate=validate, mdef=mdef, out_dir='./priors/_timing/')
    gl.init_worker(config, bins)

    #> shapely (without it newtlens.overlap_area falls back to a slow grid estimate)
    try:
        import shapely
        print(f'\n> shapely {shapely.__version__}')
    except ImportError: print('\n> shapely NOT installed --> slow grid overlap in nc_area')

    #> cell
    izl, izs = int(np.argmin(np.abs(bins['zl'] - zl))), int(np.argmin(np.abs(bins['zs'] - zs)))
    zl, zs = bins['zl'][izl], bins['zs'][izs]
    srt = time.time()
    models, bprofiles, seeds = gl.build_models(zl, zs)
    print(f'\n> cell zl = {zl:.2f}, zs = {zs:.2f}: {len(models)} models built in {time.time() - srt:.1f} s, timing every {step}th\n')

    #> timing each model
    rows = []
    for i in range(0, len(models), step):
        if bprofiles[i] is None: continue
        t0 = time.time()
        r = gl.solve(bprofiles[i], seeds[i])
        rows.append([time.time() - t0, *models[i][3:], seeds[i], r['ER'], r['pix_arc'], r['n_pass'],
                     r['window_retries'], r['window_failed'], r['n_tcurve'], r['n_rcurve'], r['nims']])
    rows = np.array(rows, dtype=float)
    dt = rows[:,0]                                                          # s

    #> summary
    print(f'> {len(dt)} models: median {1e3*np.median(dt):.0f} ms, mean {1e3*dt.mean():.0f} ms, max {dt.max():.2f} s')
    print(f'> estimated cell time (1 process): {dt.mean()*len(models)/60:.1f} min')
    print(f'> slowest {min(ntop, len(dt))} models take {100*np.sort(dt)[::-1][:ntop].sum()/dt.sum():.0f}% of the time\n')

    #> slowest models
    names = ['time [s]', 'logMvir', 'c', 'logM*', 'Reff', 'm*_nsig', 're_nsig', 'seed ER', 'ER', 'pix_arc',
             'passes', 'retries', 'w_fail', 'n_t', 'n_r', 'nims']
    print(' '.join(f'{n:>8s}' for n in names))
    for row in rows[np.argsort(dt)[::-1][:ntop]]:
        print(' '.join(f'{v:8.3f}' for v in row))

    return rows


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> name
    print('> '+os.path.basename(__file__),'\n')

    #> arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdef', type=str, default='vir', choices=['vir', '200c'])
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument('--zl', type=float, default=0.5)
    parser.add_argument('--zs', type=float, default=2.0)
    parser.add_argument('--step', type=int, default=7)
    args = parser.parse_args()

    function(mdef=args.mdef, validate=not args.no_validate, zl=args.zl, zs=args.zs, step=args.step)

    # end
# thank
