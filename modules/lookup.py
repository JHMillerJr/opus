#> name: lookup.py
#> author: John Miller Jr
#> descrp: scrapes the newt look-up tables (priors/genlookup_newt.py) for quick prior draws: einstein radius, pixel scale,
#>         and quad probability, w/ the axis-ratio corrections of priors/calib_qfactor.py (old versions: pastlens/oldlookup.py)

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import json
import numpy as np
import pandas as pd

#> caching
from functools import lru_cache
from itertools import product

#> adding dir to sys paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> DECLARATIONS ==================
================================== """

#> look-up tables for each halo mass definition (priors/genlookup_newt.py)
lookup_locs_newt = {'vir':  './priors/newt_tables_cvir/',              # nfw w/ delta_vir, c_vir
                    '200c': './priors/newt_tables_c200/'}              # nfw w/ delta = 200, c_200
gridFile         = 'lookup_grid_newt.json'                             # grid metadata

#> axis-ratio corrections (priors/calib_qfactor.py); the tables are made at q_ref = 0.85
qfactor_loc = './priors/qfactor.json'

#> grid axes (table rows: izs, then logmvir, mstar_nsig, re_nsig)
gridKeys = ['zl', 'zs', 'logmvir', 'mstar_nsig', 're_nsig']

#> galaxy columns used for the look-up (same order as gridKeys)
galCols = ['zl', 'zs', 'log10_M_vir', 'mstar_nsig', 're_nsig']

#> interpolated table values
valCols = ['ER', 'q_area', 'q_area_max', 'r_area', 't_area']


""" #> TABLE LOCATION ================
================================== """

#> true if a newt table folder is finished (metadata + every zl table)
def complete_newt(loc):

    #> metadata
    meta = os.path.join(loc, gridFile)
    if not os.path.isfile(meta): return False
    with open(meta) as f: zls = json.load(f)['zl_grid']

    #> every zl table
    return all(os.path.isfile(table_file(loc, zl)) for zl in zls)


#> look-up directory for a halo mass definition
def lookup_dir_newt(mdef='vir'):

    #> imports
    import modules.error as error

    #> checking mdef
    if mdef not in lookup_locs_newt:
        error.phrase(f"mdef={mdef} not understood. Please use 'vir' or '200c'")

    #> checking tables
    if not complete_newt(lookup_locs_newt[mdef]):
        error.phrase(f'newt look-up tables for mdef={mdef} not finished in {lookup_locs_newt[mdef]}. Run priors/genlookup_newt.py --mdef {mdef}')

    return lookup_locs_newt[mdef]


#> file of one zl table
def table_file(loc, zl):
    return os.path.join(loc, f'model_zl_{zl:.2f}.parquet')


""" #> LOADING =======================
================================== """

#> loads newt look-up bins
@lru_cache(maxsize=None)
def load_lookup_grid_newt(loc=lookup_locs_newt['vir']):

    #> loading metadata
    with open(os.path.join(loc, gridFile)) as f: meta = json.load(f)

    return {key: np.asarray(meta[f'{key}_grid']) for key in gridKeys}


#> largest quad area w/ ER <= max_er @ each zs [arcsec^2]
def quad_area_max(df, max_er=2.4):

    #> declarations
    q_area_max = pd.Series(np.nan, index=df.index)

    #> each zs
    for izs in df['izs'].unique():
        mask = (df['izs'] == int(izs) ) & (df['ER'] <= max_er) # mask @ zs
        q_area_max[df['izs'] == izs] = df.loc[mask,'q_area'].max()

    return q_area_max


#> loads one zl look-up table (ER, q_area, q_area_max, r_area, t_area)
@lru_cache(maxsize=None)
def load_model_table_newt(izl, loc=lookup_locs_newt['vir'], max_er=2.4):

    #> lens redshift
    zl = load_lookup_grid_newt(loc)['zl'][izl]

    #> loading needed columns
    df = pd.read_parquet(table_file(loc, zl), columns=['izs','ER','q_area','r_area','t_area'])

    #> normalization
    df['q_area_max'] = quad_area_max(df, max_er)

    return df[valCols].to_numpy()


""" #> AXIS-RATIO CORRECTIONS ========
================================== """

#> loads the f(q, log10 rho) fits
@lru_cache(maxsize=None)
def load_qfactor(loc=qfactor_loc):

    #> imports
    from scipy.interpolate import RegularGridInterpolator as rgi

    #> fits
    with open(loc) as f: fit = json.load(f)
    q, lrho = np.asarray(fit['q']), np.asarray(fit['lrho'])

    return {'q':        q,
            'lrho':     lrho,
            'f_ER':     rgi((q, lrho), np.asarray(fit['f_ER'])),      # ER(q)/ER(q_ref)
            'f_q_area': rgi((q, lrho), np.asarray(fit['f_q_area']))}  # q_area(q)/q_area(q_ref)


#> f_ER, f_q_area at axis ratio q and rho = r_area/t_area (at q_ref); clipped to the calibrated range
def q_factors(q, rho, loc=qfactor_loc):

    #> fits
    qf = load_qfactor(loc)

    #> points (clipped)
    q    = np.clip(q, qf['q'][0], qf['q'][-1])
    lrho = np.clip(np.log10(rho), qf['lrho'][0], qf['lrho'][-1])
    pts  = np.column_stack((q, lrho))

    return qf['f_ER'](pts), qf['f_q_area'](pts)


#> largest f_q_area at q (over rho), so p_quad <= 1 for every galaxy
def q_factor_max(q, loc=qfactor_loc):

    #> fits
    qf  = load_qfactor(loc)
    rho = 10**qf['lrho']

    #> f_q_area at every rho bin
    f_q = q_factors(np.full(len(rho), q), rho, loc)[1]

    return float(np.max(f_q))


#> corrected ER [arcsec], q_area + q_area_max [arcsec^2] for the galaxies' axis ratios
def q_correct(vals, q, q_min=None):

    #> unpacking
    ER, q_area, q_area_max, r_area, t_area = vals.T

    #> rho = r_area/t_area (at q_ref)
    rho = np.divide(r_area, t_area, out=np.ones_like(t_area), where=t_area > 0)

    #> corrections
    f_ER, f_q = q_factors(q, rho)
    q_low     = load_qfactor()['q'][0] if q_min is None else q_min

    return ER*f_ER, q_area*f_q, q_area_max*q_factor_max(q_low)


""" #> INTERPOLATION =================
================================== """

#> grid indices and weights
def _grid_newt(x, grid, interp=True):

    #> below grid
    if x <= grid[0]:
        return ((0,1.0),)

    #> above grid
    if x >= grid[-1]:
        return ((len(grid)-1,1.0),)

    #> surrounding bins
    i1 = np.searchsorted(grid,x)
    i0 = i1-1

    #> nearest
    if not interp:

        if x-grid[i0] <= grid[i1]-x:
            return ((i0,1.0),)

        return ((i1,1.0),)

    #> interpolation
    w = (x-grid[i0])/(grid[i1]-grid[i0])

    return ((i0,1-w),(i1,w))


#> interpolation flags per axis (zl, zs nearest; masses + sizes interpolated)
def interp_flags(**kwargs):
    return {'zl':         kwargs.get('interp_zl',    False),
            'zs':         kwargs.get('interp_zs',    False),
            'logmvir':    kwargs.get('interp_mvir',  True),
            'mstar_nsig': kwargs.get('interp_mstar', True),
            're_nsig':    kwargs.get('interp_re',    True)}


#> interpolated table values of one galaxy
def interp_galaxy(galaxy, bins, flags, loc, max_er=2.4):

    #> dimensions
    nmvir  = len(bins['logmvir'])
    nmstar = len(bins['mstar_nsig'])
    nre    = len(bins['re_nsig'])

    #> indices and weights per axis
    inds = [_grid_newt(x, bins[key], flags[key]) for x, key in zip(galaxy, gridKeys)]

    #> surrounding points
    val = np.zeros(len(valCols))
    for (izl,wz),(izs,ws),(ih,wh),(im,wm),(ir,wr) in product(*inds):

        #> total weight
        w = wz*ws*wh*wm*wr

        #> flattened row
        ind = (((izs*nmvir + ih)*nmstar + im)*nre + ir)

        #> adding point
        val += w*load_model_table_newt(izl,loc,max_er)[ind]

    return val


#> interpolated table values of every galaxy (rows: galaxies, columns: valCols)
def interp_galaxies(df, loc, **kwargs):

    #> kwargs
    max_er = kwargs.get('max_er', 2.4)      # max ER for area normalization [arcsec]

    #> grids + flags
    bins  = load_lookup_grid_newt(loc)
    flags = interp_flags(**kwargs)

    #> galaxies
    galaxies = df[galCols].to_numpy()

    return np.array([interp_galaxy(g, bins, flags, loc, max_er) for g in galaxies]).reshape(-1, len(valCols))


""" #> SELECTION =====================
================================== """

#> quad probability (q_area / q_area_max, clipped to [0, 1])
def quad_probability(q_area, q_area_max):

    p_quad = np.divide(q_area, q_area_max,
                       out=np.zeros_like(q_area),
                       where=q_area_max > 0)

    return np.clip(p_quad,0,1)


#> pixel scale [pix/arcsec] that puts the ER at r_pix_percentage of nph
def pixel_scale(ER, nph=50, r_pix_percentage=0.5):
    return np.divide(nph*r_pix_percentage, ER,
                     out=np.full_like(ER,np.nan),
                     where=ER > 0)


#> selection (selDict from params.getSelDict, otherwise the older sel_res / sel_slcs kwargs)
def selection(ER, p_quad, **kwargs):

    #> kwargs
    ER_resolution = kwargs.get('ER_resolution', 0.15)
    selDict = kwargs.get('selDict', None) or {'res':  {'on': kwargs.get('sel_res', True), 'ER_resolution': ER_resolution},
                                              'slcs': {'on': kwargs.get('sel_slcs', True)}}
    sel_res = selDict['res']['on']
    sel_slcs = selDict['slcs']['on']
    ER_resolution = selDict['res'].get('ER_resolution', ER_resolution)

    #> no selection
    if not sel_res and not sel_slcs:
        return np.ones(len(ER), dtype=bool)

    #> selection effects
    if sel_res and sel_slcs:
        return ((ER >= ER_resolution) & np.isfinite(ER) &
                (np.random.random(len(ER)) < p_quad))

    if sel_res and not sel_slcs:
        return ((ER >= ER_resolution) & np.isfinite(ER))

    return (np.random.random(len(ER)) < p_quad)


""" #> FIRST LOOK ====================
================================== """

#> applies newt first-look selection
def firstLook_selFunc_newt(df, **kwargs):

    #> kwargs
    nph = kwargs.get('nph', 50)
    r_pix_percentage = kwargs.get('r_pix_percentage', 0.5)
    q_min = kwargs.get('q_min', None)       # lower edge of the axis-ratio prior (p_quad normalization), None = calibrated min
    loc = kwargs['lookup_loc'] if 'lookup_loc' in kwargs else lookup_dir_newt(kwargs.get('mdef', 'vir'))

    #> look-up
    vals = interp_galaxies(df, loc, **kwargs)
    ER, q_area, q_area_max = vals[:,0], vals[:,1], vals[:,2]

    #> axis-ratio corrections (galaxies w/ a drawn axis ratio)
    if 'axisrat' in df: ER, q_area, q_area_max = q_correct(vals, df['axisrat'].to_numpy(dtype=float), q_min)

    #> quad probability + pixel scale
    p_quad  = quad_probability(q_area, q_area_max)
    pix_arc = pixel_scale(ER, nph, r_pix_percentage)

    #> saving
    df['ER'] = ER
    df['q_area'] = q_area
    df['q_area_max'] = q_area_max
    df['p_quad'] = p_quad
    df['pix_arc'] = pix_arc
    df['sel'] = selection(ER, p_quad, **kwargs)

    return df


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> name
    print('> '+os.path.basename(__file__))

    #> finished tables
    for mdef, loc in lookup_locs_newt.items():
        print(f'>   mdef={mdef}: {loc} finished = {complete_newt(loc)}')

    # end
# thank
