#> name: lensing.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt

#> contour finding
# from modules.MarchingNumPy.MarchingNumPy.MarchingSquares import marching_squares
# from modules.MarchingNumPy.MarchingCuPy.MarchingSquares import marching_squares

#> special imports
from shapely import geometry as geo
from scipy.interpolate import RectBivariateSpline as spline
from scipy.spatial.distance import cdist

from skimage import measure
from shapely.geometry import LineString
from shapely.geometry import Point, Polygon
import geopandas as gpd

#> main directory
global proj_loc
proj_loc = 'projects/'

#> modules
import modules.geometry as geometry
from modules.units import u; u=u()
import modules.error as error


""" #> CONTOUR INTERSECTION ==========
================================== """


""" #> PARALLEL FIMS =================
================================== """

#> parallelizes fims
def pfims(x, y, gradx, grady, lamt, pix_arc, nph, numQuads_gal, observables, profiles):
    
    import multiprocessing as mp
    
    #> declarations
    workers = mp.cpu_count()
    print(f'> There are {workers} workers!')
    pool = mp.Pool(processes=workers)
    
    # submit all tasks first
    async_results = []
    for i in range(len(gradx)):
        for j in range(numQuads_gal):
            async_results.append(
                pool.apply_async(
                    pfimsFn, 
                    args=(x, y, 
                          gradx[i], grady[i], 
                          lamt[i], 
                          pix_arc[i], nph, 
                          observables, profiles)))
    pool.close()
    
    #> collecting results
    results = []
    for r in async_results: results.append(r.get())
    pool.join()
    plt.show()
    
    return results


#> parallelizes finding ims (picking source, etc.)
def pfimsFn(x, y, gradx, grady, lamt, pix_arc, nph, observables, profiles):
    
    #> get caustic box
    srt = time.time()
    source_apexes = causticBox(x, y, gradx, grady, pix_arc, lamt=lamt)
    # print(f'> CausticBox: {time.time()-srt:.5e} s')
    # plot.causticBox(x, y, gradx, grady, pix_arc, source_apexes)
    
    #> if galaxy does not have a caustic
    if len(source_apexes) == 0: 
        print(source_apexes)
        error.phrase('GALAXY HAS NO CAUSTIC!!')
        
    #> if caustic area is 0? or very small?
    if Polygon(source_apexes).area < 1e-6:
        import plot
        plot.caustics(x, y, gradx, grady, pix_arc=60, outFile='zero-caustic')
        plot.causticBox(x, y, gradx, grady, boxPoints=source_apexes, pix_arc=60, outFile='caustic-box', zoom=1)
        print('NO CAUSTIC!')
        os._exit()
    
    #> while not quad
    images = []
    max_tries = 100
    srt = time.time()
    while len(images) != 5 and max_tries > 0:
    
        #> get random source inside box
        source = ranSources2(source_apexes, snum=1) / u.arc_rad # arc --> rad
        
        #> lens
        images = lfims(x / pix_arc / u.arc_rad,  # radians
                       y / pix_arc / u.arc_rad,  # radians
                       gradx,                    # radians
                       grady,                    # radians
                       source[0][0],             # radians
                       source[0][1],             # radians
                       pix_arc, nph)
        
        max_tries -= 1
    # print(f'> fims: {time.time()-srt:.5e} s')
    
    #> if search didn't time out
    if max_tries != 0: 
        
        #> if only wanting image positions
        if observables == None: return images
        
        #> if wanting lensing observables
        srt = time.time()
        obs = geometry.lensObs((0, 0), images, observables)
        # print(f'> Observables: {time.time()-srt:.5e} s')
        return obs
    
    else: return []



""" #> FINDING IMAGES CPU ============
================================== """

#> finding contours
def contourFinding(data):
    return marching_squares(data, 0.0, interpolation="LINEAR", resolve_ambiguous=True)

def fims(X, Y, gradx, grady, xs, ys, nph, pix_arc):
    
    #> pix --> radians
    X /= (pix_arc * u.arc_rad)
    Y /= (pix_arc * u.arc_rad)
    
    #> grid
    dx = X-gradx-xs
    dy = Y-grady-ys
    
    #np.save('deltx', dx)
    #np.save('delty', dy)
    
    vx, lx = contourFinding(dx)
    vy, ly = contourFinding(dy)
    
    bpts = vx[lx]
    rpts = vy[ly]
    
    rpts = rpts[:,:,::-1]
    bpts = bpts[:,:,::-1]
    
    # ensure proper shape (DO NOT destroy geometry)
    rpts = np.asarray(rpts)
    bpts = np.asarray(bpts)
        
    xims, yims = [], []
    
    for vx in rpts:
        polyx = geo.LineString(vx)
    
        for vy in bpts:
            polyy = geo.LineString(vy)
    
            if polyx.intersects(polyy):
                try:
                    intersection = polyx.intersection(polyy)
    
                    if isinstance(intersection, geo.Point):
                        xims.append(intersection.x)
                        yims.append(intersection.y)
    
                    elif hasattr(intersection, "geoms"):
                        for g in intersection.geoms:
                            xims.append(float(g.x))
                            yims.append(float(g.y))
    
                except RuntimeWarning:
                    continue
                
    #> converting to dict
    xims = [ (x-nph)/pix_arc for x in xims ]
    yims = [ (y-nph)/pix_arc for y in yims ]
    images = np.column_stack((xims, yims))
    
    #> orders based on inverse distance to center + lensing theory
    if images.shape !=  (0,2):
        images = arrivalOrder(images, x0=0, y0=0)
        
    print(images)
        
    return np.array(images)


""" #> FINDING IMAGES ================
================================== """

#> finding images
def lfims(x, y, gradx, grady, xs, ys, pix_arc, nph):
    
    #> lens eq, x & y
    deltx = x-gradx-xs
    delty = y-grady-ys
    
    #> finding red and blue lines
    rpts, bpts = [], []
    for axis, points, color in zip([deltx, delty], [rpts, bpts], ['b', 'r']):
        
        #> finding 0.0 contour
        contours = measure.find_contours(axis.T, 0.0)
        for contour in contours:
            points.append( LineString(contour) )
        # points.append([ LineString(contour) for contour in contours ])
    
    #> finding intersections
    xims, yims = [], []
    for vx in rpts:
        for i, vy in enumerate(bpts):
            polyx = geo.LineString(vx)
            polyy = geo.LineString(vy)
            if polyx.intersects(polyy):
                try:
                    intersection = polyx.intersection(polyy)
                    if isinstance(intersection, geo.point.Point):
                        xims.append(intersection.x)
                        yims.append(intersection.y)
                    else:
                        for i in range(len(intersection.geoms)):
                            xims.append(float(intersection.geoms[i].x))
                            yims.append(float(intersection.geoms[i].y))
                    del intersection
                except RuntimeWarning: # Lines don't intersect!
                    continue
    
    #> converting to dict
    xims = [ (x-nph)/pix_arc for x in xims ]
    yims = [ (y-nph)/pix_arc for y in yims ]
    images = np.column_stack((xims, yims))
    # if len(images) != 5: print(images)
    
    #> orders based on inverse distance to center + lensing theory
    if images.shape !=  (0,2):
        images = arrivalOrder(images, x0=0, y0=0)
        
    return np.array(images)


#> returns images in inverse order of distance from origin
def dorder(images, x0=0.0, y0=0.0):
    
    oims = []
    for img in images:
        oims.append([abs(np.linalg.norm(img - (x0,y0))),
                     img[0], img[1]])
    oims = np.array(oims)
    oims = oims[oims[:,0].argsort()][:,1:] # sorts by distance, then drops
    
    return oims[::-1] # reverses order


# law of cosines
def lawCos(a,b,c):
    val = (a**2 + b**2 - c**2) / (2*a*b)
    agl = math.acos(np.clip(val, -1.0, 1.0)) * (180/(np.pi))
    return agl


#> returns images in arrival time order (inverse + quad theory)
def arrivalOrder(images, x0=0.0, y0=0.0):
    
    #> collecting distances
    oims = abs(np.linalg.norm(images - np.full(shape=images.shape, fill_value=[x0, y0]), axis=1)) # getting distances
    o_angles = np.arctan2(images[:,1], images[:,0])
    oims = np.hstack((np.array([oims]).T, np.array([o_angles]).T, images))

    #> sorts by distance
    oims = oims[oims[:,0].argsort()][::-1]
    # print(oims)
    
    #> if quad
    ordered = np.zeros(oims.shape)
    if len(oims) >= 5:
        
        #> finding max distance between ims 12 and 34
        ind = np.argmax([oims[0][0] - oims[1][0], oims[2][0] - oims[3][0]]) * 3 # either 0 or 3
        ordered[ind] = oims[ind] # adds 1st (or 4th) image to final array
        # print(ordered)
        #> dropping 5th (not important)
        dummy = oims[:-1]
        
        #> finding opposite image
        dummy = dummy[dummy[:,1].argsort()]             # sorting by angle
        index = np.where(dummy[:,0] == oims[ind][0])[0] # finding chosen image
        dummy = np.roll(dummy, shift=-index, axis=0)    # rolling to make 1st (or 4th) image the 1st element
        
        #> assigning opposite
        if ind == 0: ordered[1] = dummy[2]
        else: ordered[2] = dummy[2]
        
        #> finding law of cosines angle between remaining images (other angle does not work for this!!)
        angle21 = lawCos(a=dummy[2][0], b=dummy[1][0], c=np.linalg.norm(dummy[2][2:]-dummy[1][2:]))
        angle23 = lawCos(a=dummy[2][0], b=dummy[3][0], c=np.linalg.norm(dummy[2][2:]-dummy[3][2:]))

        #> assigning remaining images
        if ind == 0: # if found 1st & 2nd
            if angle21 < angle23:
                ordered[2] = dummy[1] # image 3
                ordered[3] = dummy[3] # image 4
            else:
                ordered[2] = dummy[3] # image 3
                ordered[3] = dummy[1] # image 4
        else: # if found 4th & 3rd
            if angle21 < angle23:
                ordered[1] = dummy[1] # image 2
                ordered[0] = dummy[3] # image 1
            else:
                ordered[1] = dummy[3] # image 2
                ordered[0] = dummy[1] # image 1
        ordered[-1] = oims[-1] # filling in 5th image

        return ordered[:,2:] # quad
    
    return oims[:,2:] # if not quad


""" #> CRITICAL CURVRES ==============
================================== """

#> returns the critical curves of a given lens
# https://galaxiesbook.org/chapters/III-04.-Gravitational-Lensing_3-Magnification-and-shear.html
def ccurves(x, y, gradx, grady, pix_arc):
    
    #> getting grid size
    nph = int(len(x)/2)

    #> calculating kappa & gamma
    kappa, gamma = angles_to_kappa_gamma(gradx, grady, pix_arc)
    
    #> eigen values
    lamt = (1 - kappa - gamma) # tangential
    lamr = (1 - kappa + gamma) # radial
    
    #> finding tangential points
    tcurve = []
    contours = measure.find_contours(lamt.T, 0.0)
    for contour in contours: 
        tcurve.append(LineString((contour-nph)/pix_arc)) # arcsec
    
    #> finding radial points
    rcurve = []
    contours = measure.find_contours(lamr.T, 0.0)
    for contour in contours: 
        rcurve.append(LineString((contour-nph)/pix_arc)) # arcsec

    return tcurve, rcurve # arcsec


#> back-projects critical curves to source plane
def caustics(x, y, gradx, grady, pix_arc):
    
    #> interpolating deflection angles
    wrange = x[0]/pix_arc
    fx = spline(wrange, wrange, gradx.T * u.arc_rad)
    fy = spline(wrange, wrange, grady.T * u.arc_rad)
    
    #> getting critical curves
    tcurve, rcurve = ccurves(x, y, gradx, grady, pix_arc)
    
    #> iterating through each point & back-projecting
    dcaustic, ocaustic = [], []
    for tstring, rstring in zip(tcurve, rcurve):
        
        #> diamond caustic (from tangential critical curve)
        t_dummy = []
        for tpoint in tstring.coords:
            xp, yp = tpoint
            t_dummy.append([ xp - fx(xp, yp)[0][0],
                             yp - fy(xp, yp)[0][0] ])
        dcaustic.append(LineString(t_dummy)) # converting to linestring
            
        #> oval caustic (from radial tangential critical curve)    
        o_dummy = []
        for rpoint in rstring.coords:
            xp, yp = rpoint
            o_dummy.append([ xp - fx(xp, yp)[0][0],
                             yp - fy(xp, yp)[0][0] ])
        ocaustic.append(LineString(o_dummy)) # converting to linestring
    
    return dcaustic, ocaustic


#> returns the points of a rectangular box inscribed by the diamond caustic  
def causticBox(x, y, gradx, grady, pix_arc, lamt=None):
    
    #> grid size
    nph = int(len(x)/2)
    
    #> if not calculated before hand
    if lamt is None:
        kappa, gamma = angles_to_kappa_gamma(gradx, grady, pix_arc) #> calculating kappa & gamma
        lamt = (1 - kappa - gamma) # tangential
    
    #> finding tangential points
    # tcurve = []
    contours = measure.find_contours(lamt.T, 0.0)
    if not contours: return [] # if empty (no caustic)
    if len(contours) > 1: 
        error.highlight('Tangential critical curve has more than one contour!?')
    
    #> reducing to (likely only) tcurve
    tcurve = (np.vstack(contours) - nph) / pix_arc
    
    #> back-projecting
    #> interpolating deflection angles
    wrange = x[0] / pix_arc
    fx = spline(wrange, wrange, gradx.T * u.arc_rad)
    fy = spline(wrange, wrange, grady.T * u.arc_rad)
    
    #> splitting coords
    tx = tcurve[:,0]
    ty = tcurve[:,1]
    
    #> iterating through each point & back-projecting
    fx_vals = fx(tx, ty, grid=False)
    fy_vals = fy(tx, ty, grid=False)
    source_apexes = np.column_stack((tx - fx_vals, ty - fy_vals))

    #> getting min and max
    xmin = source_apexes[source_apexes[:,0].argmin()]
    xmax = source_apexes[source_apexes[:,0].argmax()]
    ymin = source_apexes[source_apexes[:,1].argmin()]
    ymax = source_apexes[source_apexes[:,1].argmax()]
    source_apexes = np.array([xmin, ymin, xmax, ymax])
    
    #> checking to ensure caustic is within the window
    if any(abs(source_apexes[:,0]) > nph/pix_arc) or any(abs(source_apexes[:,1]) > nph/pix_arc):
        print('> Caustic points (arcsec):\n', source_apexes)
        error.phrase('Caustic is wider than window!')
    
    return source_apexes # arcsec

#> returns the points of a rectangular box inscribed by the diamond caustic  
def caustic(x, y, gradx, grady, pix_arc, lamt=None):
    
    #> grid size
    nph = int(len(x)/2)
    
    #> if not calculated before hand
    if lamt is None:
        kappa, gamma = angles_to_kappa_gamma(gradx, grady, pix_arc) #> calculating kappa & gamma
        lamt = (1 - kappa - gamma) # tangential
    
    #> finding tangential points
    # tcurve = []
    contours = measure.find_contours(lamt.T, 0.0)
    if not contours: return [] # if empty (no caustic)
    if len(contours) > 1: 
        error.highlight('Tangential critical curve has more than one contour!?')
    
    #> reducing to (likely only) tcurve
    tcurve = (np.vstack(contours) - nph) / pix_arc
    
    #> back-projecting
    #> interpolating deflection angles
    wrange = x[0] / pix_arc
    fx = spline(wrange, wrange, gradx.T * u.arc_rad)
    fy = spline(wrange, wrange, grady.T * u.arc_rad)
    
    #> splitting coords
    tx = tcurve[:,0]
    ty = tcurve[:,1]
    
    #> iterating through each point & back-projecting
    fx_vals = fx(tx, ty, grid=False)
    fy_vals = fy(tx, ty, grid=False)
    source_apexes = np.column_stack((tx - fx_vals, ty - fy_vals))

    #> checking to ensure caustic is within the window
    if any(abs(source_apexes[:,0]) > nph/pix_arc) or any(abs(source_apexes[:,1]) > nph/pix_arc):
        print('> Caustic points (arcsec):\n', source_apexes)
        error.phrase('Caustic is wider than window!')
    
    return source_apexes # arcsec


#> returns the magnification map
def magnification_map(gradx, grady, pix_arc):

    #> calculating kappa & gamma
    kappa, gamma = angles_to_kappa_gamma(gradx, grady, pix_arc)
    
    #> eigen values
    lamt = (1 - kappa - gamma) # tangential
    lamr = (1 - kappa + gamma) # radial
    
    #> magnification map
    magmap = 1/abs(lamt * lamr)
        
    return magmap


#> returns the magnification map
def magnification(gridx, gridy, gradx, grady, pix_arc, images):

    #> calculating kappa & gamma
    kappa, gamma = angles_to_kappa_gamma(gradx, grady, pix_arc)
    
    #> eigen values
    lamt = (1 - kappa - gamma) # tangential
    lamr = (1 - kappa + gamma) # radial
    
    #> magnification map
    # magmap = 1/(lamt * lamr)
    detA = (lamt * lamr)
    # magmap = np.abs(magmap)
    
    #> interpolating deflection angles
    wrange = gridx[0] / pix_arc
    fmag = spline(wrange, wrange, detA) # interpolating 1/detA is unstable
    
    #> splitting coords
    tx = images[:,0]
    ty = images[:,1]
    
    #> calculating mags
    im_detA = fmag(ty, tx, grid=False)
    im_mags = 1 / im_detA # inverting for mags
    parities = np.sign(im_mags)
    im_mags = np.abs(im_mags)
    if any(parities!=[ 1.,  1., -1., -1.,  1.]):
        if (np.log10(im_mags[1]) > 2.) and (np.log10(im_mags[2]) > 2.):
            error.highlight(f'[continuing] Magnification parities are not correct! {im_mags}')
            pass
        else:
            error.phrase(f'Magnification parities are not correct! {im_mags}')
        
    return im_mags


""" #> GETTING OTHER FIELDS ==========
================================== """

#> calculates kappa from deflection angles
def angles_to_kappa(gradx, grady, pix_arc):
    gradxy, gradxx = np.gradient(gradx * pix_arc * u.arc_rad) # grad^2 x
    gradyy, gradyx = np.gradient(grady * pix_arc * u.arc_rad) # grad^2 y
    return 0.5 * (gradxx + gradyy)

#> calculates kappa and gamma from deflection angles
def angles_to_kappa_gamma(gradx, grady, pix_arc):
    
    #> calculating kappa
    gradxy, gradxx = np.gradient(gradx * pix_arc * u.arc_rad) # grad^2 x
    gradyy, gradyx = np.gradient(grady * pix_arc * u.arc_rad) # grad^2 y
    kappa = 0.5 * (gradxx + gradyy)
    
    #> calculating gammas
    gamma1 = 0.5 * (gradxx - gradyy)
    gamma = np.sqrt( gamma1**2 + gradxy**2 )
    
    return kappa, gamma


""" #> RANDOM SOURCES ================
================================== """

#> getting random list of sources
def ranSources(source_apexes, snum):
    
    #> declarations
    sources = []
    
    #> getting polygon
    caustic_poly = Polygon(source_apexes)
    minx, miny, maxx, maxy = caustic_poly.bounds
    
    #> looking for sources
    i = 0
    tries = snum*1000
    while (i < snum) and (tries > 0):
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if caustic_poly.contains(pnt):
            sources.append([pnt.x, pnt.y])
            i += 1
        tries -= 1
    
    return np.array(sources)


#> version 2 of finding random sources
def ranSources2(source_apexes, snum=1):
    
    #> polygon
    poly = Polygon(source_apexes)
    if poly.area <= 1e-6: 
        print(poly.area)
        print('CAUSTIC AREA IS TOO SMALL!')
        return None
    
    #> creating geopandas object
    gs = gpd.GeoSeries([poly])
    
    #> sampling to find points
    sampled_sources = gs.sample_points(size=snum, method='uniform')
    
    #> get singular point (will be an issue if requesting multiple)
    points_list = list(sampled_sources[0].coords)
    
    return np.array(points_list)


""" #> SELECTION EFFECTS =============
================================== """

#> selection effect; returns True if images would likely be resolved
#  assumes resolution is the minimum separation and is in the same units as images
def resolved(images, resolution):
    distances = cdist(images, images, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    return resolution < np.min(distances)


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> imports
    import plot
    import deflection
    import params
    
    #> declaration
    numGals = 1000
    numQuads_gal = 10
    observables = ['t23', 'dt23', 'd4/d1']
    
    #> grid declarations
    nph = 50
    pix_arc = 60 # default in bprofiles, does not change anything except for plotting here!!
    w_range = np.arange(-nph, nph, 1, dtype=float)
    xgrid, ygrid = np.meshgrid(w_range, w_range)
    
    #> generating profiles
    profiles = params.bprofiles(numGals)
    
    #> calculating deflections
    srt = time.time()
    delx, dely, lamt = deflection.deflectionGPU(xgrid, ygrid, profiles, lamt=True)
    # cp.cuda.Stream.null.synchronize() # if there is GPU available, can use
    print(f'> Generating galaxies took {time.time()-srt:.1e} s or {(time.time()-srt)/numGals:.1e} s/gal.')
    
    #> plotting kappa
    # kappa = angles_to_kappa(delx[0], dely[0], pix_arc=60)
    # plot.kappa(xgrid, ygrid, kappa, pix_arc)

    #> conducting fims in parallel    
    # pfims(xgrid, ygrid, delx, dely, pix_arc, nph=50)
    pix_arc=np.full(shape=(1,numGals),fill_value=pix_arc)[0]
    srt1 = time.time()
    images = pfims(xgrid, ygrid, delx, dely, lamt, pix_arc, nph, numQuads_gal, observables)
    # images = pfimsFn(xgrid, ygrid, delx[0], dely[0], lamt[0], pix_arc[0], nph, observables) 

    print(f'> Fims took {time.time()-srt1:.1e}s or {(time.time()-srt1)/numGals/numQuads_gal:.1e} s/src')
    
    # end
# thank