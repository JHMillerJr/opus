#> name: order.py
#> author: John Miller Jr
#> descrp: image ordering for opus: fermat (arrival-time) order and morphological order

""" #> IMPORTS =======================
================================== """

#> standard imports
import numpy as np

#> modules
import profiles
from modules.units import u; u=u()

""" #> ORDERING ======================
================================== """

#> orders images by fermat potential
def fermat_order(ims, mu, srcs, bprof):

    #> flatting images (S=# sources, I=# images per source)
    S, I, _ = ims.shape
    th      = ims.reshape(-1, 2) # thetas
    
    #> nan handling
    ok = ~np.isnan(th[:,0])
    
    #> declarations
    tau  = np.full(S*I, np.inf)        # rad^2
    beta = np.repeat(srcs, I, axis=0)  # source positions

    #> lens potential
    psi = profiles.lpot_points(th[ok,0], th[ok,1], bprof) # rad^2
    
    #> fermat potentail
    al2 = np.sum( ( (th[ok] - beta[ok]) / u.arc_rad )**2, axis=1)
    tau[ok] = 0.5*al2 - psi  # rad^2
    tau = tau.reshape(S, I) 

    #> fermat order
    order = np.argsort(tau, axis=1)
    
    #> sorting all observables
    take = lambda array: np.take_along_axis(array, order, axis=1)
    ims = np.stack([take(ims[:,:,0]), take(ims[:,:,1])], axis=-1)

    return ims, take(mu), take(tau) # arcsec, signed mags, rad^2


#> morphological arrival order
def morph_order(ims, mu=None):
    
    #> declarations
    numIms = len(ims)
    q = np.arange(numIms)[:,None]
    
    #> sorting by distance (furthest first)
    r = np.hypot(ims[:,:,0], ims[:,:,1]) # arcsec
    o = np.argsort(r, axis=1)[:,::-1]
    oims, r = ims[q, o], r[q, o]
    
    #> 1st (or 4th) image, depending on the biggest radial gap between ims 12 or 34
    ind = np.where(r[:,0]-r[:,1] >= r[:,2]-r[:,3], 0, 3)
    
    #> removing 5th image (for now)
    dummy = oims[:,:4]
    
    #> sorting by angle (ensures switching between m & s)
    ang    = np.arctan2(dummy[:,:,1], dummy[:,:,0]) # rad
    oaindx = np.argsort(ang, axis=1)
    
    #> starts quad arrays at ind, orders in ccw angle
    pos  = np.argmax(oaindx == ind[:,None], axis=1)
    wrap = (np.arange(4)[None] + pos[:,None]) % 4
    roll = oaindx[q, wrap]
    d0, d1, d2, d3 = [ dummy[ np.arange(numIms), roll[:,k] ] for k in range(4) ]
    
    #> d0 is first (fourth), and d2 is second (third)
    
    #> angles at the origin
    def lawcos(a, b):
        
        #> calculating distances
        ra = np.hypot(*a.T)
        rb = np.hypot(*b.T)
        c  = np.hypot(*(a-b).T)
        
        #> calc anlge C
        tc = (ra**2 + rb**2 - c**2) / (2*ra*rb)
        return np.degrees(np.arccos(np.clip(tc, a_min=-1, a_max=1)))
    
    #> finding third (second) by finding smallest t23
    a21   = lawcos(d2, d1)
    a23   = lawcos(d2, d3)
    near1 = (a21 < a23)[:,None]
    
    #> quads where first arriving is found (=True), (fourth=False)
    first = (ind == 0)[:,None]
    
    #> if ind == 0 (i.e., first arriving is found)
    #> d0 = first
    #> d2 = second
    #> if near1=True, d1 is third , else d3 is third
    #> if near1=True, d3 is fourth, else d1 is fourth
    
    #> if ind == 3 (i.e., fourth arriving is found)
    #> d0 = fourth
    #> d2 = third
    #> if near1=True, d1 is second, else d3 is second
    #> if near1=True, d3 is first , else d1 is first
    
    #> assigning images  # if first               # if fourth
    i1 = np.where(first, dummy[:,0]             , np.where(near1, d3, d1))
    i2 = np.where(first, d2                     , np.where(near1, d1, d3))
    i3 = np.where(first, np.where(near1, d1, d3), d2)
    i4 = np.where(first, np.where(near1, d3, d1), dummy[:,3])
    
    #> collecting + adding back fifth arriving
    out = np.stack([i1, i2, i3, i4, oims[:,4]], axis=1)
    
    #> reordering mags to match (by position)
    if mu is not None:
        
        #> distance from every output image to every input image
        diff = out[:,:,None,:] - ims[:,None,:,:]           # (Q,5,5,2)
        dist = np.hypot(diff[:,:,:,0], diff[:,:,:,1])      # (Q,5,5)
        
        #> input slot of each output image (closest match)
        idx = np.argmin(dist, axis=2)                      # (Q,5)
        
        #> reordering mags
        mu = np.take_along_axis(mu, idx, axis=1)
        return out, mu
    
    return out


""" #> COMPARING =====================
================================== """

#> morphological vs fermat order of quads (images given in fermat order)
#> comp: fermat arrival number (1-5) of each morphologically ordered image; match: 1 if the 4 outer images agree, else 0
def morph_vs_fermat(fims):
    
    #> morphological order of the fermat-ordered images
    mims = morph_order(fims)                                # (Q,5,2) arcsec
    
    #> fermat slot of each morph image (closest match)
    diff = mims[:,:,None,:] - fims[:,None,:,:]              # (Q,5,5,2) arcsec
    dist = np.hypot(diff[:,:,:,0], diff[:,:,:,1])           # (Q,5,5) arcsec
    comp = np.argmin(dist, axis=2) + 1                      # (Q,5) arrival number (1-5)
    
    #> agreement of the 4 outer images
    match = np.all(comp[:,:4] == np.arange(1, 5), axis=1).astype(float)
    
    return comp, match
