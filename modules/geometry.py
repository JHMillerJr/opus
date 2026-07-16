#> name: angles.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#> special imports
from shapely import geometry as geo
from matplotlib.patches import Arc

#> modules



""" #> CONVERSION FNS ================
================================== """

#> converts dictionary to array (for angles and distances)
def dictToArr(dict):
    
    #> declaration
    array = []
    
    for i, key in enumerate(dict.keys()):
        array.append(list(dict[key].values()))
    
    return np.array(array)
 

""" #> FUNCTIONS =====================
================================== """

#> calculates distance between two points
def dist(point1, point2):
    return float(np.linalg.norm(np.array(point1)-np.array(point2)))
    

#> determines the distances from origin to images & images to images
def imDists(origin, images):
    
    #> declarations
    dists = {}
    
    #> if image dtype is dictionary
    if isinstance(images, dict):
        
        #> checking to see if quad
        numIms = len([item for item in images.keys() if 'x' in item])
    
        #> interating through each set of images
        for i in range(numIms-1):
            image_point = (images[f'x{i+1}'], images[f'y{i+1}'])
            dists[f'd0{i+1}'] = dist(origin, image_point)
            
        #> image points
        img1 = (images['x1'], images['y1'])
        img2 = (images['x2'], images['y2'])
        img3 = (images['x3'], images['y3'])
        img4 = (images['x4'], images['y4'])
        
        
    #> if image dtype is numpy array
    if isinstance(images, np.ndarray):
        
        #> checking if quad
        numIms = len(images)
        if numIms not in [4, 5]: 
            import modules.error as error
            error.phrase(f'Cannot calculate image angles from non-quad: {images}')
    
        #> interating through each set of images
        for i in range(min(4, numIms)):
            dists[f'd0{i+1}'] = dist(origin, images[i])
            
        #> image points
        img1 = images[0]
        img2 = images[1]
        img3 = images[2]
        img4 = images[3]
    
    #> getting distances between the images
    dists['d13'] = dist(img1, img3)
    dists['d14'] = dist(img1, img4)
    dists['d23'] = dist(img2, img3)
    dists['d24'] = dist(img2, img4)
    
    return dists
    
    
""" #> NEW ANGLE FNS =================
================================== """

#> calculates the angle between two points
def angle(v1, v2):
    
    #> dot products
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    #> if vector is zero
    if magnitude_v1 == 0 or magnitude_v2 == 0: 
        import modules.error as error
        error.phrase('Vector is zero in angle calculation')

    #> length
    cosine_theta = dot_product / (magnitude_v1 * magnitude_v2)
    # cosine_theta = np.clip(cosine_theta, -1.01, 1.01)
    
    #> calculating angle
    angle_radians = np.arccos(cosine_theta)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


#> rotates point around origin
def rotate(origin, point, angle):

    #> unpacking
    ox, oy = origin
    px, py = point

    #> rotating
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return (qx, qy)

#> determines if two lines intersect ([p11, p12] -- [p21, p22])
def intersect(p11, p12, p21, p22):
    
    #> creating lines
    line1 = geo.LineString([p11, p12])
    line2 = geo.LineString([p21, p22])
    
    #> determining if intersection
    intersection = line1.intersection(line2)
    
    #> returning
    if isinstance(intersection, geo.point.Point):
        return True
    else:
        return False

#> determines the geometric, FSQ angles from origin to images
def imAngles(origin, images):
    
    #> declarations
    angles = {}
    
    #> if image dtype is dictionary
    if isinstance(images, dict):
    
        #> checking if quad
        numIms = len([item for item in images.keys() if 'x' in item])
        if numIms != 5: 
            import modules.error as error
            error.phrase(f'Cannot calculate image angles from non-quad: {images}')
        
        #> image points
        img1 = np.array((images['x1'], images['y1']))
        img2 = np.array((images['x2'], images['y2']))
        img3 = np.array((images['x3'], images['y3']))
        img4 = np.array((images['x4'], images['y4']))
        
    #> if image dtype is numpy array
    if isinstance(images, np.ndarray):
        
        #> checking if quad
        numIms = len(images)
        if numIms not in [4, 5]: 
            import modules.error as error
            error.phrase(f'Cannot calculate image angles from non-quad: {images}')
        
        #> image points
        img1 = np.array(images[0])
        img2 = np.array(images[1])
        img3 = np.array(images[2])
        img4 = np.array(images[3])
    
    origin = np.array(origin)
    
    #> t12
    #> rotating images such that img1 lies along the x-axis
    t1 = math.atan2(img1[1], img1[0])  # radians
    rimg1 = rotate(origin, img1, -t1)
    rimg2 = rotate(origin, img2, -t1)
    rimg3 = rotate(origin, img3, -t1)
    
    #> determining t12
    t12 = math.atan2(rimg2[1], rimg2[0]) * (180/np.pi)
    t13 = math.atan2(rimg3[1], rimg3[0]) * (180/np.pi)
    angles['t12'] = t12
    
    #> checking if img3 is inscribed
    # if 0 <= t12 <= 180.0: 
    #     if t12 < t13: angles['t12'] = 360 + t12 
    # else: 
    #     if abs(t12) > abs(t13): angles['t12'] = abs(t12) 
    if not intersect(img1, img2, img3, [0,0]): 
        angles['t12']= 360 - angles['t12']
    angles['t12'] = abs(angles['t12'])
        
    #> t23
    t2 = math.atan2(img2[1], img2[0])  # radians
    rimg3 = rotate(origin, img3, -t2)
    angles['t23'] = abs(math.atan2(rimg3[1], rimg3[0]) * (180/np.pi))
    
    #> t34
    t3 = math.atan2(img3[1], img3[0])  # radians
    rimg4 = rotate(origin, img4, -t3)
    angles['t34'] = abs(math.atan2(rimg4[1], rimg4[0]) * (180/np.pi))
    
    #> checking if img2 is inscribed
    if not intersect(img3, img4, img2, [0,0]): angles['t34']= 360 - angles['t34']
    
    return angles


""" #> OLD ANGLE FNS =================
================================== """

# distance formula
def distForm(x1,x2,y1,y2):
    dist = (((x2-x1)**2) + ((y2-y1)**2))**0.5
    return dist

# law of cosines
def lawCos(a,b,c):
    val = (a**2 + b**2 - c**2) / (2*a*b)
    agl = math.acos(np.clip(val, -1.0, 1.0)) * (180/(np.pi))
    return agl
    
# getting angles
def getAngles(origin, images):
    
    #> declarations
    angles = {}
    
    #> getting distances
    dists = imDists(origin, images)
    angles['t12'] = lawCos(dists['d01'], dists['d02'], distForm(images['x1'], images['x2'], images['y1'], images['y2']))
    angles['t23'] = lawCos(dists['d02'], dists['d03'], distForm(images['x2'], images['x3'], images['y2'], images['y3']))
    angles['t34'] = lawCos(dists['d03'], dists['d04'], distForm(images['x3'], images['x4'], images['y3'], images['y4']))
    
    return angles

""" #> NEWER ANGLES ==================
================================== """

#> calculates polar angle of cartesian point [0, 360)
def polar_angle_deg(p):
    return np.degrees(np.arctan2(p[1], p[0])) % 360

#> calculates counter-clockwise angle [0, 360)
def ccw_angle_deg(a, b):
    return (b - a) % 360

#> check if angle is inscribed
def angle_containing_point(pa, pb, pc):

    #> getting angles
    a = polar_angle_deg(pa)
    b = polar_angle_deg(pb)
    c = polar_angle_deg(pc)

    #> counter-clockwise relative angles
    ab = ccw_angle_deg(a, b)
    ac = ccw_angle_deg(a, c)

    #> if inscribed
    if ac <= ab:
        return ab
    else: #> if not inscribed
        return 360 - ab

#> calculates the relative angles of quad images
def new_imAngles(origin, images):
    
    #> declaration
    angles = {}
    
    #> if format is dictionary
    if isinstance(images, dict):
        numIms = len([item for item in images.keys() if 'x' in item])
        if numIms != 5: # if not quad
            import modules.error as error
            error.phrase(f'Cannot calculate image angles from non-quad: {images}')
        
        #> saving image positions
        img1 = np.array((images['x1'], images['y1']))
        img2 = np.array((images['x2'], images['y2']))
        img3 = np.array((images['x3'], images['y3']))
        img4 = np.array((images['x4'], images['y4']))
        
    #> if format is numpy
    elif isinstance(images, np.ndarray):
        numIms = len(images)
        if numIms not in [4, 5]: 
            import modules.error as error
            error.phrase(f'Cannot calculate image angles from non-quad: {images}')
        
        #> saving image positions
        img1 = np.array(images[0])
        img2 = np.array(images[1])
        img3 = np.array(images[2])
        img4 = np.array(images[3])
    
    #> angle between img1 and img2 that contains img3
    angles['t12'] = angle_containing_point(img1, img2, img3)

    #> angle between img2 and img3
    angles['t23'] = angle(img2, img3)

    #> angle between img3 and img4 that contains img2
    angles['t34'] = angle_containing_point(img3, img4, img2)

    return angles


""" #> FSQ ===========================
================================== """

#> determines dt23 from angles FOR MANY QUADS
def allFSQ(allAngles):
    
    #> declarations
    allt23 = {}
    alldt23 = {}
    
    #> iterating through each galaxy
    for i, key in enumerate(allAngles.keys()):
    
        #> changing angles format
        angles = dictToArr(allAngles[key])
        
        #> unpacking angles
        t12 = angles[:,0] * (np.pi/180)
        t23 = angles[:,1] * (np.pi/180)
        t34 = angles[:,2] * (np.pi/180)
        
        #> the FSQ! (woof) (from Woldesenbet & Williams 2012, eq18)
        t23_fsq = ( -5.792 + (1.783 * t12) + (0.1648 * t12**2) - (0.04591 * t12**3)
        - (0.0001486 * t12**4) + (1.784 * t34) - (0.7275 * t34 * t12)
        + (0.0549 * t34 * t12**2) + (0.01487 * t34 * t12**3) + (0.1643 * t34**2)
        + (0.05493 * t34**2 * t12) - (0.03429 * t34**2 * t12**2) - (0.04579 * t34**3)
        + (0.01487 * t34**3 * t12) - (0.0001593 * t34**4) )
        
        #> getting delta theta_23
        dt23 = t23 - t23_fsq
        
        #> saving
        allt23[key] = t23 * (180/np.pi)
        alldt23[key] = dt23 * (180/np.pi)
    
    return allt23, alldt23


#> determines dt23 from angles
def FSQ(angles):
    
    #> unpacking angles
    t12 = angles['t12'] * (np.pi/180)
    t23 = angles['t23'] * (np.pi/180)
    t34 = angles['t34'] * (np.pi/180)
    
    #> the FSQ! (woof) (from Woldesenbet & Williams 2012, eq18)
    t23_fsq = ( -5.792 + (1.783 * t12) + (0.1648 * t12**2) - (0.04591 * t12**3)
    - (0.0001486 * t12**4) + (1.784 * t34) - (0.7275 * t34 * t12)
    + (0.0549 * t34 * t12**2) + (0.01487 * t34 * t12**3) + (0.1643 * t34**2)
    + (0.05493 * t34**2 * t12) - (0.03429 * t34**2 * t12**2) - (0.04579 * t34**3)
    + (0.01487 * t34**3 * t12) - (0.0001593 * t34**4) )
    
    #> getting delta theta_23
    dt23 = t23 - t23_fsq
    
    #> converting
    t23 *= (180/np.pi)
    dt23 *= (180/np.pi)
    
    return t23, dt23


#> fitting new FSQ?
def fitFSQ(inFile):
    
    #> imports
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    #> loading in data
    data = np.load(inFile)
    
    #> data
    X = np.column_stack((data[:,0], data[:,2]))  # shape (N, 2)
    Y = data[:,1]                                # shape (N,)
    
    #> create polynomial features
    degree = 5
        
    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    coeffs = model.coef_
    intercept = model.intercept_
    
    feature_names = poly.get_feature_names_out()
    for name, coef in zip(feature_names, coeffs):
        print(name, coef)
    
    print("intercept:", intercept)
    
    return


""" #> ERROR DETERM ==================
================================== """

#> finds the error on the geometric angles
def angleError(origin, images):
    
    
    
    return


#> finds the error on the geometric distance
def distError(origin, images):
    
    
    
    return


""" #> 7D ============================
================================== """
    
#> returns the 7-dimensional space of lensing observables!
def sevenD(origin, images):
    
    #> getting info
    angles = imAngles(origin, images)
    dists = imDists(origin, images)
    
    array = [angles['t12'], angles['t12'], angles['t12'], 
             dists['d02']/dists['d01'], 
             dists['d03']/dists['d01'], 
             dists['d04']/dists['d01'], 
             dists['d01']]
    
    return array

#> returns lensing observables  (can be modified)
def threeD(origin, images):
    
    #> getting info
    angles = imAngles(origin, images)
    dists = imDists(origin, images)
    t23, dt23 = FSQ(angles)
    
    return [t23, dt23, dists['d04']/dists['d01']]

#> returns lensing observables  (can be modified)
def lensObs(origin, images, observables):
    
    #> declaration
    lens_obs = []
    
    #> getting info
    angles = new_imAngles(origin, images)
    dists = imDists(origin, images)
    distRats = {'d2/d1': dists['d02']/dists['d01'], 
                'd3/d1': dists['d03']/dists['d01'],
                'd4/d1': dists['d04']/dists['d01'],
                'd3/d2': dists['d03']/dists['d02']}
    t23, dt23 = FSQ(angles)
    ims = {'x1': float(images[0][0]), 'y1': float(images[0][1]),
           'x2': float(images[1][0]), 'y2': float(images[1][1]), 
           'x3': float(images[2][0]), 'y3': float(images[2][1]),
           'x4': float(images[3][0]), 'y4': float(images[3][1]), 
           'x5': float(images[4][0]), 'y5': float(images[4][1])}
    
    #> combining dictionaries
    data = angles | dists | distRats | {'dt23': dt23} | ims
    
    #> collecting requested obs
    for obs in observables:
        try: lens_obs.append(data[obs])
        except: 
            import modules.error as error
            error.phrase(f'COULD NOT FIND OBSSERVABLE {obs}')
        
     #print(lens_obs)
    # print(len(lens_obs))
    
    return lens_obs


""" #> PLOTTING ======================
================================== """

#> saving parameters
dpi = 600
imSize = 80

#> plots all images w/ angles and distances
def images(ims, origin=(0,0)):
    
    #> getting angles
    angles = imAngles(origin, ims)
    dists = imDists(origin, ims)
    t23, dt23 = FSQ(angles)
    
    #> initializing plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title('Images + Angles', fontweight='bold')
    ax.set_xlabel('RA [arcsec]'); ax.set_ylabel('DEC [arcsec]')
    ax.grid(ls=':', alpha=0.3)
    ax.invert_xaxis()
    
    #> plotting the images
    ax.scatter(ims[:,0],
               ims[:,1],
               zorder=5, s=imSize)
    
    #> annotate angles
    for i, key in enumerate(angles.keys()):
        ax.annotate(f'{key}={angles[key]:.1f}', (0.75, 0.25-(i*0.05)),
                    xycoords='figure fraction', fontweight='bold',
                    c='k')
    ax.annotate(f'dt23={dt23:.2f}', (0.75, 0.25-(3*0.05)),
                xycoords='figure fraction', fontweight='bold',
                c='k')
        
    #> annotate angles
    for i, key in enumerate(dists.keys()):
        if i > 3: continue
        ax.annotate(f'{key}={dists[key]:.5f}', (0.15, 0.25-(i*0.05)),
                    xycoords='figure fraction', fontweight='bold',
                    c='k')
        
    #> plotting distances
    for i, im in enumerate(ims): 
        if i == 4: continue
        ax.plot((0, im[0]), (0, im[1]), c='k')
        ax.annotate(f'{i+1}', (im[0], im[1]), 
                    size=20, fontweight='bold', c='k', zorder=6)

        #> plotting angles (this is a pain in the ass)
        if i > 2: continue
        value = np.sign(im[1])
        if i != 1:
            angle_arc = Arc((0, 0), width=0.2*(i+1), height=0.2*(i+1), 
                            theta1=angle((1,0), (im))*value, 
                            theta2=angle((1,0), (ims[i+1]))*np.sign(ims[i+1][1]),
                            color='r', linestyle='-', linewidth=3, zorder=6)
        else:
            angle_arc = Arc((0, 0), width=0.2*(i+1), height=0.2*(i+1), 
                        theta2=angle((1,0), (im))*value, 
                        theta1=angle((1,0), (ims[i+1]))*np.sign(ims[i+1][1]),
                        color='r', linestyle='-', linewidth=3, zorder=6)
        ax.add_patch(angle_arc)
    
    #> legend
    plt.legend(framealpha=1, edgecolor='1', prop={'weight':'bold'})
    
    plt.show(); fig.clear(); plt.close()
    del fig, ax
    
    
#> plots the 2D representation of the FSQ
def fsq2D(t23, dt23):
    
    #> initializing plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.grid(ls=':', alpha=0.3)

    ax.set_ylim(-5, 5)
    ax.set_xlim(80, 100)
    
    ax.set_ylim(-15, 15)
    ax.set_xlim(0, 100)
    
    ax.scatter(t23, dt23, alpha=0.1, c='m', s=4)
    ax.set_xlabel(r'$\theta_{23}$', fontsize=15, fontweight='bold')
    ax.set_ylabel(r'$\Delta\theta_{23}$', fontsize=15, fontweight='bold')
    
    plt.show()
    plt.close()
    
    
#> plotting d3/d2
def d3d2(inFile):
    
    #> reading file
    data = np.load(inFile, allow_pickle=True)
    
    fontsize = 15
    
    #> initializing plot
    fig, axes = plt.subplots(1, 3, figsize=(6*3+0.1*3, 6))
    
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.5)

        if i in [0,1]:
            ax.set_ylabel(r'$\mathbf{\Delta\theta_{23}}$', fontsize=fontsize)
            ax.set_ylim(-20, 20)
        
        if i == 0:
            ax.set_xlabel(r'$\mathbf{\theta_{23}}$', fontsize=fontsize, labelpad=10)
            ax.set_xlim(0, 100)
            ax.scatter(data[:,1], data[:,6], alpha=0.2, s=5, c='k')
            
        if i == 1:
            ax.set_xlabel(r'$\mathbf{1 - (d_3/d_2)}$', fontsize=fontsize, labelpad=10)
            ax.scatter(1-data[:,7], data[:,6], alpha=0.2, s=5, c='k')
            
        if i == 2:
            ax.set_xlabel(r'$\mathbf{\theta_{23}}$', fontsize=fontsize, labelpad=10)
            ax.scatter(data[:,1], 1-data[:,7], alpha=0.2, s=5, c='k')
            ax.set_ylabel(r'$\mathbf{1 - (d_3/d_2)}$', fontsize=fontsize)
           #  ax.set_ylim(-5, 5)
            
    plt.show()
    
    return
    

#> plots the histogram of angles
def obsHist(df, **kwargs):
    
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator, NullFormatter
    
    #> labels
    keys = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    labels = [r'$\mathbf{\theta_{12}}$', r'$\mathbf{\theta_{23}}$', r'$\mathbf{\theta_{34}}$', 
              r'$\mathbf{d_2/d_1}$', r'$\mathbf{d_3/d_1}$', r'$\mathbf{d_4/d_1}$', 
              r'$\mathbf{\Delta\theta_{23}}$']
    
    #> unpacking
    comp_df = kwargs.get('comp_df', None)
    
    #> declarations
    c = '#C00000'
    bb = '#0070C0'
    bins = 15
    
    #> initializing plots
    fig, axes = plt.subplots(2, 3, figsize=(3*6+2, 2*6))
    
    #> plotting!
    for key, label, ax in zip(keys, labels, axes.flat):
        
        #> grid, label, limits
        ax.grid(ls=':', alpha=0.5)
        ax.set_xlabel(label, fontsize=15, labelpad=10)
        # ax.set_xlim(min(df[key]), max(df[key]))
        
        #> adjusting color
        if 'd' in key: color=bb
        else: color = c
        
        #> plotting!
        ax.hist(df[key], bins, facecolor=color)
        if comp_df is not None: # if there is a comparison dataset
            ax.hist(comp_df[key], bins, histtype='step', edgecolor='k', lw=5)
            
        #> ticks
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=15)
        
    # plt.savefig('./figs/obs_histograms.png', bbox_inches='tight', dpi=100)
    plt.show()
    
    return


#> plots quad(s) from df
def plotdfQuad(df, num=0):
    
    ims = df[['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5']].to_numpy()[num].reshape(5,2)
    
    images(ims)
    
    return

""" #> ERROR CHECKING ================
================================== """

#> checks quads to find errors in angles
def errQuads(inFile):
    
    import params
    import lensing
    
    #> loading data
    data = np.load(inFile)[::10]
    
    #> getting angles for every quad
    obs = []
    for ims in data:
        obs.append(lensObs((0,0), ims, ['t23', 'dt23']))
    obs = np.array(obs)
    
    t23_limit = 80
    dt23_limit = 0.7
    obs_error = obs[ (obs[:, 1] > dt23_limit) & (obs[:, 0] > t23_limit) ]
    img_error = data[ (obs[:, 1] > dt23_limit) & (obs[:, 0] > t23_limit) ]
    
    for i in range(len(img_error)):
        images(img_error[i], (0,0))
    
    #> plotting
    fsq2D(obs[:,0], obs[:,1])
    # fsq2D(obs_error[:,0], obs_error[:,1])
    
    #. getting grid
    x, y = params.grid(nph=50)
    
    # for i in range(1):
        
    
    return


""" #> MAIN FUNCTION =================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #inFile = './data/2602251619-maybefix.npy'
    # errQuads(inFile)
    #fitFSQ('./data/fsq/2603011802-fsq.npy')
    
    #d3d2('./data/2605111308d3d2-1.npy')
    #d3d2('./data/2605111318d3d2-3.npy')
    #d3d2('./data/2605111335d3d2-4.npy')
    #d3d2('./data/2605111341d3d2-5.npy')
    #d3d2('./data/2605151514d3d2-5.npy')
    #d3d2('./data/2605181351d3d2-5.npy')
    #d3d2('./data/2605181411d3d2-5.npy')
    
    # config('./data/2606241416fsq_q0.85_std0.05_fit.npy')
    
    # obs_config('../notes/projects/quads/all_obs_quads.csv')
    
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23'] 
    
    import modules.parse as parse
    file = './data/baseline/2606301434q75_m3.npy'
    # data, keys = parse.fromNPY(file)
    # im_obs = []
    # for im in data['images']:
    #     im_obs.append(lensObs((0,0), im, observables))
    # data['im_obs'] = np.array(im_obs)
    # np.save(file, data)
    
    # end
# thank