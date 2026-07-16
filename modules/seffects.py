#> name: seffects.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import math
import numpy as np
import pandas as pd

#> modules
import modules.parse as parse


""" #> SELECTION EFFECTS =============
================================== """

#> resolution selection effect
def resolution(images, resolution=0.03, factor=3):

    diff = images[:, None, :] - images[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist2, np.inf)

    return ( np.sqrt(dist2.min()) >= resolution*factor )


#> magnification selection effects
def magnification(im_mags):
    
    #> total magnification
    ttl_mag = np.sum(im_mags)
    
    #> brightest mag
    max_mag = np.max(im_mags)
    
    # print(im_mags[:100])
    
    #> if bright enough
    if (ttl_mag >= 30) | (max_mag >= 2000):
        return True
    
    return False


""" #> ANALYZING BATCH ===============
================================== """

#> testing selection effects on batch of quads
def batch(inFile):
    
    #> imports
    import modules.configuration as configuration
    import modules.geometry as geometry
        
    #> getting configurations
    df = configuration.config(inFile, verbose=True)
    allImages = df[['x1','y1','x2','y2','x3','y3','x4','y4','x5','y5']].to_numpy()
    allMags = df[['mu1', 'mu2', 'mu3', 'mu4', 'mu5']].to_numpy()
    
    #> applying selection effects
    scores = []
    for im, mag in zip(allImages, allMags):
        scores.append([resolution(im.reshape(5,2)), magnification(mag)])
    df = pd.concat([df, pd.DataFrame(np.array(scores), columns=['a_seff', 'm_seff'])], axis=1)
    
    #> making master dataframe
    df['selected'] = (df['a_seff'] & df['m_seff'])
    selected_df = df[ df['selected'] == True ]
    not_selected_df = df[ df['selected'] == False ]
    
    #> plotting selected vs. not selected
    geometry.obsHist(selected_df, comp_df=not_selected_df)
    
    # configuration.plotAllAngles(df)
    
    #> plotting tri panel
    # data, _ = parse.fromNPY(inFile)
    # sort_value = 't23'
    # configuration.triPanel(selected_df, sort_value, data['lens'], comp_df=not_selected_df)
        
    return


""" #> MAIN FUNCTION =================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> declarations
    name = '2606301438q75' 
    inDir = './data/baseline/'
    inFile = inDir + name
    
    ### THESE SHOULD PRIMARILY BE APPLIED WHEN GENERATING THE LENSES
    ### AS FOR RIGHT NOW, I AM JUST USING A FILE TO TEST THEM
    ### VECTORIZING THEM IS NOT NEEDED, BUT WOULD BE GOOD
    
    #> loading in data
    # df, data = parse.fromNPYtoDF(inFile)
    data, keys = parse.fromNPY(inFile)
    images = data['images']
    im_mags = data['im_mags']
    # print(data['bprofiles'][0])
    
    batch(inFile)
    
    #> applying selection effects
    scores = []
    for im, mag in zip(images, im_mags):
        scores.append([resolution(im), magnification(mag)])
    df = pd.DataFrame(np.array(scores), columns=['a_seff', 'm_seff'])
    
    #> making master dataframe
    df['selected'] = (df['a_seff'] & df['m_seff'])
    # print(np.sum(df['selected']))
    
    # end
# thank