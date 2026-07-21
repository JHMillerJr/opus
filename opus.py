#> name: opus.py
#> author: John Miller Jr
#> descrp: main python file for the lensing code Opus

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import numpy as np

#> checking to see if GPU is available
# if error.checkGPU(): import cupy as cp

#> declarations
dataDir = './data/'
figDir = './figures/'
sys.path.append(os.path.dirname(__file__)) 


""" #> FUNCTIONS =====================
================================== """




""" #> COMMAND LINE ==================
================================== """

#> takes in arguements from command line & parses
def commandLine(argument):
    
    #> prunes script name
    args = argument[1:]
    
    #> plays opus by ryuichi sakamoto :)
    if args[0] == 'play': 
        import modules
        modules.play()
    
    if args[0] == '-help':
        print('> Can choose from:')
        print('> ')
        
    #> runs specific file
    # if '.txt' in args[0]:
    #    runMode(args[0])
        
    sys.exit() # nothing in main runs after
    return     # not triggered, but kept



""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> checking command line args 	
    if len(sys.argv) > 1: commandLine(sys.argv)
    
    #> generating quad population from galaxy population
    # observables = ['t23', 'dt23', 'd4/d1', 't12', 't23', 't34']
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'dt23']
    
    fileName = dataDir+'2603021417-same'+'.npy'
    # plot3D(fileName, observables)
    

    # end
# thank