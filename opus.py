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
opusDir = os.path.dirname(__file__)


""" #> FUNCTIONS =====================
================================== """




""" #> COMMAND LINE ==================
================================== """

#> takes in arguements from command line & parses
def commandLine(argument):
    
    #> imports
    import argparse
    
    #> parses command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='plays Opus by Ryuichi Sakamoto')
    parser.add_argument('--gentemp', nargs='?', type=str, const='./', help='copies gentemp.py to requested dir [df=%(const)s]')
    args, unknown = parser.parse_known_args()
    args.args = unknown
    
    # print(args)
    
    #> plays opus by ryuichi sakamoto :)
    if args.play: 
        import modules
        modules.play()
        
    #> copying gentemp to dir
    if args.gentemp:
        
        #> imports
        import shutil
        import modules.error as error
        from pathlib import Path
        
        #> declarations
        default_file_name = 'gentemp.py'
        src = opusDir + './modules/gentemp.py'
        
        #> seeing if folder or filename
        dst = args.gentemp.replace('\\','/')
        if '.py' not in dst:
            if dst[-1] != '/': dst += '/'
            dst += default_file_name
            
        #> getting folder tree
        parent_dir = os.path.dirname(dst)
        # file_name = os.path.basename(dst) + '.npy'
            
        #> checking to ensure parent dir exists
        if not os.path.isdir(parent_dir):
            error.highlight(f'The directory {parent_dir} does not exist. Creating...')
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
        #> copying file to destination
        shutil.copyfile(src, dst)
        print(f'> gentemp.py has been copied to {dst}')
        
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