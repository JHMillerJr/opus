#> name: error.py
#> author: John Miller Jr
#> descrp: functions to help handle errors

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import inspect

#> color imports
from colorama import Fore, Style


""" #> SYSTEM CHECK ==================
================================== """

#> check to see if GPU is available, True/False
def checkGPU():
    """
    Returns the array module to use: cupy if GPU is available, otherwise numpy.
    """
    try:
        import cupy as cp
        # Check if there is at least one GPU device
        if cp.cuda.runtime.getDeviceCount() > 0:
            return True
        else:
            return False
    except:
        return False

""" #> PRINT STMNTS / CUT OFFS =======
================================== """

#> prints error if key word is not found
def inError(word):
    """
    word: CANNOT FIND {word}
    
    quits program
    """
    
    #> getting info of error loc
    cf = inspect.currentframe()
    pyName = os.path.basename(inspect.getfile(cf))
    line = cf.f_back.f_lineno
    
    #> printing error
    print(Fore.RED + Style.BRIGHT + 
          f'▄▀▄▀ CANNOT FIND \'{word}\'! [py={pyName}, line={line}] ▀▄▀▄')
    print(Style.RESET_ALL)
    sys.exit() # quitting
    
#> prints error due to given phrase
def phrase(string):
    """
    string: error statement
    
    quits program
    """
    
    #> getting info of error loc
    cf = inspect.currentframe()
    pyName = os.path.basename(inspect.getfile(cf))
    line = cf.f_back.f_lineno
    
    #> printing error
    print(Fore.RED + Style.BRIGHT + 
          f'▄▀▄▀ \'{string}\'! [py={pyName}, line={line}] ▀▄▀▄')
    print(Style.RESET_ALL)
    sys.exit() # quitting
    
#> highlights a certain phrase
def highlight(string):
    """
    string: error statement
    """
    print(Fore.RED + Style.BRIGHT +  f'> {string}')
    print(Style.RESET_ALL)
    return
    

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    # end
# thank