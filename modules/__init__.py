#> name: __init__.py

#> play opus from command line with 'play'
def play():
    
    #> checking if playsound is imported
    import os
    import multiprocessing
    try: from playsound import playsound
    except ImportError: downloadPlaySound()
    try: from playsound import playsound
    except ImportError: print('> Unable to play :('); return
    
    #> playing
    sound_file = os.path.dirname(__file__)+'/../info/opus.mp3'
    p = multiprocessing.Process(target=playsound, args=(sound_file,))
    p.start()

    #> stopping
    input('> Press enter to stop.')
    p.terminate()
    p.join() 
    
    return

#> pip installs playsound if not in environment
def downloadPlaySound():
    
    import subprocess, sys
    try: subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'playsound==1.2.2'])
    except subprocess.CalledProcessError as e:
        print(f"Error installing vlc: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    from playsound import playsound
    return