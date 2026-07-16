#> name: memtrack.py
#> author: John Miller Jr
#> descrp: tracks peak memory usage of provided .py (slurm-ready)

""" #> IMPORTS =======================
================================== """

import os
import sys
import time
import csv

import numpy as np

import psutil
import argparse
import subprocess
import threading

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


""" #> CONSTANTS =====================
================================== """

GiB = 1024 ** 3 # bytes --> GiB
dpi = 200 
margin = 0.25 # safety margin for memory allocation (1+margin)


""" #> GETS CHILDREN =================
================================== """

#> gets total memory of process tree (parent + children)
def get_tree_rss(proc):

    #> declarations
    ttl = 0

    #> getting parent
    try:
        ttl += proc.memory_info().rss
    except psutil.NoSuchProcess:
        return 0

    #> getting children total
    for child in proc.children(recursive=True):
        try:
            ttl += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue

    return ttl


""" #> MONITOR =======================
================================== """

#> function that actually monitors the memory
def monitor_proc(proc, interval, log_file, timeline):
    
    #> starting time
    srt = time.time()

    #> saving to log file as it tracks
    with open(log_file, "w", newline="") as f:
        
        #> writer + headers
        writer = csv.writer(f)
        writer.writerow(["t_sec", "rss_gib"])

        #> checks process every interval
        while proc.poll() is None:           # still running

            try:      # if still going
                p = psutil.Process(proc.pid) # process id
                rss = get_tree_rss(p)        # memory usage
            except psutil.NoSuchProcess:
                break # not running anymore
            
            #> writes & saves info
            t = time.time() - srt
            writer.writerow([t, rss / GiB])
            timeline.append((t, rss / GiB))

            #> waits
            time.sleep(interval)

    return


""" #> SINGLE RUN ====================
================================== """

#> runs the .py script
def run(script, args, interval, workers):
    
    #> adding workers to args
    args = list(np.hstack([args, ['--workers', str(workers)]]))

    #> initializes command to run .py
    cmd = [sys.executable, script] + args
    print(f"> running: {' '.join(cmd)}")
    print("\n==============================")    
        
    #> logging dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(dir_path, 'memlog/')
    os.makedirs(directory, exist_ok=True)
    
    #> creates log flie
    fileName = f'memlog_{os.path.basename(script)}_{int(time.time())}.csv'
    log_file = os.path.join(directory, fileName)


    #> runs the command
    proc = subprocess.Popen(cmd)
    
    #> main monitor function
    timeline = []
    def monitor():
        monitor_proc(proc, interval, log_file, timeline)
        
    #> starting threading
    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    proc.wait()
    t.join()
    
    #> prints
    endPrint(timeline, workers, log_file)

    return log_file, timeline


#> printing the run stats & info
def endPrint(timeline, workers, log_file):

    #> if nothing was done
    if not timeline: return

    #> gets the max time & value
    peak_t, peak_val = max(timeline, key=lambda x: x[1])
    
    #> memory run values
    print("\n------------------------------")
    print(f'> Num workers = {workers}')
    print(f'> Peak  = {peak_val:.3f} GiB')
    print(f'> @time = {peak_t:.2f} sec')
    print(f'> Final = {timeline[-1][1]:.3f} GiB')
    print(f"> Log@  = {log_file}")
    
    #> for slurm scheduling
    slurm_mem = peak_val * (1 + margin)
    print(f'> Request --mem ~ {slurm_mem:.3f} GiB for {workers} workers ({margin*100:.0f}% safety margin)')
    print(f'> Request ~ {slurm_mem/workers:.3f} GiB/worker')
    print("==============================\n")
    
    return


""" #> SWEEP  ========================
================================== """

#> fits quadratic to workers/peak
def sweepFit(results):
    
    #> fits quadratic to results
    fit = np.polyfit(results[:,0], results[:,1], 2)
    func = np.poly1d(fit)
    
    return func


#> most efficient # of workers per memory usage
def mostEff(func, max_workers=128):

    ws = np.arange(1, max_workers + 1)
    eff = func(ws) / ws   # GiB per worker
    i = np.argmin(eff)

    return ws[i], eff[i]
    

#> plots the sweep run
def sweepPlot(results, fileName):
    
    #> imports
    import matplotlib.pyplot as plt
    
    #> directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(dir_path, 'memlog/')
    
    #> initializing plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.set_xlabel('# Workers', fontweight='bold', labelpad=10, fontsize=12)
    ax.set_ylabel('Peak [GiB]', fontweight='bold', labelpad=10, fontsize=12)
    ax.grid(ls=':', alpha=0.5)
    
    #> fitting quadratic
    func = sweepFit(results)
    
    #> x range
    x = np.linspace(1, 128, 1000)
    
    #> plotting!
    ax.plot(x, func(x), label='fit', lw=3)
    ax.scatter(results[:,0], results[:,1], label='data', s=50)
    
    #> saving
    plt.savefig(directory+fileName[:-3]+'png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return


#> conducts sweep of workers
def sweep(script, interval, sweep):
    
    #> imports
    import math
    
    #> ensures not over reaching
    max_workers = min(sweep, os.cpu_count())
    
    #> initializing workers
    log2 = (math.log(max_workers, 2))
    workers_list = 2**(np.arange(0,  int(log2) + 1) )
    if log2 % 1 != 0:
        workers_list = np.hstack([workers_list, [int(max_workers)]])
        
    #> setting numMocks
    numMocks_per_worker = 2
    
    #> starting print
    print(f'> Sweeping through workers: {workers_list}')
    if max_workers == os.cpu_count(): print(f'> max_workers hit cpu limit ({os.cpu_count()})')
    
    #> calls runs for each worker in list       
    results = []
    for w in workers_list:
        args = np.array(['--numMocks', str(w * numMocks_per_worker)])
        log_file, timeline = run(script, args, interval, w)  
        peak_t, peak_val = max(timeline, key=lambda x: x[1])
        results.append([w, peak_val, timeline[-1][0]])
    results = np.array(results)
    
    #> saving results
    
    #> logging dir
    dir_path = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(dir_path, 'memlog/')
    os.makedirs(directory, exist_ok=True)
    
    #> creates log flie
    fileName = f'sweep_{os.path.basename(script)}_{int(time.time())}.csv'
    log_file = os.path.join(directory, fileName)
    
    #> saving to log file as it tracks
    with open(log_file, 'w', newline='') as f:
        
        #> writer + headers
        writer = csv.writer(f)
        writer.writerow(['worker', 'peak_gib', 'ttl_time', 'w_eff'])
        
        #> writing all lines
        for w, peak, ttl_time in results:
            writer.writerow([w, peak, ttl_time, peak/w])
    
    #> plotting results
    sweepPlot(results, fileName)
    
    #> declarations
    max_workers = 128
    max_memory = 128
    
    #> fitting quad to sweep results
    func = sweepFit(results)
    
    #> evaluating
    max_workers_memory = func(max_workers)
    
    #> max workers given max memory
    most_workers = 0
    for i in range(max_workers):
        if func(i+1) < max_memory:
            most_workers = i+1
        else:
            break
    mem_margin = max_memory - func(most_workers)
    
    #> most efficient
    eff_w, eff_mem_per_w = mostEff(func)
    ttl_eff = eff_w * eff_mem_per_w
    
    #> prints
    print('#############################')
    print('#######> Slurm Info <########')
    print('#############################')
    print()
    print(f'> {max_workers} workers will need {max_workers_memory:.2f} GiB or {max_workers_memory*(1+margin):.2f} GiB (w/ {margin*100}% safety margin)')
    print(f'> Can request a maximum of {most_workers} workers with a memory of {max_memory} GiB and {mem_margin:.3f} GiB left over')
    print(f'> Most efficent # of workers is {eff_w} at {eff_mem_per_w:.3f} GiB/ worker for a total of {ttl_eff:.3f} GiB needed or {ttl_eff*(1+margin):.3f} GiB (w/ {margin*100}% safety margin)')
    print()
    
    return



""" #> MAIN ==========================
================================== """

#> main
if __name__ == "__main__":

    #> name
    print("> " + os.path.basename(__file__))
    
    #> making sure run in a terminal
    if not sys.stdout.isatty() or 'SPYDER_ARGS' in os.environ or 'SPYDER_DISPLAY' in os.environ:
        import module.error as error
        error.phrase('Please run in a terminal!')

    #> parses command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('script')
    parser.add_argument('--interval', type=float, default=0.01, help='check-in time interval')
    parser.add_argument('--sweep', type=int, default=None, help='max # workers in sweep')
    args, unknown = parser.parse_known_args()
    args.args = unknown
    
    #> if sweeping
    if args.sweep is not None:
        sweep(args.script, args.interval, args.sweep)
        sys.exit()
        
    #> gets number of workers (if requested)
    if "--workers" in args.args:
        i = args.args.index("--workers")
        workers = int(args.args[i + 1])
        args.args.pop(i+1) # removing bc will be added again later
        args.args.pop(i)
    else:
        workers = 1

    #> runs the memory tracker
    log_file, timeline = run(args.script, args.args, args.interval, workers)


    # end
# thank