#> name: slurm_tuner.py
#> author: John Miller Jr
#> descrp: slurm memory + multiprocessing scaling tuner (linux only)

""" #> IMPORTS =======================
================================== """

import os
import sys
import time
import csv
import psutil
import argparse
import subprocess
import threading


""" #> CONSTANTS =====================
================================== """

GIB = 1024 ** 3


""" #> OS CHECK ======================
================================== """

def check_linux():

    if os.name != "posix":
        print("> ERROR: this tuner is linux-only.")
        print("> detected OS:", os.name)
        print("> exiting...")
        sys.exit(1)

    if sys.platform.startswith("win"):
        print("> ERROR: Windows detected.")
        print("> this tuner is designed for SLURM (Linux only).")
        sys.exit(1)


""" #> MEMORY ========================
================================== """

def get_tree_rss(proc):

    total = 0

    try:
        total += proc.memory_info().rss
    except psutil.NoSuchProcess:
        return 0

    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue

    return total


""" #> MONITOR =======================
================================== """

def monitor(proc, interval=0.01):

    peak = 0

    try:
        p = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return 0

    while proc.poll() is None:

        rss = get_tree_rss(p)
        peak = max(peak, rss)

        time.sleep(interval)

    return peak


""" #> RUN EXPERIMENT ================
================================== """

def run_experiment(script, workers, interval):

    cmd = [sys.executable, script, str(workers)]

    print(f"> workers={workers} | running: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd)

    peak_holder = {"peak": 0}

    def track():
        peak_holder["peak"] = monitor(proc, interval)

    t = threading.Thread(target=track)
    t.start()

    proc.wait()
    t.join()

    return peak_holder["peak"]


""" #> MAIN TUNER ====================
================================== """

def tune(script, max_workers, interval, safety=1.25):

    results = []

    print("\n==============================")
    print("> SLURM MEMORY + CPU TUNER")
    print("==============================\n")

    for w in range(1, max_workers + 1):

        peak = run_experiment(script, w, interval)
        peak_gib = peak / GIB

        results.append((w, peak_gib))

        print(f"> workers={w:3d} | peak={peak_gib:.3f} GiB")

    #> find best valid config (monotonic safe estimate)
    best_w = 1
    best_mem = results[0][1]

    for w, mem in results:

        # heuristic: stop when memory explodes nonlinearly
        if mem > best_mem * 2.5:
            break

        best_w = w
        best_mem = mem

    recommended_mem = best_mem * safety

    print("\n==============================")
    print("> FINAL RECOMMENDATION")
    print("==============================")
    print(f"optimal cpus-per-task: {best_w}")
    print(f"peak memory: {best_mem:.3f} GiB")
    print(f"recommended --mem: {recommended_mem:.3f} GiB")
    print("==============================\n")

    return results, best_w, recommended_mem


""" #> MAIN ==========================
================================== """

if __name__ == "__main__":

    check_linux()

    parser = argparse.ArgumentParser()

    parser.add_argument("script")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.01)

    args = parser.parse_args()

    tune(
        script=args.script,
        max_workers=args.max_workers,
        interval=args.interval
    )

# thank