import os
import numpy as np
from time import time
from os import system
from joblib import Parallel, delayed


def display_time(time):

    hour = int(time/3600)
    minute = (int(time % 3600)) // 60
    second = time - (3600. * hour + 60. * minute)
    print("Done in %d hours %d minutes %09.6f seconds"
          % (hour, minute, second))


def run_command(args):
    system("python3 populations.py {0} {1}".format(*args))


def batch_run(n_jobs):

    args = []
    for i in range(len(g_StoG)):
        for j in range(len(g_GtoG)):
            args.append([g_StoG[i],
                         g_GtoG[j]])
    Parallel(n_jobs=n_jobs)(map(delayed(run_command), args))


n_jobs = 4
g_StoG = np.linspace(0.0, 0.1, 6)
g_GtoG = np.linspace(0.0, 0.1, 6)


start = time()
batch_run(n_jobs)
display_time(time() - start)