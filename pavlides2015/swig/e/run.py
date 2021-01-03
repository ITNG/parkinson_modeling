import os
import lib
import numpy as np 
import pylab as plt
from time import time

if not os.path.exists('data'):
    os.makedirs('data')

t_simulation = 1000.0
dt = 0.005
nstart = 50
num = 4

labels = ["STN", "GPE", "E", "I"]

if __name__ == "__main__":
    
    start_time = time()

    sol = lib.DDE()
    sol.set_params(t_simulation, dt, nstart)
    sol.set_history([0] * num)
    sol.euler_integrator()
    times = sol.t_ar
    y = sol.y
    # print(type(times), type(y))
    # print(len(times), len(y), len(y[0]))

    print("Done in {:10.4f} seconds".format(time() - start_time))
    
    fig, ax = plt.subplots(1, figsize=(10, 4))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Firing Rate (spk/ms")

    for i in range(num):
        ax.plot(times, y[i], label=labels[i])
    ax.margins(x=0)
    ax.legend(frameon=False, loc='upper right')
    fig.savefig("data/fig.png")

