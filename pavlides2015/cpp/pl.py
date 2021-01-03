import numpy as np
import pylab as pl

y = np.loadtxt("data/euler.txt")

fig, ax = pl.subplots(1, figsize=(5,4))
for i in range(1, 4):
    pl.plot(y[:,0], y[:, i])


pl.tight_layout()
pl.savefig("data/fig.png", dpi=150)
