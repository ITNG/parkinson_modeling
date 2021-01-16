import numpy as np
import brian2 as b2
import pylab as plt
from time import time

par = {
    'vsyntc': -85,
    'gsyntc': .08,
    'asg': 200,
    'bsg': .4,
    'itc': 6,
    'shi': -80,
    'shi2': -90,
    'dur': 5,
    'dur2': 10,
    'period': 25,
    'gnabar': 3,
    'gkbar': 5,
    'glbar': 0.05,
    'gtbar': 5,
    'ena': 50,
    'ek': -90,
    'eleak': -70,
    'qht': 2.5,
    'tadj': 1,
    'apr': 4,
    'apt': 1,
}

