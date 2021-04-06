import numpy as np

sim_params = {
    'num': 4,
    'dt': 0.005,
    't_simulation': 1500.0,
    'I_C': 172.18 / 1000.0,
    'I_Str': 8.46 / 1000.0,
}


# wSG, wGS, wCS, wSC, wGG, wCC
# wSG = 4.87
# wGS = 1.33
# wCS = 9.98
# wSC = 8.93
# wGG = 0.53
# wCC = 6.17
true_params = np.array([4.87, 1.33, 9.98, 8.93, 0.53, 6.17])
num_workers = 8
num_simulations = 100
method = 'SNPE'
num_rounds = 2

prior_min = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prior_max = [10.0, 10.0, 20.0, 20.0, 10.0, 10.0]
num_samples = 10000
num_threads = 1
