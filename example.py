import annealing
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import importlib
import time
import random
import math

# %%

n_bins = 30
n_dim = 3
n_sig = 10 * n_bins**n_dim
n_bkg = 100 * n_bins**n_dim
print('number of signal events:', n_sig)
print('number of background events:', n_bkg)
print('number of bins:', n_bins**n_dim)

# %%


def project_to_dim(hist, dim=2):
    h = hist.copy()
    bins = h.shape
    if len(bins) == 1:
        return np.tile(h, (bins[0], 1))
    for i in range(len(bins) - dim):
        h = h[int(bins[i] / 2)]
    return h


def show_state(state, n_bins):
    if len(state) == 1:
        state = np.concatenate((state, [[0, n_bins - 1]]))
    z = np.zeros(shape=(n_bins, n_bins))
    z[state[1, 0]:state[1, 1] + 1, state[0, 0]:state[0, 1] + 1] = 1
    plt.pcolormesh(z)
    plt.show()

# %%


coord = scipy.stats.multivariate_normal(np.zeros(n_dim), np.identity(n_dim)).rvs(n_sig)
h_signal, edges = np.histogramdd(coord, bins=n_bins)
plt.pcolormesh(project_to_dim(h_signal))
plt.colorbar()
plt.show()

# %%

h_background = scipy.stats.poisson(50).rvs(n_bins**n_dim).reshape([n_bins for i in range(n_dim)])
plt.pcolormesh(project_to_dim(h_background))
plt.colorbar()
plt.show()

# %%

init_state = np.array([[0, n_bins - 1] for i in range(n_dim)], dtype='int')

for i in range(10 - h_signal.ndim):
    h_signal = np.expand_dims(h_signal, axis=0)
    h_background = np.expand_dims(h_background, axis=0)

for i in range(10 - len(init_state)):
    init_state = np.concatenate(([[0, 0]], init_state))

# %%

importlib.reload(annealing)

# %%

best_state, best_energy = annealing.selanneal(init_state, h_signal, h_background, n_bins, n_dim)
