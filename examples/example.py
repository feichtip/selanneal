import selanneal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import importlib
import numba as nb
import time
import random
import math
import argparse
# import boost_histogram as bh

# %%

CLI = argparse.ArgumentParser()
CLI.add_argument(
    "-m", "--mode",
    nargs="?",
    type=str,
    default='edges',
)
args = CLI.parse_args()
print('mode: ', args.mode)

# %%

np.random.seed(42)

# %%

bins = (100, 100)
n_bins = 1
for bin in bins:
    n_bins *= bin

n_dim = len(bins)
n_sig = 10 * n_bins
n_bkg = 100 * n_bins
print('number of signal events:', n_sig)
print('number of background events:', n_bkg)
print('number of bins:', n_bins)

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


# generate data points
coord = scipy.stats.multivariate_normal(np.zeros(n_dim), np.identity(n_dim)).rvs(n_sig)
h_signal, edges = np.histogramdd(coord, bins=bins)
# plt.pcolormesh(project_to_dim(h_signal))
# plt.colorbar()
# plt.show()

# %%

# # alternative with boost_histogram
# mins = [-5] * n_dim
# maxs = [5] * n_dim
# hist = bh.Histogram(*[bh.axis.Regular(bin, min, max) for bin, min, max in zip(bins, mins, maxs)])
#
# sample_size = 10_000_000
# if n_sig < sample_size:
#     sample_size = n_sig
# for i in range(n_sig // sample_size):
#     print(i)
#     coord = scipy.stats.multivariate_normal(np.zeros(n_dim), np.identity(n_dim)).rvs(sample_size)
#     hist.fill(*coord.T)
# h_signal = hist.view()
#
# # Make the plot
# fig, ax = plt.subplots()
# mesh = ax.pcolormesh(hist.axes.edges[0].flatten(), hist.axes.edges[1].flatten(), h_signal.sum(axis=tuple(j for j in range(n_dim) if j != 0 and j != 1)))
# fig.colorbar(mesh)
# plt.show()

# %%

h_background = scipy.stats.poisson(50).rvs(n_bins).reshape(bins)
# plt.pcolormesh(project_to_dim(h_background))
# plt.colorbar()
# plt.show()

#%%

h_sys_up = scipy.stats.poisson(0.1).rvs(n_bins).reshape(bins)
h_sys_down = scipy.stats.poisson(0.1).rvs(n_bins).reshape(bins)

# %%

importlib.reload(selanneal)

# %%

start = time.time()
best_state, best_energy = selanneal.run(h_signal, h_background, h_sys_up, h_sys_down, mode=args.mode, coupling=0.005)
print(time.time() - start)

if args.mode == 'bins':
    plt.pcolormesh(best_state)
    plt.show()
else:
    print(best_state)
