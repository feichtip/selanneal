import importlib
import math
import os
import random
import time

import numpy as np
import pandas as pd
import selanneal
from selanneal import annealing

# can disable numba, but is much slower
# os.environ['DISABLE_NUMBA'] = '1'

# %%

np.random.seed(42)

# %%

# generate toy data points

features = ['A', 'B', 'C', 'D', 'E']
# features = ['A', 'B']

n_dim = len(features)
n_sig = 10_000
n_bkg = 1_000_000

# generated signal events
sig_gen = np.random.multivariate_normal(np.zeros(n_dim), np.identity(n_dim), size=n_sig)
Ngen = len(sig_gen)

# select random subset for reconstructed events
rand_10p = np.random.rand(n_sig) > 0.9
rand_50p = np.random.rand(n_sig) > 0.5
sig_rec = sig_gen[((sig_gen[:, 0] >= 0) & rand_10p) | ((sig_gen[:, 0] < 0) & rand_50p)]

# uniform background between -5 and 5
bkg_rec = np.random.uniform(- np.ones(n_dim) * 5, np.ones(n_dim) * 5, size=[n_bkg, n_dim])

# %%

# create dataframe from generated data
sig_df = pd.DataFrame(sig_rec, columns=features)
sig_df['isSignal'] = 1
sig_df['weight'] = 1.0

bkg_df = pd.DataFrame(bkg_rec, columns=features)
bkg_df['isSignal'] = 0
bkg_df['weight'] = 2.0  # scale to data lumi

df = pd.concat([sig_df, bkg_df], ignore_index=True)

# %%

# importlib.reload(selanneal)
# importlib.reload(annealing)

# %%

result = selanneal.optimise.iterate(df[features].values, df['isSignal'].values, features=features, weight=df['weight'].values,
                                    Nexp=Ngen, new_bins=5, Tmin=1E-7, Tmax=1E-2, steps=5_000, quantile=0, punzi_a=3, fom='punzi', verbosity=0)
print(f'Punzi FOM = {-result[-1]}')
result[0]

# %%

# cross-check
query = '(' + ') and ('.join(result[0]) + ')'
print(query)
sig_sel = sig_df.query(query)
bkg_sel = bkg_df.query(query)
print((sig_sel.shape[0] / Ngen) / (3 / 2 + np.sqrt(bkg_sel.shape[0] * 2.0)))

# %%
