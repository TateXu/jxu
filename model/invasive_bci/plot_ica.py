#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-09-27 15:36:51
# Name       : plot_ica.py
# Version    : V1.0
# Description: .
#========================================

import pickle
import numpy as np
from matplotlib import pyplot as plt

# '2019_after_chn_reject'
with open('trained_ica_2017_1500Hz.pkl', 'rb') as f:
    ica, simulated_raw = pickle.load(f)

# Plot raw signal
simulated_raw.plot()

# Plot psd
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
simulated_raw.plot_psd(ax=ax, show=False)

# Plot source activity after ICA
ica.plot_sources(simulated_raw)

# Repair with ICAA
cp_raw = simulated_raw.copy()
ic_incl_list = [24, 27, 34, 35, 36, 37, 38, 41, 44, 49, 50, 51, 52, 54, 55, 57,
                58, 59, 67, 79, 81, 82, 83, 84, 85, 87, 89, 92, 94]
ic_excl_list = np.setdiff1d(np.arange(95), ic_incl_list)
ica.exclude = ic_excl_list
ica.apply(cp_raw)

import pdb;pdb.set_trace()
