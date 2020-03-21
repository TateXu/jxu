from mne.io import read_raw_brainvision as rrb

from mne.viz import plot_raw

from mne.viz import plot_raw_psd

channel_list = ['FCC4h', 'C1', 'C2', 'C3', 'C3', 'C4', 'Cz', 'CCP1h', 'CCP2h', 'CCP3h', 'CCP4h', 'CCP5h', 'CCP6h', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CPz', 'CPP1h', 'CPP2h', 'CPP3h', 'CPP4h', 'CPP5h', 'P1', 'P2', 'P3', 'P4', 'P5', 'Pz', 'PPO1h', 'PPO2h', 'PO3', 'POz']
# = vhdr_load('pure_tACS_1mA_10Hz/pure_tACS_1mA_10Hz.vhdr').pick_channels(channel_list)
def vhdr_load(filename):
    
    data = rrb(filename, eog=('HEOGL', 'HEOGR', 'VEOGb'), misc='auto', scale=1.0, preload=True, verbose=None)
 
    return data


def single_to_multi(filename, channel_list):

data = {}
for chn in channel_list:
     data[chn] = vhdr_load(filename).pick_channels([chn])

