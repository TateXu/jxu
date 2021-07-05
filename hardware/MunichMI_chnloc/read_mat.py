from scipy import io 
import mne
import numpy as np
from mne import create_info


locmat = io.loadmat('Munich128ChannelsLocs.mat')

field_name = list( locmat['Chanlocs'][0].dtype.names)
x_ind = field_name.index('X')
y_ind = field_name.index('Y')
z_ind = field_name.index('Z')

chn_loc_dict = {}
for chn in range(128):
    chn_loc_dict[str(chn)] = np.asarray([locmat['Chanlocs'][0][chn][x_ind][0,0],
                                         locmat['Chanlocs'][0][chn][y_ind][0,0],
                                         locmat['Chanlocs'][0][chn][z_ind][0,0]])

montage = mne.channels.make_dig_montage(ch_pos=chn_loc_dict)

ch_names = list(chn_loc_dict.keys())
ch_types = 'eeg'
sfreq = 500.0

info = create_info(
    ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
info.set_montage(montage)
import pdb;pdb.set_trace()
