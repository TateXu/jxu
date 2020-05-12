from jxu.data.loader import *
import matplotlib.pyplot as plt

file_root = '/home/jxu/File/Data/pure_tACS/'


for i_freq, freq in enumerate([4, 10, 40]):
    for i_sg, sg in enumerate(['wo_SG', 'w_SG']):
        filename = 'pure_tACS_1mA_' + str(freq) + 'Hz_' + sg
        data = vhdr_load(file_root + filename + '/' + filename + '.vhdr')
        fmax = freq * 5 if freq * 5 > 90 else 90

        data.plot_psd(picks=['tACS'], fmax=fmax, show=False, ax=plt.subplot(3, 2, 2*i_freq + i_sg + 1))

import pdb 
pdb.set_trace()
