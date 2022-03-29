#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2022-03-28 22:41:41
# Name       : plot_re.py
# Version    : V1.0
# Description: .
#========================================

import seaborn as sns
import numpy as np
import pickle
from matplotlib import pyplot as plt
test_percent_1 = {'Model1_pval0.2_bs10.pkl': ['Pct=0.2', 'k'],
                  'Model1_pval0.3_bs10.pkl': ['Pct=0.3', 'r'],
                  'Model1_pval0.4_bs10.pkl': ['Pct=0.4', 'b'],
                  'Model1_pval0.5_bs10.pkl': ['Pct=0.5', 'g'],
                  }

test_percent_2 = {'Model2_pval0.2_bs10.pkl': ['Pct=0.2', 'k'],
                  'Model2_pval0.3_bs10.pkl': ['Pct=0.3', 'r'],
                  'Model2_pval0.4_bs10.pkl': ['Pct=0.4', 'b'],
                  'Model2_pval0.5_bs10.pkl': ['Pct=0.5', 'g'],
                  }
bs_1 = {'Model1_pval0.2_bs5.pkl': ['Batch_size=5', 'r'],
        'Model1_pval0.2_bs10.pkl': ['Batch_size=10', 'k'],
        'Model1_pval0.2_bs20.pkl': ['Batch_size=20', 'b'],
        'Model1_pval0.2_bs40.pkl': ['Batch_size=40', 'g'],
        'Model1_pval0.2_bs80.pkl': ['Batch_size=80', 'purple'],
        'Model1_pval0.2_bs160.pkl': ['Batch_size=160', 'cyan'],
        'Model1_pval0.2_bs320.pkl': ['Batch_size=320', 'orange'],
        }

bs_2 = {'Model2_pval0.2_bs5.pkl': ['Batch_size=5', 'r'],
        'Model2_pval0.2_bs10.pkl': ['Batch_size=10', 'k'],
        'Model2_pval0.2_bs20.pkl': ['Batch_size=20', 'b'],
        'Model2_pval0.2_bs40.pkl': ['Batch_size=40', 'g'],
        'Model2_pval0.2_bs80.pkl': ['Batch_size=80', 'purple'],
        'Model2_pval0.2_bs160.pkl': ['Batch_size=160', 'cyan'],
        'Model2_pval0.2_bs320.pkl': ['Batch_size=320', 'orange'],
        }

structure = {'Model4_pval0.2_bs10.pkl': ['128-128-128-128', 'r'],
             'Model3_pval0.2_bs10.pkl': ['128-128-128-10', 'b'],
             'Model2_pval0.2_bs10.pkl': ['128-128-10-10', 'k'],
             'Model1_pval0.2_bs10.pkl': ['128-10-10-10', 'g'],
             }
depth_128_128 = {'five_layers.pkl': ['128-128-10-10-10-10', 'r'],
                 'four_layers.pkl': ['128-128-10-10-10', 'b'],
                 'three_layers.pkl': ['128-128-10-10', 'k'],
                 'two_layers.pkl': ['128-128-10', 'purple'],
                 'one_layers.pkl': ['128-128', 'g'],
                 }

depth_128_10 = {'Model5_pval0.2_bs10.pkl': ['128-10-10-10-10-10', 'r'],
                'Model6_pval0.2_bs10.pkl': ['128-10-10-10-10', 'b'],
                'Model2_pval0.2_bs10.pkl': ['128-10-10-10', 'k'],
                'Model7_pval0.2_bs10.pkl': ['128-10-10', 'purple'],
                'Model8_pval0.2_bs10.pkl': ['128-10', 'g'],
                }
title_list = ['Accuracy-Train', 'Loss-Train', 'Accuracy-Test', 'Total Time']

filename_list = ['Test_percentage_128-128-10-10',
                 'Test_percentage_128-10-10-10',
                 'Batch_size_128-128-10-10',
                 'Batch_size_128-10-10-10',
                 'Structure', 'Depth_128_128', 'Depth_128_10']
data_dict = [test_percent_1, test_percent_2, bs_1, bs_2,
             structure, depth_128_128, depth_128_10]

for dict_pkl, filename in zip(data_dict, filename_list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for key, val in dict_pkl.items():
        with open(f'./files/{key}', 'rb') as f:
            data = pickle.load(f)
        acc = np.asarray(data[0])
        acc_train = acc[:, 0]
        loss_train = acc[:, 1]
        acc_test = acc[:, 2]
        time = np.cumsum(np.asarray(data[1]))

        for ind, (ax, var) in enumerate(zip(np.asarray(axes).reshape(1, -1)[0],
                                        [acc_train, loss_train, acc_test, time])):
            ax.plot(var, val[1], label=val[0])
            ax.set_title(title_list[ind])
            ax.set_xlabel('Epoch')
            if ind != 3:
                ax.set_ylim([-0.01, 1.01])

    handles, labels = ax.get_legend_handles_labels()

    ncol = dict_pkl.__len__()
    if ncol > 5:
        ncol = round(ncol/2)
    fig.legend(handles, labels, loc='upper center', ncol=ncol)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.grid('on')
    fig.savefig(f'./results/{filename}.jpg')
import pdb;pdb.set_trace()

