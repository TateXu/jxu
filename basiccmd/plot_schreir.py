
import numpy as np
import pandas as pd
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))
data = {'S1': np.array([0.538, 0.856, 0.834, 0.841, 0.813, 0.830]),
        'S2': np.array([0.831, 0.927, 0.900, 0.932, 0.902, 0.942]),
        'S3': np.array([0.998, 1.000, 1.000, 1.000, 1.000, 1.000]),
        'S4': np.array([0.993, 0.992, 0.984, 0.991, 0.983, 0.996]),
        'S5': np.array([0.636, 0.855, 0.811, 0.854, 0.803, 0.819]),
        'S6': np.array([0.640, 0.830, 0.764, 0.834, 0.768, 0.813]),
        'S7': np.array([0.478, 0.787, 0.808, 0.767, 0.786, 0.740]),
        'S8': np.array([0.792, 0.908, 0.865, 0.916, 0.865, 0.930]),
        'S9': np.array([0.621, 0.757, 0.739, 0.743, 0.730, 0.725]),
        'S10': np.array([0.978, 0.983, 0.934, 0.974, 0.882, 0.973]),
        'S11': np.array([0.535, 0.536, 0.504, 0.514, 0.532, 0.528]),
        'S12': np.array([0.657, 0.999, 0.996, 0.999, 0.991, 0.996]),
        'S13': np.array([0.738, 0.917, 0.890, 0.893, 0.884, 0.835]),
        'S14': np.array([0.994, 0.986, 0.967, 0.979, 0.973, 0.986]),
        }

clean = pd.DataFrame(data)
new = clean.melt()
new['Pipelines'] = ['CSP (4)', 'TSSF_Cov_2_step (4)', 'TSSF_Cov_1_step (4)', 'TSSF_Var_2_step (4)', 'TSSF_Var_1_step (4)', 'TS'] * 14
new.columns = ['Subject', 'Accuracy', 'Pipelines']
ax = sns.barplot(x="Subject", y="Accuracy", hue="Pipelines", data=new)
ax.figure.savefig('ds05_128chns.pdf')
plt.show()
pdb.set_trace() 

