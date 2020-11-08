# =============================================================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2020-10-21 19:26:32
# Name       : fmri.py
# Version    : V1.0
# Description: A toy example for loading/classifying fmri data
# =============================================================================

import numpy as np
import nilearn
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

file_root = '/home/jxu/File/Data/fMRI/Emotion/'

fmri_file = 'fmri/swuasub-01_RealActor_run-01_bold.nii'
marker = 'marker/pain_sub1_run1_temporal_info.mat'
roi = 'ROI/voxelFWE_main_effect_conjunction_rAI_aal_33_29_2.nii'

fmri_filename = file_root + fmri_file


from scipy.io import loadmat

behavioral = loadmat(file_root + marker)

label = np.squeeze(behavioral['pain_sub1_run1'][0, 0][0])
onsettime = np.squeeze(behavioral['pain_sub1_run1'][0, 0][1])
offtime = np.squeeze(behavioral['pain_sub1_run1'][0, 0][2])


onset_slices = np.floor(onsettime/1200).astype('int16')
off_slices = np.floor(offtime/1200).astype('int16')

ind_class_1 = np.where(label == 1)[0]
ind_class_2 = np.where(label == 2)[0]

start_1 = onset_slices[ind_class_1]
start_2 = onset_slices[ind_class_2]
end_1 = off_slices[ind_class_1]
end_2 = off_slices[ind_class_2]

end_1[(end_1-start_1) != 2] += 1
end_2[(end_2-start_2) != 2] += 1

aa = np.arange(656)

ind_class_1 = aa[((aa>start_1[:,None]) & (aa<end_1[:,None])).any(0)]
ind_class_2 = aa[((aa>start_2[:,None]) & (aa<end_2[:,None])).any(0)]


from numpy import concatenate, sort

all_ind = concatenate((ind_class_1, ind_class_2))
all_ind.sort(kind='mergesort')


import pdb;pdb.set_trace()
from nilearn import plotting
from nilearn.image import mean_img
plotting.view_img(mean_img(fmri_filename), threshold=None)


mask_filename = file_root + roi
plotting.plot_roi(mask_filename, cmap='Paired')

import pdb;pdb.set_trace()


from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
fmri_masked = masker.fit_transform(fmri_filename)
selected_fmri = fmri_masked[all_ind]
fmri_label = label[(label==1) | (label==2)]

masker.generate_report()
print(fmri_masked)
print(fmri_masked.shape)
import pdb;pdb.set_trace()
from sklearn.model_selection import KFold
from sklearn.svm import SVC
svc = SVC(kernel='rbf')

cv = KFold(n_splits=5)
for train, test in cv.split(X=selected_fmri):
    selected_label = fmri_label[train]
    svc.fit(selected_fmri[train], selected_label)
    pred = svc.predict(selected_fmri[test])

    print((pred == fmri_label[test]).sum()/float(len(fmri_label[test])))

import pdb;pdb.set_trace()

import pdb;pdb.set_trace()
