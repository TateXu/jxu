"""
===========================================================================
Example of the usage for tangent space spatial filter (TSSF) standalone

Author: Jiachen XU <jiachen.xu.94@gmail.com>,
        Vinay Jayaram <vinayjayaram13@gmail.com>

===========================================================================

This example can be successfully run on following dependencies:
Sklearn version == 0.20.3
MNE version == 0.18.2
Pyriemann version == 0.2.5
MOABB version == 0.2.1
"""

import os
import numpy as np
import warnings
import yaml
import pickle

from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pkg_file.pyriemann.estimation import Covariances
from pkg_file.pyriemann.tangentspace import TangentSpace
from pkg_file.features import Feat_convertor
from pkg_file.tssf_features import TSSF


def score(clf, X, y, scoring):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return acc.mean()

def load_data(subject, session):

    import moabb
    from moabb.datasets import MunichMI, BNCI2014001
    from moabb.paradigms import MotorImagery

    MI = MotorImagery(fmin=8, fmax=32, events=['left_hand', 'right_hand'])
    dataset = MunichMI()  #  BNCI2014001() #
    X, y, metadata = MI.get_data(dataset, [subject])
    return X, y


if __name__ == "__main__":
    ###############################################################################
    # Parameter Initialization
    # ------------------------


    # L2 regularized SVM classifier using grid search to find reg. coefficients
    parameters = {'C': np.logspace(-2, 2, 10)}
    opt_SVM = GridSearchCV(SVC(kernel='linear'), parameters)

    if os.path.exists('cov_analytic.pkl'):
        os.remove('cov_analytic.pkl')
    all_decomp = ['GED', 'ED']
    all_logvar = [False, True]
    text_logvar = ['Cov', 'Var']

    nr_comp = 4  # Number of applied (visualized) filter components
    nr_decomp = 0  # Use GED or ED to generate filter

    ###############################################################################
    # Initializing algorithms pipeline for analysis
    # ---------------------------------------------

    pipelines = {}
    # Standard CSP based classifier
    pipelines['CSP (' + str(nr_comp) + ') + SVM'] = make_pipeline(
        CSP(n_components=nr_comp),
        opt_SVM)

    # Standard Riemannian methods
    pipelines['TS + w/o SF + SVM + AIRM'] = make_pipeline(
        Covariances('scm'),
        TangentSpace(metric='riemann'),
        opt_SVM)

    # Tangent space spatial filter
    for nr_logvar in range(2):
        ts_name = 'TSSF_' + text_logvar[nr_logvar] + ' (' + str(nr_comp) + ') + '
        # Two-steps classifcation of TSSF
        pipelines[ts_name + 'Two-steps'] = make_pipeline(
            TSSF(clf_str='SVM', func='clf', n_components=nr_comp,
                 decomp=all_decomp[nr_decomp], logvar=all_logvar[nr_logvar]),
            opt_SVM)
        # One-step classifcation of TSSF
        pipelines[ts_name + 'One-step'] = make_pipeline(
            TSSF(clf_str='SVM', func='clf', n_components=nr_comp,
                 decomp=all_decomp[nr_decomp], logvar=all_logvar[nr_logvar]))

    # Classification based on Hilbert transform derived features

    part = 'real'  # real, imag, anal
    with open('cov_analytic.pkl', 'wb') as in_part:
        pickle.dump(part, in_part)

    pipelines['Cov_real + AIRM'] = make_pipeline(Feat_convertor(['Analytic'], fs=500),
                                        Covariances(estimator='oas'),
                                        TangentSpace(),
                                        LDA())
    pipelines['IM + CSP'] = make_pipeline(Feat_convertor(['IM'], fs=500),
                                        CSP(),
                                        LDA())

    pipelines['RE + AIRM'] = make_pipeline(Feat_convertor(['RE'], fs=500),
                                        Covariances(estimator='oas'),
                                        TangentSpace(),
                                        LDA())


    # You can replace load_data() function with your own data instead of using MOABB dependencies
    # X: float& ndarray, shape (n_trials, n_channels, n_samples)
    # y: float& ndarray, shape (n_trials,)
    # X, y = load_data(subject=1, session=0)
    with open("temp_data.pkl", "rb") as f_out:
        X_sub, y = pickle.load(f_out)
    for name, clf in pipelines.items():
        acc = score(clf, X_sub, y, scoring='roc_auc')
        print(name + ': '+ str(acc))


