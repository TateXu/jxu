import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder


import moabb
import numpy as np
import warnings

from mne.decoding import CSP
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import MunichMI, BNCI2014001
from moabb.paradigms import MotorImagery
from moabb.pipelines.features import TSSF
from copy import deepcopy as dc
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import warnings

from pymtl.linear_regression import MTLRegressionClassifier as mtl
from pymtl.feature_decomposition import FeatureDecompositionModel as mtl_fd

from jxu.sp.laplacian import surf_LP
from jxu.viz.channel_location.load import electrodes

warnings.filterwarnings("ignore")


def score(clf, X, y, scoring):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    acc = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    return acc.mean()

def load_data(subject, session, fmin=8, fmax=32):
    MI = MotorImagery(fmin=fmin, fmax=fmax, events=['left_hand', 'right_hand'])
    dataset = MunichMI()  #  BNCI2014001()
    X, y, metadata = MI.get_data(dataset, [subject])
    for nr_ses, name_ses in enumerate(np.unique(metadata.session).tolist()):
        if nr_ses == session:
            ix = metadata.session == name_ses
            return X[ix], y[ix]

file_root = '/home/jxu/File/Code/Git/pyMTL/examples/'
loc = electrodes().MunichMI()
nr_comp = 6
pre_load_data = False

if pre_load_data:
    all_data = np.empty((10, 300, 12, 128))
    all_label = np.empty((10, 300))

    le = LabelEncoder().fit(["left_hand", "right_hand"])
    for nr_subj in range(10):
        feat_mat = np.empty((300, 12, 128))
        for ind_band, band in enumerate(range(7, 31, 2)):
            X, y = load_data(subject=nr_subj+1, session=0, fmin=band, fmax=band+2)
            for nr_trial in range(X.shape[0]):
                feat_mat[nr_trial, ind_band] = np.log(np.var(surf_LP(X[nr_trial], loc, filter_type='small'), axis=1))
            print(str(nr_subj) + ' ' + str(ind_band) + ' ' + str(nr_trial))
        all_data[nr_subj] = feat_mat
        all_label[nr_subj] = le.transform(y)

    import pdb;pdb.set_trace()
    with open(file_root + 'Logvar_MunichMI_{0}.pkl'.format(str(nr_comp)), 'wb') as f:
        pickle.dump([all_data, all_label], f)
else:
    with open(file_root + 'Logvar_MunichMI_{0}.pkl'.format(str(nr_comp)), 'rb') as f:
        source_data, label = pickle.load(f)
    source_data = source_data.transpose((0, 1, 3, 2))
    source_data_reshaped = source_data.copy().reshape(source_data.shape[0], source_data.shape[1], -1)
    label = np.int8(label)

le = LabelEncoder().fit(["left_hand", "right_hand"])
n_repeat = 5
acc_mat = np.zeros((10, 11, n_repeat))
for nr_subj in range(10):

    X_pool = np.delete(dc(source_data_reshaped), nr_subj, axis=0)
    y_pool = np.delete(dc(label), nr_subj, axis=0)

    single_data = source_data_reshaped[nr_subj]
    single_label = label[nr_subj]

    trained = mtl(max_prior_iter=1000, prior_conv_tol=0.0001, C=1.0, C_style='ML', estimator='EmpiricalCovariance')
    trained.fit_multi_task(X_pool, y_pool, verbose=True, n_jobs=1)
    for ind_trial, nr_train_trial in enumerate(range(10, 120, 10)):

        sss = StratifiedShuffleSplit(n_splits=n_repeat, test_size=(150-nr_train_trial)/150, random_state=0)
        for ind_repeat, (ind_train, ind_test) in enumerate(sss.split(single_data, single_label)):
            X_train = single_data[ind_train]
            X_test = single_data[ind_test]
            y_train = single_label[ind_train]
            y_test = single_label[ind_test]

            individual = trained.clone()
            individual.fit(X_train, y_train)

            y_predict = np.int8((np.sign(individual.predict(X_test)) + 1) / 2)
            acc = 1 - np.sum((y_predict - y_test) ** 2) / y_predict.shape[0]
            acc_mat[nr_subj, ind_trial, ind_repeat] = acc
            print('S_{0}, #Trial_{1}, Accuracy: {2}'.format(str(nr_subj+1), str(nr_train_trial), str(acc)))
import pdb;pdb.set_trace()
with open('MTL_acc.pkl', 'rb') as f:
    pickle.dump(acc_mat, f)
import pdb;pdb.set_trace()
