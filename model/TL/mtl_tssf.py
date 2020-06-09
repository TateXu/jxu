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
from sklearn.model_selection import train_test_split
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
pre_load_data = True

le = LabelEncoder().fit(["left_hand", "right_hand"])
if pre_load_data:

    le = LabelEncoder().fit(["left_hand", "right_hand"])
    for id_comp, nr_comp in enumerate([4, 6, 10, 20, 40, 128]):
        all_logvar = np.empty((10, 300, nr_comp, 12))
        all_logcov = np.empty((10, 300, int(nr_comp * (nr_comp + 1) / 2), 12))

        all_label = np.empty((10, 300))
        all_filters = np.empty((10, 12, 128, 128))

        for nr_subj in range(10):
            feat_mat = np.empty((300, 12, nr_comp))
            for ind_band, band in enumerate(range(7, 31, 2)):
                X, y = load_data(subject=nr_subj+1, session=0, fmin=band, fmax=band+2)
                y = le.transform(y)

                sf = TSSF(clf_str='SVM', func='clf', n_components=nr_comp,
                          decomp='GED', logvar=False, cov_reg='oas')
                fitted_tssf = sf.fit(X, y)
                all_logcov[nr_subj, :, :, ind_band] = fitted_tssf.transform(X)
                fitted_tssf.logvar = True
                all_logvar[nr_subj, :, :, ind_band] = fitted_tssf.transform(X)
                all_filters[nr_subj, ind_band] = fitted_tssf.all_filters  # C*K
                import pdb;pdb.set_trace()
                print('Comp {0}, Subj {1}, Band {2}'.format(
                    str(nr_comp), str(nr_subj), str(ind_band)) )
        all_label[nr_subj] = y

        label = np.asarray(all_label)
        with open(file_root + 'TSSF_MunichMI_{0}.pkl'.format(str(nr_comp)), 'wb') as f:
            pickle.dump([all_logcov, all_logvar, all_filters, label], f)
else:
    with open(file_root + 'Logvar_MunichMI_{0}.pkl'.format(str(nr_comp)), 'rb') as f:
        source_data, label = pickle.load(f)
    source_data = source_data.transpose((0, 1, 3, 2))
    source_data_reshaped = source_data.copy().reshape(source_data.shape[0], source_data.shape[1], -1)
    label = np.int8(label)


TSSF_flag = False

acc_mat = np.zeros((10, 11))
for nr_subj in range(10):

    # trained = mtl_fd(max_prior_iter=1000, prior_conv_tol=0.0001, C=1, C_style='ML')

    X_pool = np.delete(dc(source_data_reshaped), nr_subj, axis=0)
    y_pool = np.delete(dc(label), nr_subj, axis=0)
    # trained.fit_multi_task(X_pool, y_pool, verbose=True, n_jobs=1)
    X_pool_ = X_pool.reshape(-1, X_pool.shape[-1])
    y_pool_ = y_pool.reshape(-1)

    if TSSF_flag:
        single_data, single_label = load_data(subject=nr_subj+1, session=0)
        single_label = le.transform(single_label)
    else:
        single_data = source_data[nr_subj]
        single_label = label[nr_subj]

    trained = mtl(max_prior_iter=1000, prior_conv_tol=0.0001, C=1.0, C_style='ML', estimator='EmpiricalCovariance')
    trained.fit_multi_task(X_pool, y_pool, verbose=True, n_jobs=1)
    for ind, nr_train_trial in enumerate(range(10, 120, 10)):  # range(120, 120, 10)
        individual = trained.clone()
        X_train, X_test, y_train, y_test = train_test_split(
            single_data, single_label, test_size=(150-nr_train_trial)/150, random_state=0)

        if TSSF_flag:
            sf = TSSF(clf_str='SVM', func='clf', n_components=nr_comp, decomp='GED', logvar='Cov')
            fitted_tssf = sf.fit(X_train, y_train)
            source = fitted_tssf.transform(X_train)
        else:
           X_train = X_train.reshape(X_train.shape[0], -1)
           X_test = X_test.reshape(X_test.shape[0], -1)

        individual.fit(X_train, y_train)

        if TSSF_flag:
            y_predict = np.sign(individual.predict(fitted_tssf.transform(X_test)))
        else:
            y_predict = np.sign(individual.predict(X_test))
        y_predict = np.int8((y_predict + 1) / 2)
        acc = 1 - np.sum((y_predict - y_test) ** 2) / y_predict.shape[0]
        acc_mat[nr_subj, ind] = acc
        print('S_{0}, #Trial_{1}, Accuracy: {2}'.format(str(nr_subj+1), str(nr_train_trial), str(acc)))
import pdb
pdb.set_trace()




