import pickle
import numpy as np
from time import gmtime, strftime
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
pre_load_data = False

le = LabelEncoder().fit(["left_hand", "right_hand"])
if pre_load_data:

    le = LabelEncoder().fit(["left_hand", "right_hand"])
    for id_comp, nr_comp in enumerate([4, 6, 10, 20, 40, 128]):
        all_logvar = np.empty((10, 300, nr_comp, 12))
        all_logcov = np.empty((10, 300, int(nr_comp * (nr_comp + 1) / 2), 12))

        all_label = np.empty((10, 300))
        all_filters = np.empty((10, 12, 128, 128))

        for nr_subj in range(10):
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

                print('{3} Comp {0}, Subj {1}, Band {2}'.format(
                    str(nr_comp), str(nr_subj), str(ind_band),
                    strftime("%Y-%m-%d %H:%M:%S", gmtime())))
            all_label[nr_subj] = y

        label = np.asarray(all_label)
        import pdb;pdb.set_trace()
        with open(file_root + 'TSSF_MunichMI_{0}.pkl'.format(str(nr_comp)), 'wb') as f:
            pickle.dump([all_logcov, all_logvar, all_filters, label], f)
else:
    nr_comp = 4
    with open(file_root + 'TSSF_MunichMI_{0}.pkl'.format(str(nr_comp)), 'rb') as f:
        all_logcov, all_logvar, all_filters, _ = pickle.load(f)

    with open(file_root + 'label.pkl', 'rb') as f:
        label = pickle.load(f)[0]

    all_logcov_reshaped = all_logcov.copy().reshape(all_logcov.shape[0], all_logcov.shape[1], -1)
    all_logvar_reshaped = all_logvar.copy().reshape(all_logvar.shape[0], all_logvar.shape[1], -1)
    label = np.int8(label)

f_logvar = False

acc_mat = np.zeros((10, 11))
if f_logvar:
    source_data_reshaped = dc(all_logvar_reshaped)
else:
    source_data_reshaped = dc(all_logcov_reshaped)

for nr_subj in range(10):

    single_data, single_label = load_data(subject=nr_subj+1, session=0)
    single_label = le.transform(single_label)
    ind_train_list = []
    ind_test_list = []
    y_train_list = []
    y_test_list = []

    source_train_list = []
    source_test_list = []
    for ind, nr_train_trial in enumerate(range(10, 120, 10)):  # range(120, 120, 10)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(150-nr_train_trial)/150, random_state=0)
        for ind_train, ind_test in sss.split(single_data, single_label):
            ind_train_list.append(ind_train)
            ind_test_list.append(ind_test)
            y_train_list.append(single_label[ind_train])
            y_test_list.append(single_label[ind_test])
            break

        if f_logvar:
            source_train_list.append(np.empty((nr_train_trial * 2, nr_comp, 12)))
            source_test_list.append(np.empty((300 - nr_train_trial * 2, nr_comp, 12)))
        else:
            source_train_list.append(np.empty((nr_train_trial * 2, int(nr_comp * (nr_comp + 1) / 2), 12)))
            source_test_list.append(np.empty((300 - nr_train_trial * 2, int(nr_comp * (nr_comp + 1) / 2), 12)))

    import pdb;pdb.set_trace()
    for ind_band, band in enumerate(range(7, 31, 2)):
        single_data, single_label = load_data(subject=nr_subj+1, session=0, fmin=band, fmax=band+2)
        single_label = le.transform(single_label)

        for ind_trial, nr_train_trial in enumerate(range(10, 120, 10)):  # range(120, 120, 10)
            X_train = single_data[ind_train_list[ind_trial]]
            X_test = single_data[ind_test_list[ind_trial]]
            y_train = single_label[ind_train_list[ind_trial]]
            y_test = single_label[ind_test_list[ind_trial]]

            sf = TSSF(clf_str='SVM', func='clf', n_components=nr_comp, decomp='GED', logvar=f_logvar, cov_reg='oas')
            fitted_tssf = sf.fit(X_train, y_train)
            source_train_list[ind_trial][:, :, ind_band] = fitted_tssf.transform(X_train)
            source_test_list[ind_trial][:, :, ind_band] = fitted_tssf.transform(X_test)
            print('Band {0}, Trial {1}'.format(str(ind_band), str(ind_trial)))
    import pdb;pdb.set_trace()

    X_pool = np.delete(dc(source_data_reshaped), nr_subj, axis=0)
    y_pool = np.delete(dc(label), nr_subj, axis=0)

    trained = mtl(max_prior_iter=1000, prior_conv_tol=0.0001, C=1.0, C_style='ML', estimator='EmpiricalCovariance')
    trained.fit_multi_task(X_pool, y_pool, verbose=True, n_jobs=1)

    for ind, nr_train_trial in enumerate(range(10, 120, 10)):  # range(120, 120, 10)
        individual = trained.clone()
        source_train = source_train_list[ind]
        source_test = source_test_list[ind]

        source_train = source_train.reshape(source_train.shape[0], -1)
        source_test = source_test.reshape(source_test.shape[0], -1)

        y_train = y_train_list[ind]
        y_test = y_test_list[ind]
        individual.fit(source_train, y_train)

        y_predict = np.sign(individual.predict(source_test))
        y_predict = np.int8((y_predict + 1) / 2)
        acc = 1 - np.sum((y_predict - y_test) ** 2) / y_predict.shape[0]
        acc_mat[nr_subj, ind] = acc
        print('S_{0}, #Trial_{1}, Accuracy: {2}'.format(str(nr_subj+1), str(nr_train_trial), str(acc)))
import pdb
pdb.set_trace()




