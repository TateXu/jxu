#========================================
# Author     : Jiachen Xu
# Blog       : www.jiachenxu.net
# Time       : 2021-03-30 10:18:13
# Name       : dist_cov.py
# Version    : V1.0
# Description: .
#========================================

import os
import numpy as np
from dcor import distance_covariance as dcov
from dcor import distance_correlation as dcor
from dcor import distance_covariance_sqr as dcov_sqr
from dcor import distance_correlation_sqr as dcor_sqr
import pickle

import concurrent.futures
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import pdist, squareform

class Distcov(BaseEstimator, TransformerMixin):
    def __init__(self, save=False, subject=0, ds_name='', method='d_cov',
                 multiprocess=False):
        self.save = save
        self.subject = subject
        self.ds_name = ds_name
        self.multiprocess = multiprocess
        if self.multiprocess:
            self.method_dict = {'d_cov': dcov,
                                'd_cor': dcor,
                                'd_cov_sqr': dcov_sqr,
                                'd_cor_sqr': dcor_sqr}
        else:
            self.method_dict = {'d_cov': self.d_cov,
                                'd_cor': self.d_cor,
                                'd_cov_sqr': self.d_cov_sqr,
                                'd_cor_sqr': self.d_cor_sqr}

        self.method = method
        self.method_func = self.method_dict[self.method]

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X, force=False):
        """transform"""
        distcov_mat = np.empty((X.shape[0], X.shape[1], X.shape[1]))
        X = np.transpose(X, (0, 2, 1))
        filename = './temp_data/Distcov_{0}_S{1}_{2}.pkl'.format(self.ds_name, str(self.subject), self.method)

        if os.path.exists(filename):
            print('Loading local data' + filename)
            with open(filename, 'rb') as f:
                distcov_mat = pickle.load(f)
        else:
            for id_trial in range(X.shape[0]):
                if self.multiprocess:
                    distcov_mat[id_trial] = self.generic_parallel(X[id_trial])
                else:
                    distcov_mat[id_trial] = self.method_func(X[id_trial])
                print('{0}/{1}'.format(str(id_trial), str(X.shape[0])))

            if self.save:
                with open(filename, 'wb') as f:
                    pickle.dump(distcov_mat, f)

        return distcov_mat

    def fit_transform(self, X, y=None, force=False):

        return self.transform(X, force=force)

    def generic_parallel(self, samples):
        import dcor._fast_dcov_avl
        assert self.multiprocess, "Must turn on multiprocess"
        num_samps, num_feats = samples.shape
        cov_mat = np.empty((num_feats, num_feats), float)
        row, col = np.tril_indices(num_feats)
        data = samples.T
        for irow in range(num_feats):
            cov_mat[irow] = dcor.rowwise(
                self.method_func,
                np.tile(data[irow], (num_feats, 1)),
                data, compile_mode=dcor.CompileMode.COMPILE_PARALLEL)

        return cov_mat


    def d_cov(self, samples):
        import dcor._fast_dcov_avl
        r"""Compute sample distance covariance matrix.

        Parameters
        ----------
        samples : 2d numpy array of floats
                A :math:`N \times M` matrix with :math:`N` samples of
                :math:`M` random variables.

        Returns
        -------
        2d numpy array
            A square matrix :math:`C`, where :math:`C_{i,j}` is the sample
            distance covariance between random variables :math:`R_i` and
            :math:`R_j`.

        """
        num_samps, num_feats = samples.shape
        cov_mat = np.empty((num_feats, num_feats), float)
        row, col = np.tril_indices(num_feats)
        if self.multiprocess:
            data = samples.T
            for irow in range(num_feats):
                cov_mat[irow] = dcor.rowwise(
                    dcor.distance_covariance,
                    np.tile(data[irow], (num_feats, 1)),
                    data, compile_mode=dcor.CompileMode.COMPILE_PARALLEL)
        else:
            for pair in zip(row, col):
                feat_p1, feat_p2 = samples[:, pair].T
                if pair == pair[::-1]:
                    cov_mat[pair] = dcov(feat_p1, feat_p2)
                else:
                    cov_mat[pair] = cov_mat[pair[::-1]] = dcov(feat_p1, feat_p2)
        return cov_mat

    def d_cor(self, samples):

        import dcor._fast_dcov_avl
        num_samps, num_feats = samples.shape
        cor_mat = np.empty(shape=(num_feats, num_feats))
        row, col = np.tril_indices(num_feats)
        if self.multiprocess:
            data = samples.T
            for irow in range(num_feats):
                cor_mat[irow] = dcor.rowwise(
                    dcor.distance_correlation,
                    np.tile(data[irow], (num_feats, 1)),
                    data, compile_mode=dcor.CompileMode.COMPILE_PARALLEL)
        else:
            for pair in zip(row, col):
                feat_p1, feat_p2 = samples[:, pair].T
                if pair == pair[::-1]:
                    cor_mat[pair] = dcor(feat_p1, feat_p2)
                else:
                    cor_mat[pair] = cor_mat[pair[::-1]] = dcor(feat_p1, feat_p2)

        return cor_mat

    def d_cov_sqr(self, samples):
        num_samps, num_feats = samples.shape
        dists = np.zeros((num_feats, num_samps, num_samps))
        # compute doubly centered distance matrix for every feature:
        for feat_idx in range(num_feats):
            n = num_samps
            t = np.tile
            # raw distance matrix:
            d = squareform(pdist(samples[:, feat_idx].reshape(-1, 1),
                                 "cityblock"))
            # doubly centered:
            d -= t(d.mean(0), (n, 1)) + t(d.mean(1), (n, 1)).T - t(d.mean(), (n, n))
            dists[feat_idx] = d / n
        cov = np.zeros((num_feats, num_feats))
        for i in range(num_feats):
            for j in range(num_feats):
                if j < i:
                    cov[i, j] = cov[j, i]
                else:
                    cov[i, j] = np.sum(dists[i]*dists[j])
                # print(f'{i}/{num_feats}:{j}/{num_feats}')
        return cov

    def d_cor_sqr(self, samples):
        dCov = np.sqrt(self.d_cov_sqr(samples))
        dVar = np.diag(dCov)
        return dCov / np.sqrt(np.outer(dVar, dVar))

