import numpy as np
from scipy.spatial.distance import pdist, squareform


def d_cov_sqr(samples):
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
    num_pairs = num_samps * (num_samps - 1) // 2
    dists = np.zeros((num_feats, num_pairs))
    # compute doubly centered distance matrix for every feature:
    for feat_idx in range(num_feats):
        n = num_samps
        t = np.tile
        # raw distance matrix:
        d = squareform(pdist(samples[:, feat_idx].reshape(-1, 1), "cityblock"))
        # doubly centered:
        d -= t(d.mean(0), (n, 1)) + t(d.mean(1), (n, 1)).T - t(d.mean(), (n, n))
        d = squareform(d, checks=False)  # ignore assymmetry do to numerical error
        dists[feat_idx] = d
    return dists @ dists.T / num_samps ** 2


# Compare to faster implementation in dcor package and to pearson cov
from dcor import distance_covariance as dcov


def d_cov_sqr2(samples):
    num_samps, num_feats = samples.shape
    cov_mat = np.empty(shape=(num_feats, num_feats))
    row, col = np.tril_indices(num_feats)
    for pair in zip(row, col):
        feat_p1, feat_p2 = samples[:, pair].T
        if pair == pair[::-1]:
            cov_mat[pair] = dcov(feat_p1, feat_p2)
        else:
            cov_mat[pair] = cov_mat[pair[::-1]] = dcov(feat_p1, feat_p2)
    return cov_mat


def generate_data(num_samps):
    rng = np.random.default_rng(651469887)
    samples = np.empty(shape=(num_samps, 3))
    exog = rng.normal(0, 1, (num_samps, 3))
    samples[:, 0] = exog[:, 0]
    samples[:, 1] = exog[:, 1]
    samples[:, 2] = 2 * np.sqrt(np.abs(1 - samples[:, 0] ** 2)) + exog[:, 2]
    return samples  # 1 indep of 0, 2 and 0 dep with 2


samps = generate_data(10000)
p_cov = np.cov(samps, rowvar=False)
d_cov = d_cov_sqr(samps)
d_cov2 = d_cov_sqr2(samps)
import pdb;pdb.set_trace()
