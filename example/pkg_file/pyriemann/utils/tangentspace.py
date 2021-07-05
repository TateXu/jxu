import numpy
from .base import sqrtm, invsqrtm, logm, expm
import pdb
###############################################################
# Tangent Space
###############################################################


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space according to the given reference point Cref

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: The reference covariance matrix
    :returns: the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape

    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = numpy.empty((Nt, Nf))
    if numpy.isrealobj(Cref):
        T = numpy.empty((Nt, Nf))
    else:
        import pickle
        import os
        if os.path.exists('cov_analytic.pkl'):
            with open('cov_analytic.pkl', "rb") as in_part:
                part = pickle.load(in_part)
        else:
            part = None
        if part is 'anal' or part == 'anal':
            T = numpy.empty((Nt, 2 * Nf))
        else:
            T = numpy.empty((Nt, Nf))  # , dtype=numpy.complex_

    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    for index in range(Nt):
        if numpy.isrealobj(Cm12):
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]), Cm12)
        else:
            tmp = numpy.dot(numpy.dot(Cm12, covmats[index, :, :]),
                            numpy.conj(Cm12).T)

        tmp = logm(tmp)
        if numpy.isrealobj(tmp):
            T[index, :] = numpy.multiply(coeffs, tmp[idx])
        else:
            if part is 'real' or part == 'real':
                T[index, :] = numpy.real(numpy.multiply(coeffs, tmp[idx]))
            elif part is 'imag' or part == 'imag':
                T[index, :] = numpy.imag(numpy.multiply(coeffs, tmp[idx]))
            elif part is 'anal' or part == 'anal':
                T[index, :Nf] = numpy.real(numpy.multiply(coeffs, tmp[idx]))
                T[index, Nf:] = numpy.imag(numpy.multiply(coeffs, tmp[idx]))

    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors in the manifold according to the given reference point Cref

    :param T: the Tangent space , a matrix of Ntrials X (Nchannels * (Nchannels + 1)/2)
    :param Cref: The reference covariance matrix
    :returns: A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    """
    if numpy.isrealobj(Cref):
        covmats = numpy.empty((Nt, Ne, Ne))
    else:
        covmats = numpy.empty((Nt, Ne, Ne), dtype=numpy.complex_)
    """
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = numpy.triu(covmats[i], 1) / numpy.sqrt(2)
        covmats[i] = (numpy.diag(numpy.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats
