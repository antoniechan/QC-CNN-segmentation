#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cmath
import scipy
import numpy as np
from package_functions import MeshOperator
from scipy import sparse
from scipy.sparse import linalg
from operator import itemgetter

from package_functions.MyUtil import vertex_search, close_curve_division, image_free_boundary


def mu_metric(v, f, mapping, dimension):
    (dx, dy, dz, dc) = MeshOperator.diff_operator(v, f)
    if len(mapping) > len(mapping[0]):
        mapping = mapping.transpose()
    if dimension == 2:
        f = mapping[0, :] + 1j * mapping[1, :]
        mu = (dc * np.transpose([f])) / (dz * np.transpose([f]))
    elif dimension == 3:
        dXdu = np.concatenate(dx * np.transpose([mapping[0, :]]))
        dXdv = np.concatenate(dy * np.transpose([mapping[0, :]]))
        dYdu = np.concatenate(dx * np.transpose([mapping[1, :]]))
        dYdv = np.concatenate(dy * np.transpose([mapping[1, :]]))
        dZdu = np.concatenate(dx * np.transpose([mapping[2, :]]))
        dZdv = np.concatenate(dy * np.transpose([mapping[2, :]]))
        E = dXdu ** 2 + dYdu ** 2 + dZdu ** 2
        G = dXdv ** 2 + dYdv ** 2 + dZdv ** 2
        F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv
        mu = np.transpose([(E - G + 2 * 1j * F) / (E + G + 2 * np.sqrt(E * G - F ** 2))])
    else:
        print('Dimension should either be 2 or 3. Please check again.')
        mu = []
    return mu


def lbs_rect(face, vertex, mu, height, width):
    Ax, abc, area = generalized_laplacian2d(face, vertex, mu)
    Ay = Ax.copy()
    bx = np.zeros(len(vertex)) if len(vertex) > len(vertex[0]) else np.zeros(len(vertex[0]))
    by = bx.copy()
    corner = vertex_search(np.array([[np.min(vertex[:, 0]), np.min(vertex[:, 1])], [np.max(vertex[:, 0]), np.min(vertex[:, 1])],
                                     [np.max(vertex[:, 0]), np.max(vertex[:, 1])], [np.min(vertex[:, 0]), np.max(vertex[:, 1])]]), vertex.transpose())
    Edge = close_curve_division(image_free_boundary(height, width), corner)
    vBdyC = [*Edge[3], *Edge[1]]
    vBdy = [*Edge[3] * 0, *(Edge[1] * 0 + width - 1)]
    landmarkx = vBdyC
    targetx = vBdy
    bx[landmarkx] = targetx
    Ax[landmarkx, :] = 0
    for i in landmarkx:
        Ax[i, i] = 1
    mapx = scipy.sparse.linalg.spsolve(Ax, bx)
    hBdyC = [*Edge[0], *Edge[2]]
    hBdy = [*Edge[0] * 0, *(Edge[2] * 0 + width - 1)]
    landmarky = hBdyC
    targety = hBdy
    by[landmarky] = targety
    Ay[landmarky, :] = 0
    for i in landmarky:
        Ay[i, i] = 1
    mapy = scipy.sparse.linalg.spsolve(Ay, by)
    maps = np.squeeze([mapx, mapy, 0 * vertex[:, 0] + 1]).transpose()
    mu = mu_metric(vertex, face, maps, 2)
    return maps, mu, Edge


def generalized_laplacian2d(face, vertex, mu):
    af = (1 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1 - np.abs(mu) ** 2)
    bf = -2 * np.imag(mu) / (1 - np.abs(mu) ** 2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1 - np.abs(mu) ** 2)
    abc = np.squeeze([af, bf, gf]).transpose()
    f0 = face[:, 0]
    f1 = face[:, 1]
    f2 = face[:, 2]
    uxv0 = vertex[f1, 1] - vertex[f2, 1]
    uyv0 = vertex[f2, 0] - vertex[f1, 0]
    uxv1 = vertex[f2, 1] - vertex[f0, 1]
    uyv1 = vertex[f0, 0] - vertex[f2, 0]
    uxv2 = vertex[f0, 1] - vertex[f1, 1]
    uyv2 = vertex[f1, 0] - vertex[f0, 0]
    l = np.squeeze([np.sqrt(uxv0 ** 2 + uyv0 ** 2), np.sqrt(uxv1 ** 2 + uyv1 ** 2), np.sqrt(uxv2 ** 2 + uyv2 ** 2)]).transpose()
    s = np.sum(l, axis=1) * 0.5
    area = np.sqrt(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]))
    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area
    I = np.squeeze([f0, f1, f2, f0, f1, f1, f2, f2, f0]).flatten()
    J = np.squeeze([f0, f1, f2, f1, f0, f2, f1, f0, f2]).flatten()
    V = np.squeeze([v00, v11, v22, v01, v01, v12, v12, v20, v20]).flatten()
    A = scipy.sparse.csr_matrix((V, (I, J)), dtype=np.float) * 0.5
    return A, abc, area


def mu_chop(mu, bound, val):
    for ii in range(len(mu)):
        if abs(mu[ii]) > bound:
            mu[ii] = val * cmath.cos(cmath.phase(mu[ii])) + 1j * val * cmath.sin(cmath.phase(mu[ii]))
    return mu
