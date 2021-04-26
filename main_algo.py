#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from scipy.sparse.linalg import spsolve
from package_functions import MyUtil, MeshOperator, SimpleDemon
from package_functions.BeltramiCoef import mu_metric, mu_chop, lbs_rect


def segment_image_descent(moving, static, iteration=None, demon_iteration=None, gaussian_size=None, delta=None, lambd=None, sigma_increase=None):

    if iteration is None:
        iteration = 20
    if demon_iteration is None:
        demon_iteration = 100
    if gaussian_size is None:
        gaussian_size = [20, 20, 8]
    if delta is None:
        delta = 0.1
    if lambd is None:
        lambd = 0.1
    if sigma_increase is None:
        sigma_increase = 2

    mu = []
    C1_new = 1
    C2_new = 0
    stopcount = 0
    tolerance = 1e-3
    temp_moving = moving
    s1, s2 = static.shape
    vertex, face = MyUtil.image_meshgen(s1, s2)
    lb_operator = MeshOperator.laplace_beltrami(vertex, face)
    f2v = MeshOperator.f2v(vertex, face)
    v2f = MeshOperator.v2f(vertex, face)
    maps = vertex.copy()
    updated_map = np.zeros((len(vertex), 2))

    for k in range(iteration):

        # Update moving image
        C1_old = C1_new
        C2_old = C2_new
        med = (C1_old + C2_old) / 2
        idx_med = np.ones((s1, s2))
        idx_med[temp_moving <= med] = 0
        temp_moving = C1_old * idx_med + C2_old * (np.ones((s1, s2)) - idx_med)

        # Update modified demon descent and update the registration function (mu-subproblem)
        field, _ = SimpleDemon.simple_demon(temp_moving, static, demon_iteration, gaussian_size, 2.5)
        interpolator = interp.CloughTocher2DInterpolator(vertex[:, 0:2], field[0].flatten('F'))
        b_height = interpolator(maps[:, 0], maps[:, 1])
        interpolator = interp.CloughTocher2DInterpolator(vertex[:, 0:2], field[1].flatten('F'))
        b_width = interpolator(maps[:, 0], maps[:, 1])
        updated_map[:, 0] = maps[:, 0] - b_width
        updated_map[:, 1] = maps[:, 1] - b_height

        # Smoothen the Beltrami Coefficient (nu-subproblem)
        mu = mu_metric(vertex, face, updated_map, 2)
        delta = delta + sigma_increase
        mu = mu_chop(mu, 0.9999, 0.95)
        vmu = np.array(f2v * mu)
        smooth_operator = (1 + delta) * scipy.sparse.eye(len(vmu)) - 0.5 * lambd * lb_operator
        nvmu = scipy.sparse.linalg.spsolve(smooth_operator, delta * np.abs(vmu))
        vmu = nvmu * np.squeeze((np.cos(np.angle(vmu)) + 1j * np.sin(np.angle(vmu))))
        mu = v2f * vmu
        mu = mu_chop(mu, 0.9999, 0.95)
        updated_map, mu, _ = lbs_rect(face, vertex, mu, s1, s2)

        # Update the template image
        temp_moving = update_mapping(maps, updated_map, temp_moving, s1, s2)

        # Update c1, c2
        v_static = static.flatten()
        C1_new = np.mean(v_static[temp_moving.flatten() >= np.mean(temp_moving.flatten())])
        C2_new = np.mean(v_static[temp_moving.flatten() < np.mean(temp_moving.flatten())])

        # Display intermediate result
        if np.mod(k+1, 1) == 0:
            fig, ax = plt.subplots()
            ax.imshow(np.abs(static - temp_moving), cmap='gray')
            plt.title('Iteration {0}'.format(k+1))
            ax.axis((0, static.shape[0], static.shape[1], 0))
            plt.show()

        # Stopping criterion
        if (np.abs(C1_new - C1_old) < tolerance) & (np.abs(C2_new - C2_old) < tolerance):
            stopcount += 1
        else:
            stopcount = 0
        if stopcount == 5:
            fig, ax = plt.subplots()
            ax.imshow(np.abs(static - temp_moving), cmap='gray')
            plt.title('Iteration [{0}]. Reached stopping criterion'.format(k))
            ax.axis((0, static.shape[0], static.shape[1], 0))
            plt.show()
            break

        maps = updated_map.copy()

    # Output result
    map_mu = mu.copy()
    register = update_mapping(vertex, updated_map, moving, s1, s2)

    return maps, map_mu, register


def update_mapping(map_original, updated_map, moving, s1, s2):
    x, y = np.meshgrid(range(moving.shape[0]), range(moving.shape[1]), indexing='ij')
    inverse_vector = map_original[:, 0:2] - updated_map[:, 0:2]
    interpolator = interp.CloughTocher2DInterpolator(updated_map[:, 0:2], inverse_vector[:, 0].flatten('F'))
    ivec1 = interpolator(map_original[:, 0], map_original[:, 1])
    interpolator = interp.CloughTocher2DInterpolator(updated_map[:, 0:2], inverse_vector[:, 1].flatten('F'))
    ivec2 = interpolator(map_original[:, 0], map_original[:, 1])
    b_height = np.reshape(ivec2, newshape=(s1, s2)).transpose()
    b_width = np.reshape(ivec1, newshape=(s1, s2)).transpose()
    field = np.array([x, y]).astype('float64')
    field[0] += b_height
    field[1] += b_width
    out_map = interpolation.map_coordinates(moving, field, mode='nearest', order=1)
    return out_map
