#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.ndimage import interpolation

np.seterr(divide='ignore', invalid='ignore')


def my_gaussian_filter(gaussian_size):
    h = np.fromfunction(lambda x, y: (1 / (2 * math.pi * gaussian_size[2] ** 2)) *
                                     math.e ** ((-1 * ((x - (gaussian_size[0] - 1) / 2) ** 2 +
                                                       (y - (gaussian_size[1] - 1) / 2) ** 2))
                                                / (2 * gaussian_size[2] ** 2)),
                        (gaussian_size[0], gaussian_size[1])).astype('float64')
    h[h < np.finfo(np.float).eps * np.amax(h)] = 0
    sum_h = np.sum(np.sum(h))
    if sum_h != 0:
        h = h / sum_h
    return h


def simple_demon(im_moving, im_static, times, gaussian_size, alpha):
    s = im_static
    m = im_moving
    # my_filter = my_gaussian_filter(gaussian_size)
    x, y = np.meshgrid(range(im_moving.shape[0]), range(im_moving.shape[1]), indexing='ij')
    field = np.array([x, y]).astype('float64')
    dsdy, dsdx = np.gradient(s)

    for ii in range(times):
        image_diff = m - s
        dmdy, dmdx = np.gradient(m)
        u_cols = -image_diff * (dsdy / ((dsdy ** 2 + dsdx ** 2) + (alpha ** 2) * (image_diff ** 2)) + dmdy / (
                (dmdy ** 2 + dmdx ** 2) + (alpha ** 2) * (image_diff ** 2)))
        u_rows = -image_diff * (dsdx / ((dsdy ** 2 + dsdx ** 2) + (alpha ** 2) * (image_diff ** 2)) + dmdx / (
                (dmdy ** 2 + dmdx ** 2) + (alpha ** 2) * (image_diff ** 2)))
        u_cols[np.isnan(u_cols)] = 0
        u_rows[np.isnan(u_rows)] = 0
        # u_cols = 3 * scipy.ndimage.filters.convolve(u_cols, my_filter, mode='nearest')
        # u_rows = 3 * scipy.ndimage.filters.convolve(u_rows, my_filter, mode='nearest')
        # kernel size radius = int(truncate*sigma+0.5)
        u_cols = 3 * scipy.ndimage.gaussian_filter(u_cols, 4, order=0, mode='nearest', truncate=10)
        u_rows = 3 * scipy.ndimage.gaussian_filter(u_rows, 4, order=0, mode='nearest', truncate=10)
        field[0] += u_cols
        field[1] += u_rows
        m = interpolation.map_coordinates(im_moving, field, mode='nearest', order=1)

    field[0] = field[0] - x
    field[1] = field[1] - y
    return field, m
