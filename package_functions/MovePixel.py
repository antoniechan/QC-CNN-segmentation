#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import scipy.ndimage

# Equivalent to interpolation.map_coordinates with order=1, mode='outer_zero'
def move_pixel_2d(Iin, Tx, Ty, mode):
    # Make all x, y indices
    x, y = np.meshgrid(range(Iin.shape[0]), range(Iin.shape[1]), indexing='ij')

    # Calculate the Transformed coordinates
    Tlocalx = x + Tx
    Tlocaly = y + Ty

    # All the neighborh pixels involved in linear interpolation.
    xBas0 = np.floor(Tlocalx)
    yBas0 = np.floor(Tlocaly)
    xBas1 = xBas0 + 1
    yBas1 = yBas0 + 1

    # Linear interpolation constants (percentages)
    xCom = Tlocalx - xBas0
    yCom = Tlocaly - yBas0
    perc0 = (1 - xCom) * (1 - yCom)
    perc1 = (1 - xCom) * yCom
    perc2 = xCom * (1 - yCom)
    perc3 = xCom * yCom

    # limit indexes to boundaries
    check_xBas0 = (xBas0 < 0) | (xBas0 > (Iin.shape[0] - 1))
    check_yBas0 = (yBas0 < 0) | (yBas0 > (Iin.shape[1] - 1))
    xBas0[check_xBas0] = 0
    yBas0[check_yBas0] = 0
    check_xBas1 = (xBas1 < 0) | (xBas1 > (Iin.shape[0] - 1))
    check_yBas1 = (yBas1 < 0) | (yBas1 > (Iin.shape[1] - 1))
    xBas1[check_xBas1] = 0
    yBas1[check_yBas1] = 0

    Iout = []
    if len(Iin.shape) < 3:
        count = 1
        Iin = np.expand_dims(Iin, axis=2)
    else:
        count = Iin.shape[2]
    for i in range(count):
        Iin_one = Iin[:, :, 0]
        Iin_one_flat = Iin_one.flatten('F')
        # Get the intensities
        intensity_xyz0 = Iin_one_flat[(xBas0 + yBas0 * Iin.shape[0]).flatten().astype(int)].reshape(Iin_one.shape)
        intensity_xyz1 = Iin_one_flat[(xBas0 + yBas1 * Iin.shape[0]).flatten().astype(int)].reshape(Iin_one.shape)
        intensity_xyz2 = Iin_one_flat[(xBas1 + yBas0 * Iin.shape[0]).flatten().astype(int)].reshape(Iin_one.shape)
        intensity_xyz3 = Iin_one_flat[(xBas1 + yBas1 * Iin.shape[0]).flatten().astype(int)].reshape(Iin_one.shape)
        # Make pixels before outside Ibuffer mode
        if mode == 1 | mode == 3:
            intensity_xyz0[check_xBas0 | check_yBas0] = 0
            intensity_xyz1[check_xBas0 | check_yBas1] = 0
            intensity_xyz2[check_xBas1 | check_yBas0] = 0
            intensity_xyz3[check_xBas1 | check_yBas1] = 0
        Iout_one = intensity_xyz0 * perc0 + intensity_xyz1 * perc1 + intensity_xyz2 * perc2 + intensity_xyz3 * perc3
        print(Iout_one)
        Iout.append(np.reshape(Iout_one, [Iin.shape[0], Iin.shape[1]]))

    return Iout
