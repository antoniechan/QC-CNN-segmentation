import numpy as np
import scipy
from operator import itemgetter


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])
    return rows, cols


def ismember(A, B):
    tf = [np.sum(a == B) for a in A]
    index = np.zeros((len(tf),), dtype='int')
    for i in range(len(index)):
        if tf[i]:
            index[i] = np.where(B == A[i])[0]
        else:
            index[i] = -1
    return tf, index


def image_meshgen(height, width):
    x, y = np.meshgrid(np.arange(height), np.arange(width))
    vertex = np.array([y.flatten(), x.flatten()]).transpose()
    n, m = vertex.shape
    temp = np.arange(height - 1)
    for i in range(width - 2):
        temp = np.hstack((temp, np.arange(height - 1) + (i + 1) * height))
    face = np.vstack((np.vstack((temp, np.vstack((temp + height, temp + 1)))).transpose(), np.vstack((temp + 1, np.vstack((temp + height, temp + height + 1)))).transpose()))
    vertex = np.hstack((vertex, np.ones((n, 1))))
    return vertex, face


def image_free_boundary(height, width):
    return np.vstack((np.concatenate((np.arange(0, height * width - height, width),
                                      np.arange(height * width - height, height * width - 1),
                                      np.flip(np.arange(2 * height - 1, height * width, width)),
                                      np.flip(np.arange(1, height)))),
                      np.concatenate((np.arange(height, height * width, width),
                                      np.arange(height * width - height + 1, height * width),
                                      np.flip(np.arange(height - 1, height * width - height, width)),
                                      np.flip(np.arange(0, height - 1)))))).transpose()


def vertex_search(XY, vertex):
    k = len(XY)
    index = np.zeros(shape=(k,))
    for i in range(k):
        index[i] = min(enumerate(np.sqrt((vertex[0, :] - XY[i, 0]) ** 2 + (vertex[1, :] - XY[i, 1]) ** 2)), key=itemgetter(1))[0]
    return index


def close_curve_division(B, pt):
    for i in range(1, len(B) - 1):
        index1 = np.where(B[i::, 0] == B[i - 1, 1])
        tempa = B[i + index1[0], :]
        tempb = B[i, :]
        B[i, :] = tempa
        B[i + index1[0], :] = tempb
    n = len(pt)
    Edge = []
    _, location = ismember(pt, B[:, 0])
    sort_location = np.sort(location)
    index = np.argsort(location)
    for i in range(n - 1):
        Edge.append(B[np.arange(sort_location[i], sort_location[i + 1] + 1), 0])
    if sort_location[0] == 0:
        Edge.append(np.array([*B[sort_location[-1]::, 0], *[B[0, 0]]]))
    else:
        Edge.append(np.array([*B[sort_location[-1]::, 0], *B[np.arange(sort_location[0]), 0]]))
    Edge = np.squeeze(Edge)
    temp = []
    for i in index:
        temp.append(Edge[i])
    Edge = np.squeeze(temp)
    return Edge
