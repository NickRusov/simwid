#!/usr/bin/python2
import numpy as np
from copy import copy
from numpy.linalg import det 

def get_hermit(dimensions=5, max_minor=2):
    diag = [1]*(dimensions)
    diag[-1] = max_minor
    pre_last = np.random.randint(max_minor, size=dimensions - 1)
    hermit = np.append(np.diag(diag), np.random.randint(-max_minor, max_minor + 1, size=(1, dimensions)), 0)
    hermit[dimensions-1][:-1] = pre_last
    return hermit


def minor_mat(arr, i=0):
#     ith row, jth column removed
    return arr[np.array(range(i)+range(i+1, arr.shape[0]))[:, np.newaxis],
               np.array(range(arr.shape[1]))]


# def minor_mat(arr, i=0, j=0):
    # ith row, jth column removed
#     return arr[np.array(range(i)+range(i+1, arr.shape[0]))[:, np.newaxis],
#                np.array(range(j)+range(j+1, arr.shape[1]))]

def exclude_row(arr, i=0):
    # ith row, jth column removed
    return arr[np.array(chain(range(i),range(i+1, arr.shape[0])))[:, np.newaxis],
               np.array(range(arr.shape[1]))]


def get_det_of_minor(hermite):
    # return [abs(d) for d in hermite: ]
    for minor_counter in range(hermite.shape[1]):
        yield abs(det(minor_mat(hermite, i=0, j=minor_counter)))
        # prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


if __name__ == '__main__':
	h = get_hermit()
 	print(h)
	print('\n')
	current_det = 0
	# print(h[chain(range(1,2), range(3,5))])
	# print(exclude_row(h, 0))

	for i in range(h.shape[0]):
	    current_det = det(minor_mat(h,i))
	    if abs(current_det) > 0.0:
		print('%i - %f' % (i,current_det))
