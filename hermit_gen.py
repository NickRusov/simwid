import numpy as np


def get_hermit(dimensions=4, max_minor=3):
    diag = [1]*dimensions
    diag[-1] = max_minor
    pre_last = np.random.randint(max_minor, size=dimensions - 1)
    hermit = np.diag(diag)
    hermit[dimensions-1][:-1] = pre_last
    return hermit


def minor_mat(arr, i=0, j=0):
    # ith row, jth column removed
    return arr[np.array(range(i)+range(i+1, arr.shape[0]))[:, np.newaxis],
               np.array(range(j)+range(j+1, arr.shape[1]))]


def get_max_minor(hermite):
    # return [abs(d) for d in hermite: ]
    for minor_counter in xrange(hermite.shape[1]):
        yield abs(np.linalg.det(minor_mat(hermite, i=0, j=minor_counter)))
        # sim = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


if __name__ == '__main__':
    a = get_hermit(6, 4)
    minors = list(get_max_minor(a))
    print minors
