#!/usr/bin/env python
import numpy as np
from numpy.linalg import det, inv, matrix_rank
# from sympy import Matrix, symbols, EmptySet
# from sympy.solvers.solveset import linsolve
from copy import copy
from cvxopt.glpk import ilp
from cvxopt import matrix


def get_hermit(dimensions=6, max_minor = 5):
    diag = np.array([1]*(dimensions)).astype('i')
    diag[-1] = max_minor
    pre_last = np.random.randint(max_minor, size=dimensions - 1)
    hermit = np.diag(diag)
    hermit[dimensions-1][:-1] = pre_last
    return hermit


def apply_restrictor(hermit, dimensions=6, max_minor = 5):
    continue_flag = True
    restricted = 0
    while continue_flag:
        restricted = np.append(hermit, np.random.randint(-max_minor, max_minor + 1, size=(1, dimensions)), 0)
        continue_flag = not is_valid(restricted)
    return restricted


def minor(arr, i=0):
#     ith row removed
    return arr[np.array(range(i)+range(i+1, arr.shape[0]))[:, np.newaxis],
               np.array(range(arr.shape[1]))]


def is_valid(A):
    return all(det(minor(A,i)) for i in xrange(A.shape[0]))
    
def is_valid_solution(solution):
    return  any(x for x in solution)

if __name__ == '__main__':
	dimensions = 3
	p = 2

	h = get_hermit(dimensions, p)
	cone = apply_restrictor(h, dimensions, p)
	# cone = np.ndarray(cone, dtype=float)
	# print cone

	c = cone[-1]
	x = np.zeros(shape=(dimensions,1), dtype=float)
	x[-1] = float(dimensions)/float(p)

	sigma = c.dot(x)  # ( b[n+1])

	f = copy(x)#np.reshape(np.append(x, sigma), newshape=(dimensions + 1, 1))
	f[-1] = float(dimensions + 1)/float(p)

	# print 'f (vector-indicator`s last value): %s' % f[-1]  #  == sol

	if c.dot(f) > sigma:
	    print 'f belongs to M={x: cx >= sigma}'
	else:
	    print 'f doesn`t belong to M={x: cx >= sigma}'
    
	target_f = matrix(np.ones((dimensions + 1,1),dtype=float), tc='d')
	target_f[0] = 1

	# print target_f
	# print matrix(cone, tc='d')
	# print matrix(f, tc='d')
	# print set(range(dimensions))

	# cycle for changing sigma

	(status,s) = ilp(target_f, 
			 G=matrix(-cone, tc='d'),
			 h=matrix(np.append(f, sigma), tc='d'),
	#                  A=matrix(1., (0,6)),b=matrix(1., (0,1)),
			 I=set(range(dimensions)),
	#                  B=set()
			) 
	if status == 'LP relaxation is primal infeasible':
	    print 'Primal infeasible'
	if is_valid_solution(s):
	    print s
	else:
	    print 'Trivial solution.'
	    print s

	print matrix_rank(cone)
	print cone
	b = np.append(f, sigma)
	print 'dots\n'
	
	for i in xrange(cone.shape[0]):
		print inv(minor(cone,i)).dot(b[range(i)+range(i+1, b.shape[0])])