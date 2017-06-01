#!/usr/bin/env python
import numpy as np
from math import ceil, floor
from numpy.linalg import det, inv, matrix_rank
from copy import copy
from cvxopt.glpk import ilp
from cvxopt import matrix
from itertools import permutations, combinations

dimensions = 4
max_minor = 5

def minor(arr, i=0):  #  ith row removed
    return arr[np.array(range(i)+range(i+1, arr.shape[0]))[:, np.newaxis],
               np.array(range(arr.shape[1]))]


def get_minors(A):
    return [det(minor(A,i)) for i in xrange(A.shape[0])]


def generate_vertices(cone, b):
    vertices = []
    for i in xrange(cone.shape[0]):
        vertices.append(inv(minor(cone,i)).dot(b[range(i)+range(i+1, b.shape[0])]))
    return vertices


def is_valid_solution(solution):
    return  any(x for x in solution)

#         print restriction
#         restriction = np.asarray(restriction)#np.ndarray(buffer=np.asarray(restriction)))#, shape=(1, dimensions)))
def bruteforce(results, restriction):
    print restriction
    restricted = np.append(hermit, np.matrix([np.asarray(restriction)]), 0) # np.ndarray(buffer=restriction, shape=(1, dimensions))
    minors = get_minors(restricted)
    if not all(minors):
        return
    abs_minors = [abs(m) for m in minors]
    min_abs_minor = min(abs_minors)
    max_abs_minor = max(abs_minors)

    c = restricted[-1]
    x = np.zeros(shape=(dimensions,1), dtype=float)
    x[-1] = float(dimensions)/float(max_minor)
    sigma = c.dot(x)  # ( b[n+1])
    f = copy(x)  # np.reshape(np.append(x, sigma), newshape=(dimensions + 1, 1))
    f[-1] = float(dimensions + 1)/float(max_minor)

    is_in_M = c.dot(f) > sigma
    target_f = matrix(-np.ones((dimensions + 1,1),dtype=float), tc='d')
    s = None
    if is_in_M:
        sigma = floor(sigma)
        while not s:
#                 print 'sigma:', sigma
            b = np.append(f, sigma)
            (status,s) = ilp(target_f, 
                 G=matrix(restricted, tc='d'),
                 h=matrix(b, tc='d'),
                 I=set(xrange(dimensions)),
                )

            sigma += 1
            if status != 'optimal':
#                     print status
                break
        sigma -= 1
    else:
        sigma = ceil(sigma)
        while not s:
#                 print 'sigma:', sigma
            b = np.append(f, sigma)
            (status,s) = ilp(target_f, 
                 G=matrix(-restricted, tc='d'),
                 h=matrix(-b, tc='d'),
                 I=set(xrange(dimensions)),
                )
            sigma -= 1
            if status != 'optimal' or abs(sigma) > 100:
#                     print status
                break
        sigma += 1
    if not s:
        return
    print 'sigma after cycle:', sigma
    print 'sol:', s
    
    b = np.append(f, sigma)
    max_b = max(b)
    vv = generate_vertices(restricted, b)
    
    width_list = list()
    for i,j in combinations(xrange(len(vv)), 2):
        target_f2 = vv[i] - vv[j]
        restrictions = np.concatenate((inv(minor(restricted, i).T), -inv(minor(restricted, j).T),
                                       np.diag(np.ones(dimensions).astype('d'))))
        
        (status,s) = ilp(matrix(target_f2), 
             G=matrix(-restrictions, tc='d'),
             h=matrix(np.ones(shape=(restrictions.shape[0],1), dtype=np.double), tc='d'),
             I=set(range(len(vv)-1)),
            )
        if status == 'optimal':
            width_c = abs(target_f2.dot(s)[0])
            if is_valid_solution(s):
                width_list.append(width_c)
            else:
                return
        else:
                return
    print 'The next result is going to be added: [0][0]'
    result_ = ( min(width_list), dimensions, int(min_abs_minor), int(max_abs_minor), max_b ) 
    print(result_)
    with open( 'res_new.csv', 'a' ) as f:
        f.write(';'.join([str(l) for l in result_]) + '\r\n')
#     results.append(result_)
#     results.append( ( min(width_list), dimensions, int(min_m), int(max_m), max_b ) )
    
diag = np.array([1]*(dimensions)).astype('i')
diag[-1] = max_minor
hermit = np.diag(diag)
# print hermit


from multiprocessing import Process, Value, Array, Manager

manager = Manager()
results = manager.list()

for pre_last_row in permutations(xrange(dimensions), dimensions-1):
    hermit[dimensions-1][:-1] = list(pre_last_row)
    
    for restriction in permutations(xrange(-max_minor, max_minor + 1), dimensions):
        p = Process(target=bruteforce, args=(results, restriction))
        p.start()
        p.join(300)
        if p.is_alive():
            print "running... let's kill it..."
            # Terminate
            p.terminate()
            p.join()
        
        
with open( 'res_new.csv', 'a' ) as f:
        f.write('This is the end... My dear friend' + '\r\n')
