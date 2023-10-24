import numpy as np
from methods import *
import itertools
import time

def generate_matrices(n, m, T):
    # This function will generate all possible matrices for given dimensions
    max = 2**(n * m * T)
    for i in range(max):
        print(f"{i/max}", end='\r')
        matrix = [[[0 for _ in range(T)] for _ in range(m)] for _ in range(n)]
        # Convert i to binary and fill the matrix
        bin_rep = format(i, f'0{n*m*T}b')
        idx = 0
        for x in range(n):
            for y in range(m):
                for z in range(T):
                    matrix[x][y][z] = int(bin_rep[idx])
                    idx += 1
        matrix = np.array(matrix)

        if not np.all(np.sum(matrix, axis=0) == 1) or not np.all(np.sum(matrix, axis=1) == 1):
            continue
        # if not np.array_equal(np.sum(matrix, axis=0), np.ones_like(np.sum(matrix, axis=0))) or not np.array_equal(np.sum(matrix, axis=1), np.ones_like(np.sum(matrix, axis=1))):
        #     continue
        yield matrix

# def generate_permutation_matrices(n, m):
#     # Generate all permutation matrices of size n x m
#     base = [1] + [0] * (m - 1)
#     for perm in itertools.permutations(base, m):
#         yield np.array(list(perm))

# def generate_3d_matrices(n, m, T):
#     # Generate all 3D matrices with row and column sums equal to 1 for each time slice
#     perm_matrices = list(generate_permutation_matrices(n, m))
#     for matrix_combination in itertools.product(perm_matrices, repeat=T):
#         yield np.array(list(matrix_combination))

if __name__ == "__main__":
    #Aims to compute a true optimal solution via exhaustive search.

    n = 4
    m = 4
    T = 2

    benefit = np.random.rand(n,m,T)
    init_ass = np.array([[0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]])
    benefit[:,:,0] = np.array([[100, 1, 0, 0],
                               [1, 100, 0, 0],
                               [0, 0, 0.1, 0.2],
                               [0, 0, 0.2, 0.1]])
    benefit[:,:,1] = np.array([[1000, 1, 0, 0],
                               [1, 1000, 0, 0],
                               [0, 0, 0.3, 0.1],
                               [0, 0, 0.1, 0.3]])
    lambda_ = 1

    best_benefit = -np.inf
    best_assignment = None
    #generate all possible assignments
    for assignments in generate_matrices(n,m,T):
        assignment_list = [assignments[:,:,j] for j in range(T)]

        total_benefit = 0
        for j, ass in enumerate(assignment_list):
            total_benefit += (benefit[:,:,j]*ass).sum()

        total_benefit -= calc_handover_penalty(init_ass, assignment_list, lambda_)

        if total_benefit > best_benefit:
            best_benefit = total_benefit
            best_assignment = assignment_list

    print(f"Best Assignment: (for {best_benefit} benefit):")
    for ass in best_assignment:
        print(ass)