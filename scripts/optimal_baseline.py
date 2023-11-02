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

def gen_perms_of_perms(curr_perm_list, n, T):
    global total_perm_list

    if len(curr_perm_list) == T:
        total_perm_list.append(curr_perm_list)
        return
    else:
        for perm in itertools.permutations(range(n)):
            gen_perms_of_perms(curr_perm_list + [perm], n, T)

if __name__ == "__main__":
    #Aims to compute a true optimal solution via exhaustive search.

    n = 3
    m = 3
    T = 2

    global total_perm_list
    total_perm_list = []
    gen_perms_of_perms([], n, T)
    print(total_perm_list)

    # benefit = np.random.rand(n,m,T)
    # init_ass = np.array([[0, 0, 0, 1],
    #                      [0, 0, 1, 0],
    #                      [0, 1, 0, 0],
    #                      [1, 0, 0, 0]])
    # benefit[:,:,0] = np.array([[100, 1, 0, 0],
    #                            [1, 100, 0, 0],
    #                            [0, 0, 0.1, 0.2],
    #                            [0, 0, 0.2, 0.1]])
    # benefit[:,:,1] = np.array([[1000, 1, 0, 0],
    #                            [1, 1000, 0, 0],
    #                            [0, 0, 0.3, 0.1],
    #                            [0, 0, 0.1, 0.3]])
    # lambda_ = 1

    # best_benefit = -np.inf
    # best_assignment = None
    # #generate all possible assignments
    # for assignments in generate_matrices(n,m,T):
    #     assignment_list = [assignments[:,:,j] for j in range(T)]

    #     total_benefit = 0
    #     for j, ass in enumerate(assignment_list):
    #         total_benefit += (benefit[:,:,j]*ass).sum()

    #     total_benefit -= calc_handover_penalty(init_ass, assignment_list, lambda_)

    #     if total_benefit > best_benefit:
    #         best_benefit = total_benefit
    #         best_assignment = assignment_list

    # print(f"Best Assignment: (for {best_benefit} benefit):")
    # for ass in best_assignment:
    #     print(ass)