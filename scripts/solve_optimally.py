import numpy as np
from methods import *
import itertools
import time

def gen_perms_of_perms(curr_perm_list, n, T):
    global total_perm_list

    if len(curr_perm_list) == T:
        total_perm_list.append(curr_perm_list)
        return
    else:
        for perm in itertools.permutations(range(n)):
            gen_perms_of_perms(curr_perm_list + [perm], n, T)

def solve_optimally(benefits, init_ass, lambda_):
    """
    Given a benefit matrix and an initial assignment,
    exhaustively search for and find the optimal value
    assignment matrix.

    Only works for extremely small problems due to the exhaustive search.
    (i.e. 5x5x3)

    Returns best assignments, best value, and 
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    global total_perm_list
    total_perm_list = []
    gen_perms_of_perms([], n, T)
    
    best_value = -np.inf
    best_nh = None
    best_assignments = None

    for perm_list in total_perm_list:
        assignment_list = []
        for perm in perm_list:
            ass = np.zeros((n,m))
            for i, j in enumerate(perm):
                ass[i,j] = 1
            assignment_list.append(ass)

        total_value, nh = calc_value_and_num_handovers(assignment_list, benefits, init_ass, lambda_)

        if total_value > best_value:
            best_value = total_value
            best_assignments = assignment_list
            best_nh = nh

    return best_assignments, best_value, best_nh

if __name__ == "__main__":
    pass