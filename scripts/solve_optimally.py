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

def solve_optimally(benefit, init_ass, lambda_):
    n = benefit.shape[0]
    m = benefit.shape[1]
    T = benefit.shape[2]

    global total_perm_list
    total_perm_list = []
    gen_perms_of_perms([], n, T)
    
    best_benefit = -np.inf
    best_assignment = None

    for perm_list in total_perm_list:
        assignment_list = []
        for perm in perm_list:
            ass = np.zeros((n,m))
            for i, j in enumerate(perm):
                ass[i,j] = 1
            assignment_list.append(ass)

        total_benefit = 0
        for j, ass in enumerate(assignment_list):
            total_benefit += (benefit[:,:,j]*ass).sum()

        total_benefit -= calc_assign_seq_handover_penalty(init_ass, assignment_list, lambda_)

        if total_benefit > best_benefit:
            best_benefit = total_benefit
            best_assignment = assignment_list

    return best_benefit, best_assignment

if __name__ == "__main__":
    pass