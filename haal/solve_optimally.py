import numpy as np
from common.methods import *
import itertools

def gen_perms_of_perms(curr_perm_list, n, T):
    global total_perm_list

    if len(curr_perm_list) == T:
        total_perm_list.append(curr_perm_list)
        return
    else:
        for perm in itertools.permutations(range(n)):
            gen_perms_of_perms(curr_perm_list + [perm], n, T)

def solve_optimally(benefits, init_ass, lambda_, 
                    state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None):
    """
    Given a benefit matrix and an initial assignment,
    exhaustively search for and find the optimal value
    assignment matrix.

    Only works for extremely small problems due to the exhaustive search.
    (i.e. 5x5x3)

    Returns best assignments and best value
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

        total_value = calc_assign_seq_state_dependent_value(init_ass, assignment_list, benefits, lambda_, state_dep_fn=state_dep_fn,
                                                            extra_handover_info=extra_handover_info)

        if total_value > best_value:
            best_value = total_value
            best_assignments = assignment_list

    return best_assignments, best_value

if __name__ == "__main__":
    pass