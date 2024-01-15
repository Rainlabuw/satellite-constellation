import numpy as np
from methods import *

def solve_wout_handover(benefit_mats_over_time, init_assignment, lambda_, state_dep_fn=generic_handover_state_dep_fn,
                        task_trans_state_dep_scaling_mat=None):
    """
    Solves problem without regard to handover minimization.
    (NHA in the paper.)

    Returns assignment sequence and total value.
    """
    n = benefit_mats_over_time.shape[0]
    m = benefit_mats_over_time.shape[1]
    T = benefit_mats_over_time.shape[2]
    #solve each timestep independently
    assignment_mats = []
    for k in range(T):
        csol = solve_centralized(benefit_mats_over_time[:,:,k])
        assignment_mat = convert_central_sol_to_assignment_mat(n,m,csol)
        
        assignment_mats.append(assignment_mat)

    total_value = calc_assign_seq_state_dependent_value(init_assignment, assignment_mats, benefit_mats_over_time, lambda_, state_dep_fn=state_dep_fn,
                                                        task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)

    return assignment_mats, total_value