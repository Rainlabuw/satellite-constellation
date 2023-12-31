import numpy as np
from methods import *

def solve_wout_handover(benefit_mats_over_time, init_assignment, lambda_):
    """
    Solves problem without regard to handover minimization.
    (NHA in the paper.)

    Returns assignment sequence, value, and number of handovers induced.
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

    total_value, num_handovers = calc_value_and_num_handovers(assignment_mats, benefit_mats_over_time, init_assignment, lambda_)

    return assignment_mats, total_value, num_handovers