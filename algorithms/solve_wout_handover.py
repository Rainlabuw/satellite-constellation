import numpy as np
from common.methods import *

def solve_wout_handover(env, verbose=False):
    """
    Solves problem without regard to handover minimization.
    (NHA in the paper.)

    Returns assignment sequence and total value.
    """
    n = env.sat_prox_mat.shape[0]
    m = env.sat_prox_mat.shape[1]
    T = env.sat_prox_mat.shape[2]
    #solve each timestep independently
    assignment_mats = []
    total_val = 0
    done = False
    while not done:
        if verbose: print(f"Solving w/out handover, {env.k}/{T}", end='\r')
        #Solving in most naive fashion, so don't add a handover penalty or anything
        benefits = env.benefit_fn(env.sat_prox_mat[:,:,env.k], None, None, env.benefit_info)
        csol = solve_centralized(benefits)
        assignment_mat = convert_central_sol_to_assignment_mat(n,m,csol)

        _, val, done = env.step(assignment_mat)
        total_val += val
        assignment_mats.append(assignment_mat)

    return assignment_mats, total_val