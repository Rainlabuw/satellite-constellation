import numpy as np
from methods import *
import networkx as nx
import time

def solve_w_haal(benefits, init_assignment, lambda_, L, parallel_approx=False, verbose=False, 
                 state_dep_fn=generic_handover_state_dep_fn, task_trans_state_dep_scaling_mat=None):
    """
    Sequentially solves the problem using the HAAL algorithm.

    When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as is done to parallelize auctions in the distributed version of the algorithm.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    curr_assignment = init_assignment
    
    chosen_assignments = []

    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, benefit_mat_window, 
                                                                      lambda_, parallel_approx=parallel_approx, state_dep_fn=state_dep_fn,
                                                                      task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value = calc_assign_seq_state_dependent_value(init_assignment, chosen_assignments, benefits, lambda_, 
                                                        state_dep_fn=state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    
    return chosen_assignments, total_value

def choose_time_interval_sequence_centralized(time_interval_sequences, prev_assignment, benefit_mat_window, lambda_, parallel_approx=False, 
                                              state_dep_fn=generic_handover_state_dep_fn, task_trans_state_dep_scaling_mat=None):
    """
    Chooses the best time interval sequence from a list of time interval sequences,
    and return the corresponding assignment.
    """
    n = benefit_mat_window.shape[0]
    m = benefit_mat_window.shape[1]

    best_value = -np.inf
    best_assignment = None
    best_time_interval = None

    for time_interval_sequence in time_interval_sequences:
        total_tis_value = 0
        tis_assignment_curr = prev_assignment
        tis_first_assignment = None

        for i, time_interval in enumerate(time_interval_sequence):
            #Grab benefit matrices from this time interval
            combined_benefit_mat = benefit_mat_window[:,:,time_interval[0]:time_interval[1]+1]
            if combined_benefit_mat.ndim == 2: #make sure combined_benefit_mat is 3D
                combined_benefit_mat = np.expand_dims(combined_benefit_mat, axis=2) 

            #Adjust the benefit matrix to incentivize agents being assigned the same task twice.
            #Note that if we're not approximating, we incentivize staying close to the previous assignment calculated during this
            #time interval sequence, not the actual assignment that the agents currently have (i.e. prev_assignment)
            benefit_hat = np.copy(combined_benefit_mat)
            if not parallel_approx: benefit_hat[:,:,0] = state_dep_fn(benefit_hat[:,:,0], tis_assignment_curr, lambda_, task_trans_state_dep_scaling_mat)
            else: benefit_hat[:,:,0] = state_dep_fn(benefit_hat[:,:,0], prev_assignment, lambda_, task_trans_state_dep_scaling_mat)
            benefit_hat = benefit_hat.sum(axis=-1)

            #Generate an assignment using a centralized solution.
            central_assignments = solve_centralized(benefit_hat)
            tis_assignment = convert_central_sol_to_assignment_mat(n, m, central_assignments)

            #Calculate the value from this time interval sequence
            total_tis_value += (benefit_hat*tis_assignment).sum()

            tis_assignment_curr = tis_assignment
            #If this is the first assignment in the time interval sequence, save it
            if i == 0:
                tis_first_assignment = tis_assignment_curr

        #If this time interval sequence is better than the best one we've seen so far, save it 
        if total_tis_value > best_value:
            best_value = total_tis_value
            best_assignment = tis_first_assignment
            best_time_interval = time_interval_sequence[0]

    return best_assignment

if __name__ == "__main__":
    np.random.seed(48)
    benefits = 2*np.random.random((50, 50, 10))
    s = time.time()
    chosen_assignments, val, _ = solve_w_haal(benefits, None, 0.5, 4, distributed=True, verbose=False)
    print(val,time.time()-s)
    # print(calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, 1))