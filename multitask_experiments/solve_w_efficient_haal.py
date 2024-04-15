import numpy as np

from common.methods import *
from algorithms.solve_w_haal import HAAL_D_Parallel_Auction, choose_time_interval_sequence_centralized

def solve_w_efficient_haal(env, L, distributed=False, parallel=None, verbose=False, 
                     eps=0.01, graphs=None, track_iters=False):
    """
    Sequentially solves the problem given by the environment using the HAAL algorithm.

    When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")
    
    n = env.sat_prox_mat.shape[0]
    m = env.sat_prox_mat.shape[1]
    T = env.sat_prox_mat.shape[2]

    if graphs is None and distributed:
        graphs = [nx.complete_graph(n) for i in range(T)]
    
    total_iterations = 0 if distributed else None
    total_value = 0
    chosen_assignments = []
    done = False
    while not done:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = env.k
        tstep_end = min(curr_tstep+L, T)
        prox_mat_window = env.sat_prox_mat[:,:,curr_tstep:tstep_end]

        len_window = prox_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(prox_mat_window, env.curr_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=graphs[curr_tstep], lambda_=env.lambda_, 
                                                     benefit_fn=env.benefit_fn, benefit_info=env.benefit_info)
            haal_d_auction.run_auction()
            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, env.curr_assignment, prox_mat_window, 
                                                                      env.lambda_, env.benefit_fn, parallel_approx=parallel,
                                                                      benefit_info=env.benefit_info)

        chosen_assignments.append(chosen_assignment)
        
        _, value, done = env.step(chosen_assignment)
        total_value += value
    
    if not track_iters or not distributed:
        return chosen_assignments, total_value
    else:
        return chosen_assignments, total_value, total_iterations/T