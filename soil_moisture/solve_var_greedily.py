from soil_moisture.solve_var_w_haal import *

def solve_var_greedily(sat_prox_mat, init_assignment, lambda_, L, 
                                      distributed=False, parallel=False, verbose=False,
                                      eps=0.01, graphs=None, track_iters=False,
                                      benefit_fn=variance_based_benefit_fn, benefit_info=None):
    """
    INPUTS:
        sat_prox_mat: n x m x T array, which contains information about what satellites cover which tasks at what time.
            sat_prox_mat[i,j,k] scales inversely with the covariance of the measurement of task j by satellite i at time k.
        When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
            as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    if graphs is None:
        if distributed:
            graphs = [nx.complete_graph(n) for i in range(T)]
        else:
            graphs = [None]*T

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []

    vars_hist = np.zeros((m,T))
    total_value = 0
    for k in range(T):
        if verbose: print(f"Solving w HAAL, {k}/{T}", end='\r')

        #build benefit mat from task_objects and sat_prox_mat
        tstep_end = min(k+L, T)
        sat_prox_window = np.copy(sat_prox_mat[:,:,k:tstep_end])

        len_window = sat_prox_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(graphs[k]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(sat_prox_window, curr_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=graphs[k], lambda_=lambda_, benefit_fn=benefit_fn,
                                                     benefit_info=benefit_info)
            haal_d_auction.run_auction()
            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, sat_prox_window, 
                                                                      lambda_, parallel_approx=parallel, benefit_fn=benefit_fn,
                                                                      benefit_info=benefit_info)

        #Calculate the value yielded by this assignment
        benefit_hat = benefit_fn(sat_prox_window, curr_assignment, lambda_, benefit_info)
        total_value += (benefit_hat[:,:,0]*chosen_assignment).sum()

        #Update the variance of the task based on the new measurement
        for j in range(m):
            if np.max(chosen_assignment[:,j]) == 1:
                i = np.argmax(chosen_assignment[:,j])
                if sat_prox_mat[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = benefit_info.base_sensor_var / sat_prox_mat[i,j,k]

                if curr_assignment is not None:
                    prev_i = np.argmax(curr_assignment[:,j])
                    if prev_i != i: sensor_var *= lambda_
                
                benefit_info.task_vars[j] = 1/(1/benefit_info.task_vars[j] + 1/sensor_var)

        benefit_info.task_vars += benefit_info.var_add

        curr_assignment = chosen_assignment
        chosen_assignments.append(chosen_assignment)
        vars_hist[:,k] = benefit_info.task_vars

    if not track_iters or not distributed:
        return chosen_assignments, total_value, vars_hist
    else:
        return chosen_assignments, total_value, vars_hist, total_iterations/T