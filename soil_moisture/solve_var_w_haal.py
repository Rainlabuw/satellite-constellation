import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
import h3
from shapely.geometry import Polygon
import pickle

from constellation_sim.ConstellationSim import get_constellation_proxs_and_graphs_coverage
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from haal.solve_w_haal import HAAL_D_Parallel_Auction, choose_time_interval_sequence_centralized
from common.methods import *

def variance_based_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Create a benefit mat based on the previous assignment and the variances of each area.
    As a handover penalty, the sensor variance of the first measurement of a given task is much higher.

    INPUTS:
     - a 3D (n x m x T) mat of the satellite coverage mat, as well as previous assignments.
     - lambda_ is in this case the amount the sensor covariance is multiplied for the first measurement.
     - benefit_info should be a structure containing the following info:
        - task_vars is the m variances of the tasks at the current state.
        - base_sensor_var is the baseline sensor variance
        - var_add is how much variance is added at each time step.
    """
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    benefit_mat = np.zeros_like(sat_prox_mat)

    #Transform curr_var (m,) to (n,m) by repeating the value across the agent axis (axis 0).
    #This array will track the variance of each task j, assuming that it was done by satellite i.
    agent_task_vars = np.tile(benefit_info.task_vars, (n, 1))
    for k in range(T):
        for i in range(n):
            for j in range(m):
                #Calculate the sensor variance for sat i measuring task j at time k
                if sat_prox_mat[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = benefit_info.base_sensor_var / sat_prox_mat[i,j,k]

                #if the task was not previously assigned, multiply the sensor variance by lambda_ (>1)
                if prev_assign is not None and prev_assign[i,j] != 1 and k == 0:
                    sensor_var *= lambda_

                #Calculate the new variance for the task after taking the observation.
                #The reduction in variance is the benefit for the agent-task pair.
                new_agent_task_var = 1/(1/agent_task_vars[i,j] + 1/sensor_var)
                benefit_mat[i,j,k] = agent_task_vars[i,j] - new_agent_task_var

                #Update the variance of the task based on the new measurement
                agent_task_vars[i,j] = new_agent_task_var

        #Add variance to all tasks
        agent_task_vars += benefit_info.var_add

    if init_dim == 2: benefit_mat = np.squeeze(benefit_mat, axis=2)
    return benefit_mat

def solve_var_w_dynamic_haal(sat_prox_mat, init_assignment, lambda_, L, 
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

if __name__ == "__main__":
    pass
