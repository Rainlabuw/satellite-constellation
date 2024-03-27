import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import pickle
from copy import deepcopy

from constellation_sim.ConstellationSim import ConstellationSim, generate_smooth_coverage_hexagons
from constellation_sim.Task import Task
from constellation_sim.Satellite import Satellite
from constellation_sim.TaskObject import TaskObject

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u
import h3
import time

from common.methods import *
from haal.solve_w_haal import choose_time_interval_sequence_centralized, HAAL_D_Parallel_Auction

def get_benefit_matrix_and_graphs_multitask_area(lat_range, lon_range, T, fov=60, isl_dist=2500, dt=30*u.second):
    """
    Generate sat coverage area and graphs for all satellites which can
    see a given area over the course of some 
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    hex_to_task_mapping = {}
    #Generate tasks at the centroid of each hexagon in the area
    hexagons = generate_smooth_coverage_hexagons(lat_range, lon_range, 2)
    for j, hexagon in enumerate(hexagons):
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        lat = polygon.centroid.y
        lon = polygon.centroid.x

        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, Earth)
        task_ben = np.random.uniform(1,2)
        const.add_task(Task(task_loc, task_ben*np.ones(T))) #use benefits which are uniformly 1 to get scaling matrix

        hex_to_task_mapping[hexagon] = j
    print("Num tasks", len(hexagons))

    #add lat and lon range to constellation so we can recover it later
    const.task_lat_range = lat_range
    const.task_lon_range = lon_range

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #10 evenly spaced planes of satellites, each with n/10 satellites per plane
    a = earth.R.to(u.km) + 550*u.km
    ecc = 0*u.one
    inc = 70*u.deg
    argp = 0*u.deg
    ta_offset = 0*u.deg

    num_planes = 18
    num_sats_per_plane = 18

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg + ta_offset
            ta_offset += 1*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #generate satellite coverage matrix with all satellites, even those far away from the area
    full_sat_prox_matrix, graphs = const.propagate_orbits(T, calc_fov_based_proximities)

    #Remove satellites which never cover any tasks in the entire T window
    truncated_sat_prox_matrix = np.zeros_like(full_sat_prox_matrix)
    old_to_new_sat_mapping = {}
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_prox_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_prox_matrix[curr_sat.id,:,:] = full_sat_prox_matrix[i,:,:]

            old_to_new_sat_mapping[i] = curr_sat.id
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    sats_to_track = [deepcopy(sat) for sat in active_sats]
    truncated_sat_prox_matrix = truncated_sat_prox_matrix[:len(const.sats),:,:] #truncate unused satellites

    # #Pick a satellite that starts out of view to track as it traverses the area
    # sat_to_track = None
    # most_timesteps_in_view = 0
    # for sat in const.sats:
    #     if np.sum(truncated_sat_prox_matrix[sat.id,:,0]) == 0 and sat.id != 25 and sat.id != 26:
    #         in_view_timesteps = 0
    #         for k in range(T):
    #             if np.sum(truncated_sat_prox_matrix[sat.id,:,k]) > 0:
    #                 in_view_timesteps += 1
            
    #         if in_view_timesteps > most_timesteps_in_view:
    #             most_timesteps_in_view = in_view_timesteps
    #             sat_to_track = deepcopy(sat)

    # for sat in const.sats:
    #     if sat.id == 66:
    #         sat_to_track = deepcopy(sat)
    
    #update graphs to reflect new satellite numbering after removing useless sats
    for k in range(T):
        nodes_to_remove = [n for n in graphs[k].nodes() if n not in old_to_new_sat_mapping.keys()]
        graphs[k].remove_nodes_from(nodes_to_remove)
        graphs[k] = nx.relabel_nodes(graphs[k], old_to_new_sat_mapping)

    print("\nNum active sats", len(const.sats))

    #Create synthetic satellites to represent each satellite being able to complete multiple tasks.
    #The nth synthetic satellite will recieve (0.9**(n-1))*100% of the benefit for a given task, to incentivize
    #spreading tasks evenly amongst satellites.
    num_tasks_per_sat = 10
    num_real_sats = len(const.sats)
    num_synthetic_sats = num_real_sats*num_tasks_per_sat
    num_original_tasks = len(hexagons)
    
    print(f"Num synthetic sats: {num_synthetic_sats}")
    full_sat_prox_matrix_w_synthetic_sats = np.zeros((num_synthetic_sats, num_original_tasks, T))
    print(f"Full sat cover matrix shape: {full_sat_prox_matrix.shape}, truncated shape: {truncated_sat_prox_matrix.shape}, synthetic shape: {full_sat_prox_matrix_w_synthetic_sats.shape}")

    # #add dummy tasks to sat cover matrix, if necessary
    # base_synthetic_benefit_matrix = np.zeros((num_real_sats, num_tasks_after_synthetic_sats, T))
    # base_synthetic_benefit_matrix[:,:len(hexagons),:] = truncated_sat_prox_matrix
    # print(f"Base synthetic sat cover matrix shape: {base_synthetic_benefit_matrix.shape}")

    for task_num in range(num_tasks_per_sat):
        #Adjust sat cover matrix to reflect the synthetic satellites and tasks
        full_sat_prox_matrix_w_synthetic_sats[task_num*num_real_sats:(task_num+1)*num_real_sats,:num_original_tasks,:] = truncated_sat_prox_matrix*(0.9**task_num)

        #Add appropriate graph connections for the synthetic satellites
        if task_num > 0: #only add for non-original tasks
            for k in range(T):
                grph = graphs[k]
                for real_sat_num in range(num_real_sats):
                    synthetic_sat_num = grph.number_of_nodes()
                    grph.add_node(synthetic_sat_num)
                    grph.add_edge(real_sat_num, synthetic_sat_num)
                    for neigh in grph.neighbors(real_sat_num):
                        grph.add_edge(neigh, synthetic_sat_num)

    n = full_sat_prox_matrix_w_synthetic_sats.shape[0]
    m = full_sat_prox_matrix_w_synthetic_sats.shape[1]
    
    #Create matrix which indicates that synthetic agents representing the same real agent
    A_eqiv = np.zeros((n,n))
    for agent1 in range(n):
        for agent2 in range(n):
            if agent1 % num_real_sats == agent2 % num_real_sats:
                A_eqiv[agent1,agent2] = 1
            else:
                A_eqiv[agent1,agent2] = 0

    #Create scaling matrix for task transitions
    T_trans = np.ones((m,m))
    #no penalty when transitioning between the same task
    for j in range(num_original_tasks):
        T_trans[j,j] = 0

    # #no penalty when transitioning between tasks which are in adjacent hexagons
    # for j in range(num_original_tasks):
    #     task_hex = hexagons[j]
    #     neighbor_hexes = h3.k_ring(task_hex, 1)
    #     for neighbor_hex in neighbor_hexes:
    #         if neighbor_hex in hex_to_task_mapping.keys():
    #             T_trans[j,hex_to_task_mapping[neighbor_hex]] = 0
    #             T_trans[hex_to_task_mapping[neighbor_hex],j] = 0

    with open('multitask_experiment/sat_prox_matrix.pkl','wb') as f:
        pickle.dump(full_sat_prox_matrix_w_synthetic_sats, f)
    with open('multitask_experiment/graphs.pkl','wb') as f:
        pickle.dump(graphs, f)
    with open('multitask_experiment/T_trans.pkl','wb') as f:
        pickle.dump(T_trans, f)
    with open('multitask_experiment/A_eqiv.pkl','wb') as f:
        pickle.dump(A_eqiv, f)
    with open('multitask_experiment/hex_task_map.pkl','wb') as f:
        pickle.dump(hex_to_task_mapping, f)
    with open('multitask_experiment/const_object.pkl','wb') as f:
        pickle.dump(const, f)
    with open('multitask_experiment/sats_to_track.pkl','wb') as f:
        pickle.dump(sats_to_track, f)
    
    return full_sat_prox_matrix_w_synthetic_sats, graphs, T_trans, A_eqiv, \
        hex_to_task_mapping, const, sats_to_track

def calc_multiassign_benefit_fn(benefits, prev_assign, lambda_, benefit_info=None):
    """
    Calculates constant handover penalty, but also ensures that agents are not equivalent.

    Expects A_eqiv and T_trans
    """
    if prev_assign is None: 
        return benefits

    n = benefits.shape[0]
    m = benefits.shape[1]

    try:
        T_trans = benefit_info.T_trans
    except AttributeError:
        print("Info on T_trans not provided, using default")
        T_trans = None
    try:
        A_eqiv = benefit_info.A_eqiv
    except AttributeError:
        print("Info on A_eqiv not provided, using default")
        A_eqiv = None
    if A_eqiv is None:
        A_eqiv = np.eye(n)
    if T_trans is None:
        T_trans = np.ones((m,m)) - np.eye(m)

    state_dep_scaling = np.zeros_like(prev_assign, dtype=float)
    for i in range(n):
        #Find indices where A_eqiv[i,:] is nonzero
        eqiv_agents = np.nonzero(A_eqiv[i,:])[0]
        eqiv_agents_tasks = []
        for eqiv_agent in eqiv_agents:
            for prev_assigned_task in np.nonzero(prev_assign[eqiv_agent,:])[0]:
                eqiv_agents_tasks.append(prev_assigned_task)

        for j in range(m):
            if j not in eqiv_agents_tasks:
                state_dep_scaling[i,j] = 1
            else:
                state_dep_scaling[i,j] = 0

    return benefits-lambda_*state_dep_scaling

def get_filtered_benefits_and_graphs(benefits, graph, curr_assignment, T_trans, A_eqiv):
    """
    Given a benefit matrix and graph, filter out all satellites which are not currently in view of any task,
    and return the filtered benefit matrix and graph.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    L = benefits.shape[2]

    if T_trans is None:
        T_trans = np.ones((m,m)) - np.eye(m)

    if graph is not None:
        filtered_graph = graph.copy()

        for i in range(n):
            if np.sum(benefits[i,:,:]) == 0:
                graph.remove_node(i)
    else: filtered_graph = None

    #Iterate through every satellite - if it has a nonzero benefit at any point,
    #add it to a list of active satellites and create a map between old and new indices,
    #where the new indices are created by removing satellites with zero benefit.
    old_to_new_sat_mapping = {}
    active_sats = 0
    for i in range(n):
        total_sat_benefits = np.sum(benefits[i,:,:])
        if total_sat_benefits > 0: #it has nonzero scaling on at least one task at one timestep
            old_to_new_sat_mapping[i] = active_sats
            active_sats += 1

    new_to_old_sat_mapping = {v: k for k, v in old_to_new_sat_mapping.items()}

    #Update benefit matrix to reflect new satellite numbering
    filtered_benefits = np.zeros((active_sats, m, L))
    for old_sat_idx, new_sat_idx in old_to_new_sat_mapping.items():
        filtered_benefits[new_sat_idx,:,:] = benefits[old_sat_idx,:,:]

    #Add padding to this new benefit matrix to ensure that n <= m
    padded_m = max(active_sats, m)
    filtered_padded_benefits = np.zeros((active_sats, padded_m, L))
    filtered_padded_benefits[:,:m,:] = filtered_benefits

    #Update T_trans to reflect that padded tasks don't induce a transition penalty
    padded_T_trans = np.ones((padded_m, padded_m)) - np.eye(padded_m)
    padded_T_trans[:m,:m] = T_trans
    for j in range(m, padded_m):
        padded_T_trans[:,j] = 0

    #Update A_eqiv to remove old agents
    filtered_A_eqiv = np.zeros((active_sats, active_sats))
    for agent1 in range(n):
        for agent2 in range(n):
            if agent1 in old_to_new_sat_mapping.keys() and agent2 in old_to_new_sat_mapping.keys():
                filtered_A_eqiv[old_to_new_sat_mapping[agent1],old_to_new_sat_mapping[agent2]] = A_eqiv[agent1,agent2]

    #update graphs to reflect new satellite numbering
    if graph is not None:
        nodes_to_remove = [n for n in filtered_graph.nodes() if n not in old_to_new_sat_mapping.keys()]
        filtered_graph.remove_nodes_from(nodes_to_remove)
        filtered_graph = nx.relabel_nodes(filtered_graph, old_to_new_sat_mapping)

    #update current assignment to reflect new satellite numbering
    if curr_assignment is not None:
        new_assignment = np.zeros((active_sats, padded_m))
        for i in range(n):
            if i in old_to_new_sat_mapping.keys():
                new_assignment[old_to_new_sat_mapping[i],:m] = curr_assignment[i,:]
    else:
        new_assignment = None

    return filtered_padded_benefits, filtered_graph, filtered_A_eqiv, padded_T_trans, new_assignment, new_to_old_sat_mapping

def solve_multitask_w_haal(benefits, init_assignment, lambda_, L, distributed=False, parallel=None, verbose=False,
                 eps=0.01, graphs=None, track_iters=False, benefit_info=None):
    """
    Sequentially solves the problem using the HAAL algorithm.

    When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")
    
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    if graphs is None:
        if distributed:
            graphs = [nx.complete_graph(n) for i in range(T)]
        else: graphs = [None]*T

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []
    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        filtered_benefit_mat_window, filtered_graph, filtered_A_eqiv, filtered_T_trans, \
            filtered_assignment, new_to_old_sat_mapping = get_filtered_benefits_and_graphs(benefit_mat_window, graphs[curr_tstep], curr_assignment, 
                                                                                           benefit_info.T_trans, benefit_info.A_eqiv)
        filtered_benefit_info = ExtraHandoverPenInfo()
        filtered_benefit_info.T_trans = filtered_T_trans
        filtered_benefit_info.A_eqiv = filtered_A_eqiv

        if distributed:
            if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(filtered_benefit_mat_window, filtered_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=filtered_graph, lambda_=lambda_, 
                                                     benefit_fn=calc_multiassign_benefit_fn, benefit_info=filtered_benefit_info)
            haal_d_auction.run_auction()
            filtered_chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            filtered_chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, filtered_assignment, filtered_benefit_mat_window, 
                                                                      lambda_, parallel_approx=parallel, benefit_fn=calc_multiassign_benefit_fn,
                                                                      benefit_info=filtered_benefit_info)

        #Map filtered assignments back into the satellite indices of the original benefit matrix.
        chosen_assignment = np.zeros((n,m))
        for i in range(n):
            if i in new_to_old_sat_mapping.keys():
                chosen_assignment[new_to_old_sat_mapping[i],:] = filtered_chosen_assignment[i,:m]
        
        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value = calc_assign_seq_state_dependent_value(init_assignment, chosen_assignments, benefits, lambda_, 
                                                        benefit_fn=calc_multiassign_benefit_fn, benefit_info=benefit_info)
    
    if not track_iters or not distributed:
        return chosen_assignments, total_value
    else:
        return chosen_assignments, total_value, total_iterations/T

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)
    T = 10
    full_sat_prox_matrix_w_synthetic_sats, graphs, T_trans, A_eqiv, \
        hex_to_task_mapping, const, sat_to_track = get_benefit_matrix_and_graphs_multitask_area(lat_range, lon_range, T)

    # with open('multitask_experiment/benefit_matrix.pkl','rb') as f:
    #     full_sat_prox_matrix_w_synthetic_sats = pickle.load(f)
    # with open('multitask_experiment/graphs.pkl','rb') as f:
    #     graphs = pickle.load(f)
    # with open('multitask_experiment/hex_task_map.pkl','rb') as f:
    #     hex_to_task_mapping = pickle.load(f)
    # with open('multitask_experiment/const_object.pkl','rb') as f:
    #     const = pickle.load(f)
    # with open('multitask_experiment/T_trans.pkl','rb') as f:
    #     T_trans = pickle.load(f)
    # with open('multitask_experiment/A_eqiv.pkl','rb') as f:
    #     A_eqiv = pickle.load(f)
    
    benefit_info = ExtraHandoverPenInfo()
    benefit_info.T_trans = T_trans
    benefit_info.A_eqiv = A_eqiv
    
    ass, tv = solve_multitask_w_haal(full_sat_prox_matrix_w_synthetic_sats, None, 0.5, 3, distributed=False, verbose=True, benefit_info=benefit_info)
    # print(tv)