import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import pickle

from constellation_sim.ConstellationSim import ConstellationSim, generate_smooth_coverage_hexagons
from constellation_sim.Task import Task
from constellation_sim.Satellite import Satellite
from constellation_sim.TaskObject import TaskObject

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u
import h3

from methods import *
from solve_w_haal import solve_w_haal, choose_time_interval_sequence_centralized, HAAL_D_Parallel_Auction

def init_task_objects(num_objects, const, hex_to_task_mapping, T, velocity=6437*u.km/u.hr):
    """
    Generate randomly initialized objects moving through the regions,
    and propagate them until T.

    Track history of their associated tasks over this timeframe.

    Default velocity is approx hypersonic speed, 4000 mi/hr
    """
    #Initialize random objects
    task_objects = []
    for _ in range(num_objects):
        #choose random direction of travel (east, west, north, or south)
        dir = np.random.choice(["E","W","N","S"])
        dir = {"E":(1,0), "W":(-1,0), "N":(0,1), "S":(0,-1)}[dir]

        object_appear_time = np.random.randint(T)
        start_lat = np.random.uniform(const.task_lat_range[0], const.task_lat_range[1])
        start_lon = np.random.uniform(const.task_lon_range[0], const.task_lon_range[1])

        task_objects.append(TaskObject(start_lat, start_lon, const.task_lat_range, const.task_lon_range, dir, object_appear_time, const.dt, speed=velocity))

    for task_object in task_objects:
        task_object.propagate(hex_to_task_mapping, T)

    return task_objects

def get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_cover_matrix, task_objects):
    """
    Given list of task_objects, return a benefit matrix which encodes
    the benefit of each task at each timestep.
    """
    n = sat_cover_matrix.shape[0]
    m = sat_cover_matrix.shape[1]
    T = sat_cover_matrix.shape[2]

    benefits = np.ones_like(sat_cover_matrix) * coverage_benefit
    for k in range(T):
        for task_object in task_objects:
            #If the object is active at this timestep, add its benefits for the next L timesteps
            if k >= task_object.appear_time:
                if task_object.task_idxs[k] is not None:
                    benefits[:,task_object.task_idxs[k],k] += object_benefit

    benefits = benefits * sat_cover_matrix #scale benefits by sat coverage scaling

    return benefits

def add_tasks_with_objects(num_objects, lat_range, lon_range, dt, T):
    """
    Generate tasks and associated benefits, with randomly initialized
    objects moving through the regions.

    #TODO: more robust movement model
    """
    #Initialize random objects
    velocity = 6437*u.km/u.hr #approx hypersonic speed, 4000 mi/hr
    num_objects_in_hex_by_time = [defaultdict(int) for k in range(T)]
    for _ in range(num_objects):
        #choose random direction of travel (east, west, north, or south)
        dir = np.random.choice(["E","W","N","S"])
        dir = {"E":(1,0), "W":(-1,0), "N":(0,1), "S":(0,-1)}[dir]

        object_appear_time = np.random.randint(T)
        start_lat = np.random.uniform(lat_range[0], lat_range[1])
        start_lon = np.random.uniform(lon_range[0], lon_range[1])

        #If going east or west
        if dir == (1,0) or dir == (-1,0):
            rad_at_lat = Earth.R.to(u.km)*np.cos(start_lat*np.pi/180)
            ang_vel = velocity/rad_at_lat #rad/hr
        else: 
            ang_vel = velocity/Earth.R.to(u.km)
        deg_change_per_ts = (ang_vel*const.dt*180/np.pi).to(u.one)

        #Propagate object movement over time
        curr_lat = start_lat
        curr_lon = start_lon
        for k in range(object_appear_time,T):
            # Find the hexagon containing this lat/lon, increment target count
            hexagon = h3.geo_to_h3(curr_lat, curr_lon, 1)
            num_objects_in_hex_by_time[k][hexagon] += 1

            curr_lat += deg_change_per_ts * dir[1]
            curr_lon += deg_change_per_ts * dir[0]

    # Initialize an empty set to store unique H3 indexes
    hexagons = set()

    # Step through the defined ranges and discretize the globe
    lat_steps, lon_steps = 0.5, 0.5
    lat = lat_range[0]
    while lat <= lat_range[1]:
        lon = lon_range[0]
        while lon <= lon_range[1]:
            # Find the hexagon containing this lat/lon
            hexagon = h3.geo_to_h3(lat, lon, 1)
            hexagons.add(hexagon)
            lon += lon_steps
        lat += lat_steps

    #Add tasks at centroid of all hexagons
    for hexagon in hexagons:
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        task_loc = SpheroidLocation(polygon.centroid.y*u.deg, polygon.centroid.x*u.deg, 0*u.m, Earth)
        
        task_benefit = np.zeros(T)
        for k in range(T):
            task_benefit[k] += 1 + 10*num_objects_in_hex_by_time[k][hexagon]

        const.add_task(Task(task_loc, task_benefit))

def get_sat_coverage_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T, fov=60, isl_dist=2500, dt=30*u.second):
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
        const.add_task(Task(task_loc, np.ones(T))) #use benefits which are uniformly 1 to get scaling matrix

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

    num_planes = 40
    num_sats_per_plane = 25

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg + ta_offset
            ta_offset += 1*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #generate satellite coverage matrix with all satellites, even those far away from the area
    full_sat_cover_matrix, graphs = const.propagate_orbits(T, calc_fov_benefits)

    truncated_sat_cover_matrix = np.zeros_like(full_sat_cover_matrix)
    old_to_new_sat_mapping = {}
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_cover_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_cover_matrix[curr_sat.id,:,:] = full_sat_cover_matrix[i,:,:]

            old_to_new_sat_mapping[i] = curr_sat.id
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    truncated_sat_cover_matrix = truncated_sat_cover_matrix[:len(const.sats),:,:] #truncate unused satellites

    #update graphs to reflect new satellite numbering
    for k in range(T):
        nodes_to_remove = [n for n in graphs[k].nodes() if n not in old_to_new_sat_mapping.keys()]
        graphs[k].remove_nodes_from(nodes_to_remove)
        graphs[k] = nx.relabel_nodes(graphs[k], old_to_new_sat_mapping)

    print("Num active sats", len(const.sats))

    #Create second opportunity for tasks to be completed, with 20% of the scaling
    num_primary_tasks = truncated_sat_cover_matrix.shape[1]
    sat_cover_matrix = np.zeros((truncated_sat_cover_matrix.shape[0], truncated_sat_cover_matrix.shape[1]*2, truncated_sat_cover_matrix.shape[2]))
    sat_cover_matrix[:,:truncated_sat_cover_matrix.shape[1],:] = truncated_sat_cover_matrix
    sat_cover_matrix[:,truncated_sat_cover_matrix.shape[1]:,:] = truncated_sat_cover_matrix * 0.2

    m = sat_cover_matrix.shape[1]
    
    #Create scaling matrix for task transitions
    task_transition_state_dep_scaling_mat = np.ones((m,m))
    
    #no penalty when transitioning between backup and primary versions of the same task
    for j in range(num_primary_tasks):
        task_transition_state_dep_scaling_mat[j,j+num_primary_tasks] = 0
        task_transition_state_dep_scaling_mat[j+num_primary_tasks,j] = 0

    #no penalty when transitioning between the same task
    for j in range(m):
        task_transition_state_dep_scaling_mat[j,j] = 0

    #no penalty when transitioning between tasks which are in adjacent hexagons
    # for j in range(num_primary_tasks):
    #     task_hex = hexagons[j]
    #     neighbor_hexes = h3.k_ring(task_hex, 1)
    #     for neighbor_hex in neighbor_hexes:
    #         if neighbor_hex in hex_to_task_mapping.keys():
    #             task_transition_state_dep_scaling_mat[j,hex_to_task_mapping[neighbor_hex]] = 0
    #             task_transition_state_dep_scaling_mat[hex_to_task_mapping[neighbor_hex],j] = 0

    #             task_transition_state_dep_scaling_mat[j+num_primary_tasks,hex_to_task_mapping[neighbor_hex]] = 0
    #             task_transition_state_dep_scaling_mat[hex_to_task_mapping[neighbor_hex],j+num_primary_tasks] = 0

    with open('object_track_experiment/sat_cover_matrix.pkl','wb') as f:
        pickle.dump(sat_cover_matrix, f)
    with open('object_track_experiment/graphs.pkl','wb') as f:
        pickle.dump(graphs, f)
    with open('object_track_experiment/task_transition_scaling.pkl','wb') as f:
        pickle.dump(task_transition_state_dep_scaling_mat, f)
    with open('object_track_experiment/hex_task_map.pkl','wb') as f:
        pickle.dump(hex_to_task_mapping, f)
    with open('object_track_experiment/const_object.pkl','wb') as f:
        pickle.dump(const, f)
    
    return sat_cover_matrix, graphs, task_transition_state_dep_scaling_mat, hex_to_task_mapping, const

def get_filtered_benefits_and_graphs(benefits, graph, curr_assignment, T_trans=None):
    """
    Given a benefit matrix and graph, filter out all satellites which are not currently in view of any task,
    and return the filtered benefit matrix and graph.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    L = benefits.shape[2]

    if T_trans is None:
        T_trans = np.ones((m,m)) - np.eye(m)

    filtered_graph = graph.copy()

    for i in range(n):
        if np.sum(benefits[i,:,:]) == 0:
            graph.remove_node(i)

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

    #update graphs to reflect new satellite numbering
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

    return filtered_padded_benefits, filtered_graph, padded_T_trans, new_assignment, new_to_old_sat_mapping

def timestep_loss_state_dep_fn(benefits, prev_assign, lambda_, task_trans_state_dep_scaling_mat=None):
    """
    Adds a loss to the benefit matrix which encodes a switching cost of
    losing the entire benefit of the task in the first timestep, plus a small
    extra penalty (lambda_).

    Expects a 2D (n x m) benefit matrix.
    """
    if prev_assign is None: return benefits

    m = benefits.shape[1]
    if task_trans_state_dep_scaling_mat is None:
        task_trans_state_dep_scaling_mat = np.ones((m,m)) - np.eye(m)
    state_dep_scaling = prev_assign @ task_trans_state_dep_scaling_mat

    return np.where(state_dep_scaling > 0, benefits*(1-state_dep_scaling)-lambda_, benefits)

def solve_object_track_w_dynamic_haal(sat_coverage_matrix, task_objects, coverage_benefit, object_benefit, init_assignment, lambda_, L, 
                                      distributed=False, parallel=False, verbose=False,
                                      eps=0.01, graphs=None, track_iters=False,
                                      state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=None):
    """
    Aim to solve the problem using the HAAL algorithm, but with the benefit matrix
    changing over time as objects move through the area. We can't precalculate this
    because we don't want the algorithm to have prior knowledge of the object movement
    until it actually appears.

    INPUTS:
        sat_coverage_matrix: n x m x T array, which contains information about what satellites cover which tasks at what time.
            sat_coverage_matrix[i,j,k] = the ratio of the benefits from task j that satellite i can collect at time k (i.e. 0 if sat can't see task)
        When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
            as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")

    n = sat_coverage_matrix.shape[0]
    m = sat_coverage_matrix.shape[1]
    T = sat_coverage_matrix.shape[2]

    if graphs is None and distributed:
        graphs = [nx.complete_graph(n) for i in range(T)]

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []
    for k in range(T):
        if verbose: print(f"Solving w HAAL, {k}/{T}", end='\r')

        #build benefit matrix from task_objects and sat_cover_matrix
        tstep_end = min(k+L, T)
        sat_coverage_window = sat_coverage_matrix[:,:,k:tstep_end]
        benefit_window = np.ones_like(sat_coverage_window) * coverage_benefit
        for task_object in task_objects:
            #If the object is active at this timestep, add it's benefits for the next L timesteps
            if k >= task_object.appear_time:
                for t in range(k, tstep_end):
                    if task_object.task_idxs[t] is not None:
                        benefit_window[:,task_object.task_idxs[t],t-k] += object_benefit

        benefit_window = benefit_window * sat_coverage_window #scale benefits by sat coverage scaling

        #We don't want to run auctions with all satellites in the constellation if they can't see any current tasks.
        #Thus, we filter the benefit matrices so that we only run auctions with the satellites which can see tasks.
        filtered_benefit_window, filtered_graph, filtered_T_trans, \
            filtered_assignment, new_to_old_sat_mapping = get_filtered_benefits_and_graphs(benefit_window, graphs[k], curr_assignment, task_trans_state_dep_scaling_mat)

        len_window = filtered_benefit_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(filtered_graph): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(filtered_benefit_window, filtered_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=filtered_graph, lambda_=lambda_, state_dep_fn=state_dep_fn,
                                                     task_trans_state_dep_scaling_mat=filtered_T_trans)
            haal_d_auction.run_auction()
            filtered_chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            filtered_chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, filtered_assignment, filtered_benefit_window, 
                                                                      lambda_, parallel_approx=parallel, state_dep_fn=state_dep_fn,
                                                                      task_trans_state_dep_scaling_mat=filtered_T_trans)

        #Map filtered assignments back into the satellite indices of the original benefit matrix.
        chosen_assignment = np.zeros((n,m))
        for i in range(n):
            if i in new_to_old_sat_mapping.keys():
                chosen_assignment[new_to_old_sat_mapping[i],:] = filtered_chosen_assignment[i,:m]

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_coverage_matrix, task_objects)
    total_value = calc_assign_seq_state_dependent_value(init_assignment, chosen_assignments, total_benefits, lambda_, 
                                                        state_dep_fn=state_dep_fn, T_trans=task_trans_state_dep_scaling_mat)
    
    if not track_iters or not distributed:
        return chosen_assignments, total_value
    else:
        return chosen_assignments, total_value, total_iterations/T

if __name__ == "__main__":
    with open('object_track_experiment/sat_cover_matrix.pkl','rb') as f:
        sat_cover_matrix = pickle.load(f)
    with open('object_track_experiment/graphs.pkl','rb') as f:
        graphs = pickle.load(f)
    with open('object_track_experiment/task_transition_scaling.pkl','rb') as f:
        T_trans = pickle.load(f)
    with open('object_track_experiment/hex_task_map.pkl','rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('object_track_experiment/const_object.pkl','rb') as f:
        const = pickle.load(f)

    lat_range = (20, 50)
    lon_range = (73, 135)
    T = 60

    # get_sat_coverage_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T)
    np.random.seed(0)
    task_objects = init_task_objects(60, const, hex_to_task_mapping, T)
    benefits = get_benefits_from_task_objects(1, 10, sat_cover_matrix, task_objects)

    ass, tv = solve_object_track_w_dynamic_haal(sat_cover_matrix, task_objects, 1, 10, None, 0.5, 3, parallel=True, distributed=False, graphs=graphs,
                                                state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=T_trans,
                                                track_iters=True)
    print(tv)
    print(is_assignment_mat_sequence_valid(ass))