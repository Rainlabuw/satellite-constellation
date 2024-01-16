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
from solve_w_haal import solve_w_haal, choose_time_interval_sequence_centralized

def solve_object_track_w_dynamic_haal(const, hex_to_task_mapping, sat_coverage_matrix, init_assignment, lambda_, L, parallel_approx=False, verbose=False, 
                 state_dep_fn=generic_handover_state_dep_fn, task_trans_state_dep_scaling_mat=None):
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
    n = sat_coverage_matrix.shape[0]
    m = sat_coverage_matrix.shape[1]
    T = sat_coverage_matrix.shape[2]

    task_objects = init_task_objects(100, const, T)

    curr_assignment = init_assignment
    
    chosen_assignments = []

    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = sat_coverage_matrix[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, benefit_mat_window, 
                                                                      lambda_, parallel_approx=parallel_approx, state_dep_fn=state_dep_fn,
                                                                      task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value = calc_assign_seq_state_dependent_value(init_assignment, chosen_assignments, sat_coverage_matrix, lambda_, 
                                                        state_dep_fn=state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    
    return chosen_assignments, total_value

def init_task_objects(num_objects, const, T):
    """
    Generate tasks and associated benefits, with randomly initialized
    objects moving through the regions.
    """
    #Initialize random objects
    velocity = 6437*u.km/u.hr #approx hypersonic speed, 4000 mi/hr
    task_objects = []
    for _ in range(num_objects):
        #choose random direction of travel (east, west, north, or south)
        dir = np.random.choice(["E","W","N","S"])
        dir = {"E":(1,0), "W":(-1,0), "N":(0,1), "S":(0,-1)}[dir]

        object_appear_time = np.random.randint(T)
        start_lat = np.random.uniform(const.task_lat_range[0], const.task_lat_range[1])
        start_lon = np.random.uniform(const.task_lon_range[0], const.task_lon_range[1])

        task_objects.append(TaskObject(start_lat, start_lon, const.task_lat_range, const.task_lon_range, dir, object_appear_time, const.dt, speed=velocity))

    return task_objects

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

def get_sat_coverage_matrix_and_graphs_object_tracking_area(lat_range, lon_range, fov=60):
    """
    Generate constellation benefits based on tracking ojects
    """
    T = 60 #30 minutes
    const = ConstellationSim(dt = 30*u.second, isl_dist=2500)
    earth = Earth

    hex_to_task_mapping = {}
    #Generate tasks at the centroid of each hexagon in the area
    hexagons = generate_smooth_coverage_hexagons(lat_range, lon_range)
    for j, hexagon in enumerate(hexagons):
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        lat = polygon.centroid.y
        lon = polygon.centroid.x

        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, Earth)
        const.add_task(Task(task_loc, np.ones(T))) #use benefits which are uniformly 1 to get scaling matrix

        hex_to_task_mapping[hexagon] = j
    
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

    #generate satellit coverage matrix with all satellites, even those far away from the area
    full_sat_cover_matrix, graphs = const.propagate_orbits(T, calc_fov_benefits)

    truncated_sat_cover_matrix = np.zeros_like(full_sat_cover_matrix)
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_cover_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_cover_matrix[curr_sat.id,:,:] = full_sat_cover_matrix[i,:,:]
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    truncated_sat_cover_matrix = truncated_sat_cover_matrix[:len(const.sats),:,:] #truncate unused satellites

    #Create second opportunity for tasks to be completed, with 20% of the scaling
    sat_cover_matrix_w_backup_tasks = np.zeros((truncated_sat_cover_matrix.shape[0], truncated_sat_cover_matrix.shape[1]*2, truncated_sat_cover_matrix.shape[2]))
    sat_cover_matrix_w_backup_tasks[:,:truncated_sat_cover_matrix.shape[1],:] = truncated_sat_cover_matrix
    sat_cover_matrix_w_backup_tasks[:,truncated_sat_cover_matrix.shape[1]:,:] = truncated_sat_cover_matrix * 0.2
    non_padded_m = sat_cover_matrix_w_backup_tasks.shape[1]

    #if necessary, pad the sat cover matrix with zeros so that n<=m
    padding_size = max(0,sat_cover_matrix_w_backup_tasks.shape[0]-sat_cover_matrix_w_backup_tasks.shape[1])
    sat_cover_matrix = np.pad(sat_cover_matrix_w_backup_tasks, ((0,0), (0, padding_size), (0,0)))
    m = sat_cover_matrix.shape[1]

    #Create scaling matrix for task transitions
    task_transition_state_dep_scaling_mat = np.ones((m,m))
    
    #no penalty when transitioning between backup and primary versions of the same task
    for j in range(non_padded_m//2):
        task_transition_state_dep_scaling_mat[j,j+non_padded_m//2] = 0
        task_transition_state_dep_scaling_mat[j+non_padded_m//2,j] = 0

    #no penalty when transitioning to a dummy task
    for j in range(non_padded_m, m):
        task_transition_state_dep_scaling_mat[:,j] = 0

    with open('object_track_experiment/sat_cover_matrix_large_const.pkl','wb') as f:
        pickle.dump(sat_cover_matrix, f)
    with open('object_track_experiment/graphs_large_const.pkl','wb') as f:
        pickle.dump(graphs, f)
    with open('object_track_experiment/task_transition_scaling_large_const.pkl','wb') as f:
        pickle.dump(task_transition_state_dep_scaling_mat, f)
    with open('object_track_experiment/hex_task_map_large_const.pkl','wb') as f:
        pickle.dump(hex_to_task_mapping, f)
    with open('object_track_experiment/const_object_large_const.pkl','wb') as f:
        pickle.dump(const, f)

def timestep_loss_state_dep_fn(benefits, prev_assign, lambda_, task_trans_state_dep_scaling_mat=None):
    """
    Adds a loss to the benefit matrix which encodes a switching cost of
    losing the entire benefit of the task in the first timestep, plus a small
    extra penalty (lambda_).
    """
    if prev_assign is None: return benefits

    m = benefits.shape[1]
    if task_trans_state_dep_scaling_mat is None:
        task_trans_state_dep_scaling_mat = np.ones((m,m))
    state_dep_scaling = prev_assign @ task_trans_state_dep_scaling_mat

    return np.where((prev_assign == 0) & (state_dep_scaling > 0), -lambda_*state_dep_scaling, benefits)

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)

    get_constellation_bens_and_graphs_object_tracking_area(lat_range, lon_range)