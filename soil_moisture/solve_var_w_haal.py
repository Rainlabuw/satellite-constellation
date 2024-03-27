import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
import h3
from shapely.geometry import Polygon
import pickle

from constellation_sim.ConstellationSim import ConstellationSim, generate_smooth_coverage_hexagons
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from haal.solve_w_haal import HAAL_D_Parallel_Auction, choose_time_interval_sequence_centralized
from common.methods import *

def generate_benefits_from_satcover_and_var(sat_cover_matrix, prev_assign, lambda_,
                                            task_vars, base_sensor_var, var_add):
    """
    Create a benefit matrix based on the previous assignment and the variances of each area.
    As a handover penalty, the sensor variance of the first measurement of a given task is much higher.

    INPUTS:
     - a 3D (n x m x T) matrix of the satellite coverage matrix, as well as previous assignments.
     - lambda_ is in this case the amount the sensor covariance is multiplied for the first measurement.
     - task_vars is the m variances of the tasks at the current state.
     - base_sensor_var is the baseline sensor variance
     - var_add is how much variance is added at each time step.
    """
    n = sat_cover_matrix.shape[0]
    m = sat_cover_matrix.shape[1]
    T = sat_cover_matrix.shape[2]

    benefit_matrix = np.zeros_like(sat_cover_matrix)

    #Transform curr_var (m,) to (n,m) by repeating the value across the agent axis (axis 0).
    #This array will track the variance of each task j, assuming that it was done by satellite i.
    agent_task_vars = np.tile(task_vars, (n, 1))
    for k in range(T):
        for i in range(n):
            for j in range(m):
                #Calculate the sensor variance for sat i measuring task j at time k
                if sat_cover_matrix[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = base_sensor_var / sat_cover_matrix[i,j,k]

                #if the task was not previously assigned, multiply the sensor variance by lambda_ (>1)
                if prev_assign[i,j] != 1 and k == 0:
                    sensor_var *= lambda_

                #Calculate the new variance for the task after taking the observation.
                #The reduction in variance is the benefit for the agent-task pair.
                new_agent_task_var = 1/(1/agent_task_vars[i,j] + 1/sensor_var)
                benefit_matrix[i,j,k] = agent_task_vars[i,j] - new_agent_task_var

                #Update the variance of the task based on the new measurement
                agent_task_vars[i,j] = new_agent_task_var

        #Add variance to all tasks
        agent_task_vars += var_add

    return benefit_matrix

def get_science_constellation_satcovers_and_graphs_coverage(num_planes, num_sats_per_plane,T,inc, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
    """
    Generate benefit matrix of with (num_planes*sats_per_plane)
    satellites covering the entire surface of the earth, with tasks
    evenly covering the globe at the lowest H3 reslution possible (~10 deg lat/lon).

    Input an inclination for the satellites and the tasks.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #10 evenly spaced planes of satellites, each with n/10 satellites per plane
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    inc = inc*u.deg
    argp = 0*u.deg

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
    hexagons = generate_smooth_coverage_hexagons((-inc.to_value(u.deg), inc.to_value(u.deg)), (-180, 180))
    #Add tasks at centroid of all hexagons
    for hexagon in hexagons:
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        lat = polygon.centroid.y
        lon = polygon.centroid.x

        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        #use benefits which are uniformly 1 to get scaling matrix
        task = Task(task_loc, np.ones(T))
        const.add_task(task)

    sat_cover_matrix, graphs = const.propagate_orbits(T, calc_fov_based_proximities)
    return sat_cover_matrix, graphs

def solve_science_w_dynamic_haal(sat_coverage_matrix, init_var, base_sensor_var, init_assignment, lambda_, L, 
                                      distributed=False, parallel=False, verbose=False,
                                      eps=0.01, graphs=None, track_iters=False,
                                      benefit_fn=generic_handover_pen_benefit_fn, benefit_info=None):
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

    if graphs is None:
        if distributed:
            graphs = [nx.complete_graph(n) for i in range(T)]
        else:
            graphs = [None]*T

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []
    task_vars = init_var * np.ones(m)

    vars = np.zeros((m,T))
    for k in range(T):
        if verbose: print(f"Solving w HAAL, {k}/{T}", end='\r')

        #build benefit matrix from task_objects and sat_cover_matrix
        tstep_end = min(k+L, T)
        benefit_window = np.copy(sat_coverage_matrix[:,:,k:tstep_end])

        for i in range(n):
            for j in range(m):
                if sat_coverage_matrix[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = base_sensor_var / sat_coverage_matrix[i,j,k]
                #Benefit is scaled by the change in variance resulting from the measurement
                benefit_window[i,j,:] *= (task_vars[j] - 1/(1/task_vars[j] + 1/sensor_var))

        task_vars += 0.01
        vars[:,k] = task_vars

        len_window = benefit_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(graphs[k]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(benefit_window, curr_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=graphs[k], lambda_=lambda_, benefit_fn=benefit_fn,
                                                     benefit_info=benefit_info)
            haal_d_auction.run_auction()
            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, benefit_window, 
                                                                      lambda_, parallel_approx=parallel, benefit_fn=benefit_fn,
                                                                      benefit_info=benefit_info)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment

        #Update the variance of the task based on the new measurement
        for j in range(m):
            if np.max(curr_assignment[:,j]) == 1:
                i = np.argmax(curr_assignment[:,j])
                if sat_coverage_matrix[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = base_sensor_var / sat_coverage_matrix[i,j,k]
                task_vars[j] = 1/(1/task_vars[j] + 1/sensor_var)
        task_vars += 0.01
        vars[:,k] = task_vars

    if not track_iters or not distributed:
        return chosen_assignments, vars
    else:
        return chosen_assignments, vars, total_iterations/T

def solve_science_w_nha(sat_coverage_matrix, init_var, base_sensor_var, init_assignment, lambda_, 
                                      distributed=False, parallel=False, verbose=False,
                                      eps=0.01, graphs=None, track_iters=False,
                                      benefit_fn=generic_handover_pen_benefit_fn, benefit_info=None):
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

    if graphs is None:
        if distributed:
            graphs = [nx.complete_graph(n) for i in range(T)]
        else:
            graphs = [None]*T

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []
    task_vars = init_var * np.ones(m)

    vars = np.zeros((m,T))
    for k in range(T):
        if verbose: print(f"Solving w HAAL, {k}/{T}", end='\r')

        #build benefit matrix from task_objects and sat_cover_matrix
        benefit_window = np.copy(sat_coverage_matrix[:,:,k])
        for i in range(n):
            for j in range(m):
                if sat_coverage_matrix[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = base_sensor_var / sat_coverage_matrix[i,j,k]
                #Benefit is scaled by the change in variance resulting from the measurement
                benefit_window[i,j] *= (task_vars[j] - 1/(1/task_vars[j] + 1/sensor_var))

        csol = solve_centralized(benefit_window)
        curr_assignment = convert_central_sol_to_assignment_mat(n,m,csol)
        
        chosen_assignments.append(curr_assignment)

        #Update the variance of the task based on the new measurement
        for j in range(m):
            if np.max(curr_assignment[:,j]) == 1:
                i = np.argmax(curr_assignment[:,j])
                if sat_coverage_matrix[i,j,k] == 0: sensor_var = 1000000
                else: sensor_var = base_sensor_var / sat_coverage_matrix[i,j,k]
                task_vars[j] = 1/(1/task_vars[j] + 1/sensor_var)
        task_vars += 0.01
        vars[:,k] = task_vars

    if not track_iters or not distributed:
        return chosen_assignments, vars
    else:
        return chosen_assignments, vars, total_iterations/T

if __name__ == "__main__":
    num_planes = 25
    num_sats_per_plane = 25
    i = 70
    T = 93
    lambda_ = 0.05
    L = 6

    # sat_cover_matrix, graphs = get_science_constellation_satcovers_and_graphs_coverage(num_planes, num_sats_per_plane, T, i)
    # with open('soil_moisture/soil_data/sat_cover_matrix.pkl','wb') as f:
    #     pickle.dump(sat_cover_matrix, f)
    # with open('soil_moisture/soil_data/graphs.pkl','wb') as f:
    #     pickle.dump(graphs, f)

    with open('soil_moisture/soil_data/sat_cover_matrix.pkl','rb') as f:
        sat_cover_matrix = pickle.load(f)
    with open('soil_moisture/soil_data/graphs.pkl','rb') as f:
        graphs = pickle.load(f)
    
    ass, vars = solve_science_w_nha(sat_cover_matrix, 1, 0.1, None, lambda_)
    tv = vars.sum()
    nha_vars = np.sum(vars, axis=0)
    _, nh = calc_value_and_num_handovers(ass, np.zeros_like(sat_cover_matrix), None, lambda_)

    with open('soil_moisture/soil_data/nha_vars.pkl', 'wb') as f:
        pickle.dump(vars, f)
    with open('soil_moisture/soil_data/nha_ass.pkl', 'wb') as f:
        pickle.dump(ass, f)

    print(f"NHA: value {tv}, nh {nh}")

    ##########################################################

    ass, vars = solve_science_w_dynamic_haal(sat_cover_matrix, 1, 0.1, None, lambda_, L,
                                           verbose=True)
    haal_vars = np.sum(vars, axis=0)
    tv = vars.sum()
    _, nh = calc_value_and_num_handovers(ass, np.zeros_like(sat_cover_matrix), None, lambda_)

    with open('soil_moisture/soil_data/haal_vars.pkl', 'wb') as f:
        pickle.dump(vars, f)
    with open('soil_moisture/soil_data/haal_ass.pkl', 'wb') as f:
        pickle.dump(ass, f)

    print(f"HAAL: value {tv}, nh {nh}")

    plt.plot(nha_vars, label='NHA')
    plt.plot(haal_vars, label='HAAL')
    plt.legend()
    plt.show()
