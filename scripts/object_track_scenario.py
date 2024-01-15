import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
import pickle

from constellation_sim.ConstellationSim import ConstellationSim, generate_smooth_coverage
from constellation_sim.Task import Task
from constellation_sim.Satellite import Satellite

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u
import h3

from methods import *
from solve_w_haal import solve_w_haal

def add_tasks_with_objects(num_objects, lat_range, lon_range, const, T):
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

def get_constellation_bens_and_graphs_object_tracking_area(lat_range, lon_range, fov=60):
    """
    Generate constellation benefits based on tracking ojects
    """
    T = 60 #30 minutes
    const = ConstellationSim(dt = 30*u.second, isl_dist=2500)
    earth = Earth

    #Generate tasks and add to const, with appropriate benefit matrices as objects pass through
    num_objects = 50 #number of objects to track
    add_tasks_with_objects(num_objects, lat_range, lon_range, const, T)

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

    benefits, graphs = const.propagate_orbits(T, calc_fov_benefits)

    limited_benefits = np.zeros_like(benefits)
    active_sats = []
    for i in range(const.n):
        total_sat_benefits = np.sum(benefits[i,:,:])
        if total_sat_benefits > 0:
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            limited_benefits[curr_sat.id,:,:] = benefits[i,:,:]
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    limited_benefits = limited_benefits[:len(const.sats),:,:] #truncate unused satellites

    #Create second opportunity for tasks to be completed, with 25% of the benefits
    benefits_w_backup_tasks = np.zeros((limited_benefits.shape[0], limited_benefits.shape[1]*2, limited_benefits.shape[2]))
    benefits_w_backup_tasks[:,:limited_benefits.shape[1],:] = limited_benefits
    benefits_w_backup_tasks[:,limited_benefits.shape[1]:,:] = limited_benefits * 0.1
    non_padded_m = benefits_w_backup_tasks.shape[1]

    #if necessary, pad the benefit matrix with zeros so that n<=m
    padding_size = max(0,benefits_w_backup_tasks.shape[0]-benefits_w_backup_tasks.shape[1])
    benefits = np.pad(benefits_w_backup_tasks, ((0,0), (0, padding_size), (0,0)))
    m = benefits.shape[1]

    #Create scaling matrix for task transitions
    task_transition_state_dep_scaling_mat = np.ones((m,m))
    
    #no penalty when transitioning between backup and primary versions of the same task
    for j in range(non_padded_m//2):
        task_transition_state_dep_scaling_mat[j,j+non_padded_m//2] = 0
        task_transition_state_dep_scaling_mat[j+non_padded_m//2,j] = 0

    #no penalty when transitioning to a dummy task
    for j in range(non_padded_m, m):
        task_transition_state_dep_scaling_mat[:,j] = 0

    with open('object_track_experiment/benefits_large_const_50_tasks.pkl','wb') as f:
        pickle.dump(benefits, f)
    with open('object_track_experiment/graphs_large_const_50_tasks.pkl','wb') as f:
        pickle.dump(graphs, f)
    with open('object_track_experiment/task_transition_scaling_large_const_50_tasks.pkl','wb') as f:
        pickle.dump(task_transition_state_dep_scaling_mat, f)

def timestep_loss_state_dep_fn(benefits, prev_assign, lambda_, task_trans_state_dep_scaling_mat=None):
    """
    Adds a loss to the benefit matrix which encodes the cost of switching
    being losing the entire benefit of the task in the first timestep, plus a small
    extra penalty (lambda_).

    Designed to 
    """
    if prev_assign is None: return benefits

    m = benefits.shape[1]
    if task_trans_state_dep_scaling_mat is None:
        task_trans_state_dep_scaling_mat = np.ones((m,m))
    state_dep_scaling = prev_assign @ task_trans_state_dep_scaling_mat

    b = np.where((prev_assign == 0) & (state_dep_scaling > 0), -lambda_*state_dep_scaling, benefits)
    # #print all of numpy array
    # np.set_printoptions(threshold=np.inf)
    # print(b)
    return b

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)

    get_constellation_bens_and_graphs_object_tracking_area(lat_range, lon_range)