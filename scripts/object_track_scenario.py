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

class ConstellationAndObjectSim(ConstellationSim):
    def __init__(self, dt=30 * u.second, isl_dist=None) -> None:
        super().__init__(dt, isl_dist)

        self.tgt_objects = []

def calc_object_track_benefits(sat, task):
    """
    Given a satellite and a task, computes the benefit of the satellite.

    We calculate the angle between the satellite and the task, and then
    use a gaussian to determine the benefit, starting at 5% of the benefit
    when the angle between the satellite and the task is the maximum FOV,
    and rising to the maximum when the satellite is directly overhead.
    """
    sat_r = sat.orbit.r.to_value(u.km)
    sat_to_task = task.loc.cartesian_cords.to_value(u.km) - sat_r

    angle_btwn = np.arccos(np.dot(-sat_r, sat_to_task)/(np.linalg.norm(sat_r)*np.linalg.norm(sat_to_task)))
    angle_btwn *= 180/np.pi #convert to degrees

    if angle_btwn < sat.fov and task.loc.is_visible(*sat.orbit.r):
        gaussian_height = task.benefit
        height_at_max_fov = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-sat.fov**2/(2*np.log(height_at_max_fov/gaussian_height)))

        task_benefit = gaussian_height*np.exp(-angle_btwn**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0
    
    return task_benefit

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

    #Generate tasks, with appropriate benefit matrices as objects pass through
    num_tasks = 50
    add_tasks_with_objects(num_tasks, lat_range, lon_range, const, T)

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
    benefits_w_backup_tasks[:,limited_benefits.shape[1]:,:] = limited_benefits*0.25

    for k in range(T):
        num_active_sats = 0
        for i in range(limited_benefits.shape[0]):
            if np.sum(limited_benefits[i,:,k]) > 0:
                num_active_sats += 1

    with open('object_track_experiment/benefits_large_const_50_tasks.pkl','wb') as f:
        pickle.dump(benefits_w_backup_tasks, f)
    with open('object_track_experiment/graphs_large_const_50_tasks.pkl','wb') as f:
        pickle.dump(graphs, f)

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)

    get_constellation_bens_and_graphs_object_tracking_area(lat_range, lon_range)

    with open('object_track_experiment/benefits_large_const_50_tasks.pkl', 'rb') as f:
        ben = pickle.load(f)
        print(ben.shape)