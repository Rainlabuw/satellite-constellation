import numpy as np

from constellation_sim.ConstellationSim import ConstellationSim, generate_smooth_coverage
from constellation_sim.Task import Task
from constellation_sim.Satellite import Satellite

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

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

def get_constellation_bens_and_graphs_area_coverage(lat_range, lon_range, fov=60):
    """
    Generate constellation benefits 
    """
    const = ConstellationSim(dt = 30*u.second)
    earth = Earth

    #Generate evenly space tasks over a region of the earth
    lats, lons = generate_smooth_coverage(lat_range, lon_range)
    for lat, lon in zip(lats, lons):
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

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

    print(len(const.tasks))
    T = 60 #30 minutes
    benefits, _ = const.propagate_orbits(T, calc_object_track_benefits)

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
    print("\nnum sats",len(const.sats))

    for k in range(T):
        num_active_sats = 0
        for i in range(limited_benefits.shape[0]):
            if np.sum(limited_benefits[i,:,k]) > 0:
                num_active_sats += 1
            
        print(f"time {k}, {num_active_sats} active sats")

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)

    get_constellation_bens_and_graphs_area_coverage(lat_range, lon_range)