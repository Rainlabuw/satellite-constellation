"""
Generating training data from HAAL for RL algorithms.
"""
import numpy as np
import pickle
from common.methods import *

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation

from constellation_sim.ConstellationSim import ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

def init_random_constellation(num_planes, num_sats_per_plane, m, T, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
    """
    Initialize a random constellation with num_planes planes and num_sats_per_plane satellites per plane.

    Tasks are placed randomly.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at <altitude> km.~~~~~~~~~~~~~
    #TODO: For now, generating training data at a fixed inclination, etc. Could generalize this
    #in the future.
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    inc = 58*u.deg
    argp = 0*u.deg

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
    lon_max = 180
    lat_max = 55

    for _ in range(m):
        lon = np.random.uniform(-lon_max, lon_max)
        lat = np.random.uniform(-lat_max, lat_max)
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2, size=T)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    benefits, graphs = const.propagate_orbits(T, calc_fov_benefits)

    with open('data/uncompressed_benefits.pkl', 'wb') as f:
        pickle.dump(benefits, f)
    benefits = np.array(benefits, dtype=np.float16)
    with open('data/compressed_benefits.pkl', 'wb') as f:
        pickle.dump(benefits, f)
    with open('data/graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)

    return benefits, graphs

if __name__ == "__main__":
    init_random_constellation(10, 10, 350, 1000, isl_dist=4000)