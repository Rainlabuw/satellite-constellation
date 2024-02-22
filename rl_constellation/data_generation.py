"""
Generating training data from HAAL for RL algorithms.
"""
import numpy as np
import pickle
import multiprocessing as mp

from common.methods import *

from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation

from constellation_sim.ConstellationSim import ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from haal.solve_w_haal import solve_w_haal

def init_random_constellation(num_planes, num_sats_per_plane, m, T, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
    """
    Initialize a random constellation with num_planes planes and num_sats_per_plane satellites per plane.

    Tasks are placed randomly.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist, dtype=np.float16)
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

    return benefits, graphs

def worker_process(id, num_planes, num_sats_per_plane, m, T, isl_dist):
    n = num_planes * num_sats_per_plane

    # Perform the operations that were inside your loop
    benefits, graphs = init_random_constellation(num_planes, num_sats_per_plane, m, T, isl_dist=isl_dist)
    init_assign = np.eye(n, m)
    assigns, _ = solve_w_haal(benefits, init_assign, 0.5, 3)

    return benefits, graphs, assigns

def generate_benefit_assignment_pairs(num_sims, num_planes, num_sats_per_plane, m, T, isl_dist):
    """
    Generates a list of benefits and assignments for a given number of planes and satellites per plane.

    Lists will be len <num_sims>, each with benefit matrices and assignments from <T> time steps.
    """
    benefit_list = []
    graph_list = []
    assignments_list = []

    args = [(i, num_planes, num_sats_per_plane, m, T, isl_dist) for i in range(num_sims)]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(worker_process, args)

    benefit_list, graph_list, assignments_list = zip(*results)

    return benefit_list, graph_list, assignments_list

def convert_benefit_assignment_pairs_to_dataset(benefit_list, assignments_list):
    """
    Converts a list of benefits and assignments to a dataset for training.
    """
    gamma = 0.9
    L_required = np.ceil(np.log(0.05)/np.log(gamma))

    num_runs = 10
    for run in range(num_runs):
        with open(f"rl_constellation/data/benefits_{run}.pkl", 'rb') as f:
            benefits = pickle.load(f)
        with open(f"rl_constellation/data/assigns_{run}.pkl", 'rb') as f:
            assigns = pickle.load(f)

        n = benefits.shape[0]
        m = benefits.shape[1]
        T = benefits.shape[2]

        for k in range(T):
            pass

    for benefits, assigns in zip(benefit_list, assignments_list):
        pass

if __name__ == "__main__":
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes*num_sats_per_plane
    m = 350

    T = 1000
    isl_dist = 4000

    num_sims = 10
    benefit_list, graph_list, assignments_list = generate_benefit_assignment_pairs(num_sims, num_planes, num_sats_per_plane, m, T, isl_dist)

    for benefits, graphs, assigns, id in zip(benefit_list, graph_list, assignments_list, range(num_sims)):
        print(f"Saving data from run {id}...")
        with open(f"rl_constellation/data/benefits_{id}.pkl", 'wb') as f:
            pickle.dump(benefits, f)
        with open(f"rl_constellation/data/graphs_{id}.pkl", 'wb') as f:
            pickle.dump(graphs, f)
        with open(f"rl_constellation/data/assigns_{id}.pkl", 'wb') as f:
            pickle.dump(assigns, f)