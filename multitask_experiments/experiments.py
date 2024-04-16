import pickle
from astropy import units as u
import numpy as np

from multitask_experiments.multi_task_env import MultiTaskAssignEnv
from haal_experiments.simple_assign_env import SimpleAssignEnv

from multitask_experiments.solve_w_efficient_haal import solve_w_efficient_haal
from algorithms.solve_w_haal import solve_w_haal

from constellation_sim.constellation_generators import get_prox_mat_and_graphs_area, get_prox_mat_and_graphs_coverage

from common.methods import *
from scripts.classic_auction import Auction

def init_exp():
    num_planes = 10
    num_sats_per_plane = 10
    T = 93
    lat_range = (22, 52)
    lon_range = (-124.47, -66.87)
    lambda_ = 0.5

    # sat_prox_matrix, graphs, hex_to_task_mapping, const = get_prox_mat_and_graphs_area(num_planes, num_sats_per_plane, T, lat_range, lon_range, isl_dist=4000)
    sat_prox_matrix, graphs = get_prox_mat_and_graphs_coverage(num_planes, num_sats_per_plane, T, 70, isl_dist=4000)

    with open('multitask_experiments/data/sat_prox_mat.pkl', 'wb') as f:
        pickle.dump(sat_prox_matrix, f)
    with open('multitask_experiments/data/graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)

    with open('multitask_experiments/data/sat_prox_mat.pkl', 'rb') as f:
        sat_prox_matrix = pickle.load(f)
    with open('multitask_experiments/data/graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    print(sat_prox_matrix.shape)

    env = MultiTaskAssignEnv(sat_prox_matrix, None, lambda_, 5, 0.9, graphs=graphs)

    chosen_assignments, total_value = solve_w_efficient_haal(env, 3, distributed=True, verbose=True)
    print(total_value)

    env.reset()
    chosen_assignments, total_value = solve_w_haal(env, 3, distributed=True, verbose=True)
    print(total_value)

def prices_test():
    n = 10
    m = 10
    auction = Auction(n, m)
    _, col_ind, cval = auction.solve_centralized()
    cass = convert_central_sol_to_assignment_mat(n, m, col_ind)

    dval = auction.run_auction()
    dass = convert_agents_to_assignment_matrix(auction.agents)

    print(cval, dval)
    print(auction.agents[0].public_prices)

if __name__ == "__main__":
    prices_test()