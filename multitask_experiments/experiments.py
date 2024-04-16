import pickle
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import tqdm

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

def price_seeding_test():
    n = 50
    m = 50
    
    perturbation_levels = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]
    total_price = 0

    total_iterations = [0]*len(perturbation_levels)
    total_base_iterations = 0

    total_value = [0]*len(perturbation_levels)
    total_true_opt = 0
    total_base_value = 0
    
    num_tests = 50
    for test in tqdm.tqdm(range(num_tests)):
        benefits = np.random.rand(n, m)

        base_auction = Auction(n, m, proximities=benefits)
        bval = base_auction.run_auction()
        _, _, true_opt = base_auction.solve_centralized()
        base_prices = base_auction.agents[0].public_prices
        total_base_iterations += base_auction.n_iterations
        total_base_value += bval
        total_true_opt += true_opt
        total_price += np.sum(base_prices)/m

        for i, pert in enumerate(perturbation_levels):
            pert_prices = base_prices + np.random.normal(0, pert, m)
            auction = Auction(n, m, proximities=benefits, prices=pert_prices)
            val = auction.run_auction()

            total_iterations[i] += auction.n_iterations
            total_value[i] += val

    avg_iters = [iters/num_tests for iters in total_iterations]
    avg_base_iters = [total_base_iterations/num_tests for _ in range(len(perturbation_levels))]
    avg_value = [val/num_tests for val in total_value]
    avg_base_value = [total_base_value/num_tests for _ in range(len(perturbation_levels))]
    avg_true_opt = [total_true_opt/num_tests for _ in range(len(perturbation_levels))]
    avg_price = total_price/num_tests
    lower_bound = [ato - n*0.01 for ato in avg_true_opt]

    fig, axes = plt.subplots(2,1,figsize=(8, 6))
    axes[0].set_xscale('log')
    axes[0].plot(perturbation_levels, avg_iters, label='Seeded Auction Iterations')
    axes[0].plot(perturbation_levels, avg_base_iters, '--', label='Base Auction Iterations')
    axes[0].vlines(avg_price, 0, max(max(avg_iters), max(avg_base_iters))*1.1, colors='r', linestyles='dotted', label='Average Base Price')
    axes[1].set_xscale('log')
    axes[1].plot(perturbation_levels, avg_value, label='Seeded Auction Value')
    axes[1].plot(perturbation_levels, avg_base_value, '--', label='Base Auction Value')
    axes[1].plot(perturbation_levels, avg_true_opt, '--', label='True Optimal Value')
    axes[1].plot(perturbation_levels, lower_bound, '--', label='Auction n*eps Lower Bound')
    axes[0].set_ylim(0, max(max(avg_iters), max(avg_base_iters))*1.1)
    axes[0].legend()
    axes[1].legend()
    plt.suptitle(f"Price Seeding Test (n={n}, m={m}), {num_tests} trials")
    plt.show()
    plt.savefig('20_20_price_seeding_test.png')

if __name__ == "__main__":
    price_seeding_test()