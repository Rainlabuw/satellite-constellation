"""
Master file of all experiments for RL satellite constellation.
"""
import pickle
import time
import torch

from common.methods import *

from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_wout_handover import solve_wout_handover
from algorithms.solve_randomly import solve_randomly

from rl_experiments.data_generation import init_random_constellation
from rl_experiments.networks import PolicyNetwork
from rl_experiments.solve_w_rl import solve_w_rl

from haal_experiments.simple_assign_env import SimpleAssignEnv

def test_rl_policy():
    """
    Test the RL policy on a simple constellation.
    """
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes*num_sats_per_plane
    m = 350
    T = 93
    lambda_ = 0.5

    # benefits, graphs = init_random_constellation(num_planes, num_sats_per_plane, m, T, isl_dist=4000)
    # with open(f"rl_experiments/data/100_sat_const_benefits.pkl", 'wb') as f:
    #     pickle.dump(benefits, f)
    # with open(f"rl_experiments/data/100_sat_const_graphs.pkl", 'wb') as f:
    #     pickle.dump(graphs, f)

    with open(f"rl_experiments/data/100_sat_const_benefits.pkl", 'rb') as f:
        benefits = pickle.load(f)

    env = SimpleAssignEnv(benefits, None, lambda_)

    num_filters = 10
    hidden_units = 64
    M = 10
    L = 10
    policy_network = PolicyNetwork(L, n, m, M, num_filters, hidden_units)

    state_dict = torch.load('rl_experiments/networks/policy_network_pretrained.pt')
    policy_network.load_state_dict(state_dict)

    st = time.time()
    ass, tv = solve_w_rl(env, policy_network, M, L, verbose=True)
    print(f"Time to solve with RL: {time.time()-st}")

    print("RL sequence valid: ",is_assignment_mat_sequence_valid(ass))
    print(f"RL Total value: {tv}")

    env.reset()
    st = time.time()
    ass, tv = solve_w_haal(env, 6, verbose=True)
    print(f"Time to solve with HAAL: {time.time()-st}")
    print(f"HAAL Total value: {tv}")

    env.reset()
    ass, tv = solve_wout_handover(env)
    print(tv)

    env.reset()
    ass, tv = solve_randomly(env)
    print(tv)

def test_lightweight():
    n = 100
    m = 350 
    T = 93
    ben = generate_benefits_over_time(n,m,T, 3, 6, 0.25, 2)

    # with open(f"rl_experiments/data/100_sat_const_benefits.pkl", 'rb') as f:
    #     ben = pickle.load(f)

    # n = ben.shape[0]
    # m = ben.shape[1]
    # T = ben.shape[2]

    num_tasks_in_view = []
    for i in range(n):
        for k in range(T):
            num_tasks_in_view.append(np.sum(np.where(ben[i,:,k] > 0.01, 1, 0)))

    min_val = min(num_tasks_in_view)
    max_val = max(num_tasks_in_view)
    bins = np.arange(min_val-0.5, max_val+1.5, 1)
    plt.hist(num_tasks_in_view, bins=bins)
    plt.show()

if __name__ == "__main__":
    test_rl_policy()