from rl_experiments.networks import PolicyNetwork, ValueNetwork

from common.methods import *

def train_rl():
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()

    L = 10
    n = 10
    m = 15
    M = 10
    num_filters = 10
    hidden_units = 156
    batch_size = 10
    T = 100

    sat_prox_mat = generate_benefits_over_time(n, m, T, 3, 6)

    

    while True:


if __name__ == "__main__":
    train_rl()