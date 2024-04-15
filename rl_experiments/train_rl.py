from rl_constellation.networks import PolicyNetwork, ValueNetwork

def train_rl():
    policy_network = PolicyNetwork()
    value_network = ValueNetwork()

    L = 10
    n = 100
    m = 350
    M = 10
    num_filters = 10
    hidden_units = 156
    batch_size = 10

    while True:


if __name__ == "__main__":
    train_rl()