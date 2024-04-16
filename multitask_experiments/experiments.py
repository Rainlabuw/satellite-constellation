import pickle

from multitask_experiments.multi_task_env import MultiTaskAssignEnv
from haal_experiments.simple_assign_env import SimpleAssignEnv

from multitask_experiments.solve_w_efficient_haal import solve_w_efficient_haal
from algorithms.solve_w_haal import solve_w_haal

from constellation_sim.constellation_generators import get_prox_mat_and_graphs_area

def init_exp():
    num_planes = 10
    num_sats_per_plane = 10
    T = 93
    lat_range = (22, 52)
    lon_range = (-124.47, -66.87)
    lambda_ = 0.5

    # sat_prox_matrix, graphs, hex_to_task_mapping, const = get_prox_mat_and_graphs_area(num_planes, num_sats_per_plane, T, lat_range, lon_range, isl_dist=4000)

    # with open('multitask_experiments/data/sat_prox_mat.pkl', 'wb') as f:
    #     pickle.dump(sat_prox_matrix, f)
    # with open('multitask_experiments/data/graphs.pkl', 'wb') as f:
    #     pickle.dump(graphs, f)
    # with open('multitask_experiments/data/hex_to_task_mapping.pkl', 'wb') as f:
    #     pickle.dump(hex_to_task_mapping, f)
    # with open('multitask_experiments/data/constellation.pkl', 'wb') as f:
    #     pickle.dump(const, f)

    with open('multitask_experiments/data/sat_prox_mat.pkl', 'rb') as f:
        sat_prox_matrix = pickle.load(f)
    with open('multitask_experiments/data/graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    with open('multitask_experiments/data/hex_to_task_mapping.pkl', 'rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('multitask_experiments/data/constellation.pkl', 'rb') as f:
        const = pickle.load(f)

    print(sat_prox_matrix.shape)

    env = MultiTaskAssignEnv(sat_prox_matrix, None, lambda_, 5, 0.9, graphs=graphs)
    senv = SimpleAssignEnv(sat_prox_matrix, None, lambda_)

    chosen_assignments, total_value = solve_w_efficient_haal(env, 3, distributed=True)
    print(total_value)

    env.reset()
    chosen_assignments, total_value = solve_w_efficient_haal(env, 3)
    print(total_value)

if __name__ == "__main__":
    init_exp()