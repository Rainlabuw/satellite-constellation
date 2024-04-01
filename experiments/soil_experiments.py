import pickle
import numpy as np
from poliastro.spheroid_location import SpheroidLocation

from constellation_sim.ConstellationSim import ConstellationSim
from constellation_sim.constellation_generators import get_prox_mat_and_graphs_soil_moisture

from envs.variance_min_env import VarianceMinEnv
from envs.simple_assign_env import SimpleAssignEnv

from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_wout_handover import solve_wout_handover
from algorithms.solve_greedily import solve_greedily

from common.methods import *

def generate_sat_prox_mats():
    num_planes = 40
    num_sats_per_plane = 25
    T = 60
    #Approximately America
    lat_range = (22, 52)
    lon_range = (-124.47, -66.87)

    sat_prox_mat, graphs = get_prox_mat_and_graphs_soil_moisture(num_planes, num_sats_per_plane, T,
                                                                 lat_range, lon_range, dt=30*u.second)

    with open('experiments/soil_data/sat_prox_mat_usa.pkl','wb') as f:
        pickle.dump(sat_prox_mat, f)
    with open('experiments/soil_data/graphs_usa.pkl','wb') as f:
        pickle.dump(graphs, f)

def experiment1():
    num_planes = 40
    num_sats_per_plane = 25
    i = 70
    T = 93
    L = 3

    with open('experiments/soil_data/sat_prox_mat_usa.pkl','rb') as f:
        sat_prox_mat = pickle.load(f)
    with open('experiments/soil_data/graphs_usa.pkl','rb') as f:
        graphs = pickle.load(f)

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    print(sat_prox_mat.shape)

    init_ass = None
    lambda_ = 5

    env = VarianceMinEnv(sat_prox_mat, init_ass, lambda_)

    # # NHA
    # env.reset()
    # nha_ass, nha_tv = solve_wout_handover(env, verbose=True)
    # nha_vars = env.task_var_hist
    # print(f"NHA: value {nha_tv}, total variance {nha_vars.sum()}")

    # # Greedy
    # env.reset()
    # greedy_ass, greedy_tv = solve_greedily(env, verbose=True)
    # greedy_vars = env.task_var_hist
    # print(f"Greedy: value {greedy_tv}, total variance {greedy_vars.sum()}")

    # # HAAL
    # haal_vars_list = []
    # haal_asses = []
    # haal_tvs = []
    # for l in range(1,L+1):
    #     env.reset()
    #     haal_ass, haal_tv = solve_w_haal(env, l, verbose=True)
    #     haal_vars = env.task_var_hist
    #     haal_vars_list.append(haal_vars)
    #     haal_asses.append(haal_ass)

    #     print(f"HAAL (L={l}): value {haal_tv}, total variance {haal_vars.sum()}")
    
    # with open('experiments/soil_data/nha_ass_usa.pkl', 'wb') as f:
    #     pickle.dump(nha_ass, f)
    # with open('experiments/soil_data/greedy_ass_usa.pkl', 'wb') as f:
    #     pickle.dump(greedy_ass, f)
    # with open('experiments/soil_data/haal_asses_usa.pkl', 'wb') as f:
    #     pickle.dump(haal_asses, f)
    
    # with open('experiments/soil_data/nha_var_usa.pkl', 'wb') as f:
    #     pickle.dump(nha_vars, f)
    # with open('experiments/soil_data/greedy_var_usa.pkl', 'wb') as f:
    #     pickle.dump(greedy_vars, f)
    # with open('experiments/soil_data/haal_vars_usa.pkl', 'wb') as f:
    #     pickle.dump(haal_vars_list, f)

    with open('experiments/soil_data/nha_ass_usa.pkl', 'rb') as f:
        nha_ass = pickle.load(f)
    with open('experiments/soil_data/greedy_ass_usa.pkl', 'rb') as f:
        greedy_ass = pickle.load(f)
    with open('experiments/soil_data/haal_asses_usa.pkl', 'rb') as f:
        haal_asses = pickle.load(f)
    
    with open('experiments/soil_data/nha_var_usa.pkl', 'rb') as f:
        nha_vars = pickle.load(f)
    with open('experiments/soil_data/greedy_var_usa.pkl', 'rb') as f:
        greedy_vars = pickle.load(f)
    with open('experiments/soil_data/haal_vars_usa.pkl', 'rb') as f:
        haal_vars_list = pickle.load(f)

    # for i in range(n):
    #     hist = []
    #     for k in range(T):
    #         hist.append(np.argmax(greedy_ass[k][i,:]))
    #     plt.plot(hist)
    # plt.show()

    _, _, nha_ass_len =  calc_pass_statistics(sat_prox_mat, nha_ass)
    _, _, greedy_ass_len =  calc_pass_statistics(sat_prox_mat, greedy_ass)
    print(f"NHA ass len {nha_ass_len}")
    print(f"Greedy ass len {greedy_ass_len}")

    # for l in range(L):
    #     _, _, haal_ass_len = calc_pass_statistics(sat_prox_mat, haal_asses[l])
    #     print(f"HAAL (L={l+1}) ass len {haal_ass_len}")
    minutes = [0.5*k for k in range(T)]
    plt.plot(minutes,np.sum(haal_vars_list[2], axis=0), color='green', label=f"HAAL, L={2+1} (ours)")
    plt.plot(minutes,np.sum(greedy_vars, axis=0), color='red', label="Greedy")
    plt.plot(minutes,np.sum(nha_vars, axis=0), color='blue', label="Naively Optimal")
    plt.xlabel("Time (min)")
    plt.ylabel("Total Variance")
    plt.title("Total Variance over Time")
    plt.legend()
    plt.show()

    nha_nh = calc_handovers_generically(nha_ass, None, env.benefit_info)
    greedy_nh = calc_handovers_generically(greedy_ass, None, env.benefit_info)
    haal_nh = calc_handovers_generically(haal_asses[2], None, env.benefit_info)

    labels = ['Naively Optimal', 'Greedy', 'HAAL']
    values = [nha_nh, greedy_nh, haal_nh]

    plt.bar(labels, values)
    plt.ylabel('Number of Handovers')
    plt.title('Handovers Comparison')
    plt.show()

if __name__ == "__main__":
    # generate_sat_prox_mats()
    experiment1()