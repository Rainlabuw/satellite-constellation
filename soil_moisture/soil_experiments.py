import pickle
import numpy as np

from soil_moisture.solve_var_w_haal import solve_var_w_dynamic_haal, variance_based_benefit_fn
from envs.variance_min_env import VarianceMinEnv
from haal.solve_w_haal import solve_w_haal

from common.methods import *

def create_basic_data():
    num_planes = 25
    num_sats_per_plane = 25
    i = 70
    T = 93
    lambda_ = 5
    L = 1

    # sat_prox_mat, graphs = get_constellation_proxs_and_graphs_coverage(num_planes, num_sats_per_plane, T, i)
    # with open('soil_moisture/soil_data/sat_prox_mat.pkl','wb') as f:
    #     pickle.dump(sat_prox_mat, f)
    # with open('soil_moisture/soil_data/graphs.pkl','wb') as f:
    #     pickle.dump(graphs, f)

    with open('soil_moisture/soil_data/sat_prox_mat.pkl','rb') as f:
        sat_prox_mat = pickle.load(f)
    with open('soil_moisture/soil_data/graphs.pkl','rb') as f:
        graphs = pickle.load(f)
    
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    init_task_vars = 1 * np.ones(m)

    benefit_info = BenefitInfo()
    benefit_info.task_vars = np.copy(init_task_vars)
    benefit_info.base_sensor_var = 0.1
    benefit_info.var_add = 0.01

    # ass, vars = solve_science_w_nha(sat_prox_mat, 1, 0.1, None, lambda_)
    # tv = vars.sum()
    # nha_vars = np.sum(vars, axis=0)
    # _, nh = calc_value_and_num_handovers(ass, np.zeros_like(sat_prox_mat), None, lambda_)
    print(f"L = 1")
    ass, tv, vars = solve_var_w_dynamic_haal(sat_prox_mat, None, lambda_, 1, verbose=True, benefit_info=benefit_info)
    haa_total_var = vars.sum()
    haa_vars = np.sum(vars, axis=0)
    haa_nh = calc_handovers_generically(ass)

    with open('soil_moisture/soil_data/haa_vars.pkl', 'wb') as f:
        pickle.dump(vars, f)
    with open('soil_moisture/soil_data/haa_ass.pkl', 'wb') as f:
        pickle.dump(ass, f)

    print(f"HAA: total value {tv}, total variance {haa_total_var}, nh {haa_nh}")

    env = VarianceMinEnv(sat_prox_mat, None, lambda_)
    _, tv = solve_w_haal(env, 1)
    vars = env.task_var_hist
    haa_total_var = vars.sum()
    haa_vars = np.sum(vars, axis=0)
    haa_nh = calc_handovers_generically(ass)
    print(f"HAA env: total value {tv}, total variance {haa_total_var}, nh {haa_nh}")

    ##########################################################
    benefit_info = BenefitInfo()
    benefit_info.task_vars = np.copy(init_task_vars)
    benefit_info.base_sensor_var = 0.1
    benefit_info.var_add = 0.01

    print(f"L = {L}")
    ass, tv, vars = solve_var_w_dynamic_haal(sat_prox_mat, None, lambda_, L, benefit_info=benefit_info,
                                           verbose=True)
    haal_vars = np.sum(vars, axis=0)
    haal_total_var = vars.sum()
    haal_nh = calc_handovers_generically(ass)

    with open('soil_moisture/soil_data/haal_vars.pkl', 'wb') as f:
        pickle.dump(vars, f)
    with open('soil_moisture/soil_data/haal_ass.pkl', 'wb') as f:
        pickle.dump(ass, f)

    print(f"HAAL: total value {tv}, total variance {haal_total_var}, nh {haal_nh}")

    plt.plot(haa_vars, label='HAA')
    plt.plot(haal_vars, label='HAAL')
    plt.legend()
    plt.show()

def analyze_var():
    with open('soil_moisture/soil_data/sat_prox_mat.pkl', 'rb') as f:
        sat_prox_mat = pickle.load(f)

    with open('soil_moisture/soil_data/haal_vars.pkl', 'rb') as f:
        haal_vars = pickle.load(f)

    with open('soil_moisture/soil_data/haal_ass.pkl', 'rb') as f:
        haal_ass = pickle.load(f)

    with open('soil_moisture/soil_data/haa_vars.pkl', 'rb') as f:
        haa_vars = pickle.load(f)

    with open('soil_moisture/soil_data/haa_ass.pkl', 'rb') as f:
        haa_ass = pickle.load(f)

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]

    # avg_pass_len, avg_pass_ben, avg_ass_len = calc_pass_statistics(sat_prox_mat, haal_ass)
    # print("HAAL", avg_pass_len, avg_pass_ben, avg_ass_len)

    # avg_pass_len, avg_pass_ben, avg_ass_len = calc_pass_statistics(sat_prox_mat, haa_ass)
    # print("HAA", avg_pass_len, avg_pass_ben, avg_ass_len)

    haa_total_vars = np.sum(haa_vars, axis=0)
    haal_total_vars = np.sum(haal_vars, axis=0)

    # plt.plot(haal_total_vars, label="HAAL")
    # plt.plot(haa_total_vars, label="HAA")
    # plt.legend()
    # plt.show()

    # plt.plot(haal_vars[0,:])
    # plt.plot(haa_vars[0,:])
    # plt.show()

    for j in range(m):
        plt.plot(haal_vars[j,:], 'g', alpha=0.1)
        plt.plot(haa_vars[j,:], 'r', alpha=0.1)
    plt.show()

def experiment1():
    num_planes = 25
    num_sats_per_plane = 25
    i = 70
    T = 93
    L = 4

    # sat_prox_mat, graphs = get_constellation_proxs_and_graphs_coverage(num_planes, num_sats_per_plane, T, i)
    # with open('soil_moisture/soil_data/sat_prox_mat.pkl','wb') as f:
    #     pickle.dump(sat_prox_mat, f)
    # with open('soil_moisture/soil_data/graphs.pkl','wb') as f:
    #     pickle.dump(graphs, f)

    with open('soil_moisture/soil_data/sat_prox_mat.pkl','rb') as f:
        sat_prox_mat = pickle.load(f)
    with open('soil_moisture/soil_data/graphs.pkl','rb') as f:
        graphs = pickle.load(f)

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    # # ~~~~~~~~~~~~~~~~~~~~~~ NHA ~~~~~~~~~~~~~~~~~~~~~~ #

    # benefit_info = BenefitInfo()
    # benefit_info.task_vars = 1 * np.ones(m)
    # benefit_info.base_sensor_var = 0.1
    # benefit_info.var_add = 0.01

    # nha_ass, nha_tv, nha_vars = solve_var_w_dynamic_haal(sat_prox_mat, None, 1, 1, verbose=True, benefit_info=benefit_info)
    # nha_total_var = nha_vars.sum()
    # nha_vars_over_time = np.sum(nha_vars, axis=0)
    # print(nha_vars_over_time.shape)
    # nha_nh = calc_handovers_generically(nha_ass)
    # print(f"NHA total variance {nha_total_var}, nhs {nha_nh}")

    # # ~~~~~~~~~~~~~~~~~~~~~~~ HAAL ~~~~~~~~~~~~~~~~~~~~~~ #
    # haal_asses = []
    # haal_tvs = []
    # haal_total_vars = []
    # haal_vars_over_times = []
    # haal_nhs = []
    # haal_base_vars = []

    # lambda_ = 5
    # for l in range(1,L+1):
    #     benefit_info.task_vars = 1 * np.ones(m) #reset benefit info
    #     print(f"L={l}")
    #     ass, tv, vars = solve_var_w_dynamic_haal(sat_prox_mat, None, lambda_, l, verbose=True, benefit_info=benefit_info)
    #     haal_total_var = vars.sum()
    #     haal_vars_over_time = np.sum(vars, axis=0)
    #     haal_nh = calc_handovers_generically(ass)

    #     haal_asses.append(ass)
    #     haal_tvs.append(tv)
    #     haal_total_vars.append(haal_total_var)
    #     haal_vars_over_times.append(haal_vars_over_time)
    #     haal_nhs.append(haal_nh)
    #     haal_base_vars.append(vars)

    # with open('soil_moisture/soil_data/nha_ass.pkl', 'wb') as f:
    #     pickle.dump(nha_ass, f)
    # with open('soil_moisture/soil_data/nha_vars.pkl', 'wb') as f:
    #     pickle.dump(nha_vars, f)

    # with open('soil_moisture/soil_data/haal_ass_diffLs.pkl', 'wb') as f:
    #     pickle.dump(haal_asses, f)
    # with open('soil_moisture/soil_data/haal_vars_diffLs.pkl', 'wb') as f:
    #     pickle.dump(haal_base_vars, f)

    with open('soil_moisture/soil_data/nha_ass.pkl', 'rb') as f:
        nha_ass = pickle.load(f)
    with open('soil_moisture/soil_data/nha_vars.pkl', 'rb') as f:
        nha_vars = pickle.load(f)

    with open('soil_moisture/soil_data/haal_ass_diffLs.pkl', 'rb') as f:
        haal_asses = pickle.load(f)
    with open('soil_moisture/soil_data/haal_vars_diffLs.pkl', 'rb') as f:
        haal_base_vars = pickle.load(f)

    apl, apb, aal = calc_pass_statistics(sat_prox_mat, nha_ass)
    print(apl, aal)
    apl, apb, aal = calc_pass_statistics(sat_prox_mat, haal_asses[0])
    print(apl, aal)
    apl, apb, aal = calc_pass_statistics(sat_prox_mat, haal_asses[1])
    print(apl, aal)

    print(haal_base_vars[0].shape)
    #
    print(np.sum(haal_base_vars[0], axis=0).shape)
    # plt.plot(np.sum(nha_vars, axis=0), label="NHA")
    for i, haal_base_var in enumerate(haal_base_vars):
        plt.plot(np.sum(haal_base_var, axis=0), label=f"HAAL, L={i+1}")
    plt.legend()
    plt.ylabel("Total variance for all areas")
    plt.xlabel("Timestep")
    plt.show()

    for j in range(m):
        plt.plot(haal_base_vars[0][j,:], 'r', alpha=0.1)
        plt.plot(haal_base_vars[1][j,:], 'g', alpha=0.1)
    plt.plot(0, 'r', label='HAAL, L=1')
    plt.plot(0, 'g', label='HAAL, L=2')
    plt.legend()
    plt.ylabel("Individual area variances")
    plt.xlabel("Timestep")
    plt.show()
    # print("Total variances", haal_total_vars)
    # print("NHs", haal_nhs)

    # #Plot
    # plt.plot(range(1,L+1), haal_tvs)
    # plt.show()


    # plt.plot(nha_vars_over_time, label="NHA")
    # for l in range(1,L+1):
    #     plt.plot(haal_vars_over_time[l], label=l)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    create_basic_data()