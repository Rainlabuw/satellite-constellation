import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from tqdm import tqdm
import time
import pickle
from collections import defaultdict

from common.methods import *
from common.plotting_utils import plot_object_track_scenario, plot_multitask_scenario

from haal.solve_optimally import solve_optimally
from haal.solve_wout_handover import solve_wout_handover
from haal.solve_w_haal import solve_w_haal
from haal.solve_w_accelerated_haal import solve_w_accel_haal
from haal.solve_w_centralized_CBBA import solve_w_centralized_CBBA
from haal.solve_w_CBBA import solve_w_CBBA
from haal.solve_greedily import solve_greedily
from haal.object_track_scenario import timestep_loss_pen_benefit_fn, init_task_objects, get_benefits_from_task_objects, solve_object_track_w_dynamic_haal, get_sat_prox_mat_and_graphs_object_tracking_area
from haal.object_track_utils import calc_pct_objects_tracked, object_tracking_history
from haal.multi_task_scenario import solve_multitask_w_haal, calc_multiassign_benefit_fn, get_benefit_matrix_and_graphs_multitask_area

from envs.simple_assign_env import SimpleAssignEnv

from constellation_sim.ConstellationSim import get_constellation_proxs_and_graphs_coverage, get_constellation_proxs_and_graphs_random_tasks, ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

import h3
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split
from shapely.geometry import box
import matplotlib.image as mpimg
from math import radians, cos, sin, asin, sqrt, atan2, degrees

def optimal_baseline_plot():
    """
    Compare various solutions types against the true optimal,
    plot for final paper
    """
    n = 5
    m = 5
    T = 3
    
    L = T

    init_ass = None
    
    lambda_ = 0.5

    avg_best = 0
    avg_haal = 0
    avg_mha = 0
    avg_no_handover = 0

    # num_avgs = 50
    # for _ in tqdm(range(num_avgs)):
    #     benefit = np.random.rand(n,m,T)

    #     #SMHGL centralized, lookahead = 3
    #     _, haal_ben = solve_w_haal(benefit, init_ass, lambda_, L)
    #     avg_haal += haal_ben/num_avgs

    #     #MHA
    #     _, mha_ben = solve_w_haal(benefit, init_ass, lambda_, 1)
    #     avg_mha += mha_ben/num_avgs

    #     #Naive
    #     _, ben = solve_wout_handover(benefit, init_ass, lambda_)
    #     avg_no_handover += ben/num_avgs

    #     #Optimal
    #     _, ben = solve_optimally(benefit, init_ass, lambda_)
    #     avg_best += ben/num_avgs
    results = [avg_no_handover, avg_mha, avg_haal, avg_best]

    fig = plt.figure()
    
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    
    results = [7.564384786984931, 8.863084750293483, 9.743930778962238, 9.753256871430807]
    plt.bar([r"HAA ($\lambda = 0$)","HAA", f"HAAL (L={L})", "Optimal"], results)
    plt.ylabel("Total Value")
    print(["No Handover","HAA", f"HAAL (L={L})", "Optimal"])
    print([avg_no_handover, avg_mha, avg_haal, avg_best])
    plt.savefig("optimal553.pdf")
    plt.show()

def distributed_comparison():
    """
    Compare the performance of the distributed HAAL algorithm
    to other algorithms (including centralized HAAL)
    """
    n = 20
    m = 20
    T = 10
    L = 4
    lambda_ = 0.5

    graphs = [rand_connected_graph(n) for i in range(T)]

    ctot = 0
    catot = 0
    dtot = 0

    no_handover_tot = 0
    cbba_tot = 0

    n_tests = 10
    for _ in tqdm(range(n_tests)):
        benefits = np.random.rand(n, m, T)
        init_assignment = np.eye(n,m)

        # st = time.time()
        _, d_val, _ = solve_w_haal(benefits, init_assignment,lambda_, L, distributed=True)

        _, c_val, _ = solve_w_haal(benefits, lambda_, init_assignment, L, distributed=False)

        _, ca_val, _ = solve_w_haal(benefits, init_assignment, lambda_, L, distributed=False, central_approx=True)

        _, no_handover_val, _ = solve_wout_handover(benefits, init_assignment, lambda_)
        _, cbba_val, _ = solve_w_centralized_CBBA(benefits, init_assignment, lambda_)

        ctot += c_val/n_tests
        catot += ca_val/n_tests
        dtot += d_val/n_tests

        no_handover_tot += no_handover_val/n_tests
        cbba_tot += cbba_val/n_tests

    print([no_handover_tot, cbba_tot, ctot, catot, dtot])
    plt.bar(range(6), [no_handover_tot, cbba_tot, ctot, catot, dtot], tick_label=["Naive", "CBBA", "Centralized", "Centralized Approx", "Distributed"])
    plt.show()

def realistic_orbital_simulation():
    """
    Simulate a realistic orbital mechanics case
    using distributed and centralized HAAL.

    Compute this over several lookahead windows.
    """
    n = 648
    m = 100
    T = 93
    max_L = 6
    lambda_ = 0.5
    init_assignment = None

    cbba_tot = 0
    no_handover_tot = 0
    greedy_tot = 0

    tot_iters_by_lookahead = np.zeros(3)
    tot_value_by_lookahead = np.zeros(3)

    num_avgs = 1
    for _ in range(num_avgs):
        print(f"\nNum trial {_}")
        num_planes = 36
        num_sats_per_plane = 18
        if n != num_planes*num_sats_per_plane: raise Exception("Make sure n = num planes * num sats per plane")
        # benefits, graphs = get_constellation_proxs_and_graphs_random_tasks(num_planes,num_sats_per_plane,m,T, benefit_func=calc_distance_based_proximities)
        benefits, graphs = get_constellation_proxs_and_graphs_coverage(num_planes,num_sats_per_plane,T,70,benefit_func=calc_distance_based_proximities)

        #Ensure all graphs are connected
        for i, graph in enumerate(graphs):
            if not nx.is_connected(graph):
                print("Graph not connected! Editing graph at timestep")
                graphs[i] = nx.complete_graph(n)

        # #CBBA
        # print(f"Done generating benefits, solving CBBA...")
        # _, cbba_val, _ = solve_w_centralized_CBBA(benefits, init_assignment, lambda_, verbose=True)
        # cbba_tot += cbba_val/num_avgs

        #Naive
        print(f"Done solving CBBA, solving no_handover...")
        _, no_handover_val, _ = solve_wout_handover(benefits, init_assignment, lambda_)
        no_handover_tot += no_handover_val/num_avgs

        #greedy
        print(f"Done solving no_handover, solving greedy...")
        _, greedy_val, _ = solve_greedily(benefits, init_assignment, lambda_)
        greedy_tot += greedy_val/num_avgs

        iters_by_lookahead = []
        value_by_lookahead = []
        # for L in range(1,max_L+1):
        for L in [1,3,6]:
            print(f"lookahead {L}")
            _, d_val, _, avg_iters = solve_w_haal(benefits, init_assignment, lambda_, L, graphs=graphs, verbose=True, track_iters=True)

            iters_by_lookahead.append(avg_iters)
            value_by_lookahead.append(d_val)

        tot_iters_by_lookahead += np.array(iters_by_lookahead)/num_avgs
        tot_value_by_lookahead += np.array(value_by_lookahead)/num_avgs

    fig, axes = plt.subplots(2,1)
    
    print(tot_value_by_lookahead, no_handover_tot, greedy_tot)
    # axes[0].plot(range(1,max_L+1), tot_value_by_lookahead, 'g', label="HAAL-D")
    axes[0].plot([1,3,6], tot_value_by_lookahead, 'g', label="HAAL-D")
    # axes[0].plot(range(1,max_L+1), [cbba_tot]*max_L, 'b--', label="CBBA")
    axes[0].plot([1,3,6], [no_handover_tot]*max_L, 'r--', label="Naive")
    # axes[0].plot(range(1,max_L+1), [no_handover_tot]*max_L, 'r--', label="Naive")
    # axes[0].plot(range(1,max_L+1), [greedy_tot]*max_L, 'k--', label="Greedy")
    axes[0].plot([1,3,6], [greedy_tot]*max_L, 'k--', label="Greedy")
    axes[0].set_ylabel("Total value")
    axes[0].set_xticks(range(1,max_L+1))
    axes[0].set_ylim((0, 1.1*max(tot_value_by_lookahead)))
    axes[0].legend()

    axes[1].plot([1,3,6], tot_iters_by_lookahead, 'g', label="HAAL-D")
    # axes[1].plot(range(1,max_L+1), tot_iters_by_lookahead, 'g', label="HAAL-D")
    axes[1].set_ylim((0, 1.1*max(tot_iters_by_lookahead)))
    axes[1].set_ylabel("Average iterations")
    axes[1].set_xlabel("Lookahead window")
    axes[1].set_xticks(range(1,max_L+1))

    plt.savefig('real_const.png')
    plt.show()

def epsilon_effect():
    """
    Simulate a realistic orbital mechanics case
    using distributed and centralized HAAL.

    Determine how the choice of epsilon for the auction
    effects the difference
    """
    n = 50
    m = 50
    T = 40
    L = 5
    lambda_ = 1
    init_assignment = np.eye(n,m)

    d_tot_0p1 = 0
    d_tot_0p01 = 0
    d_tot_0p001 = 0
    c_tot = 0
    cbba_tot = 0
    no_handover_tot = 0

    num_avgs = 1
    for _ in tqdm(range(num_avgs)):
        benefits, graphs = get_constellation_proxs_and_graphs_random_tasks(10,5,m,T)

        #Distributed
        print(f"Done generating benefits, solving distributed 0.1...")
        _, d_val_0p1, _ = solve_w_haal(benefits, init_assignment, lambda_, L, distributed=True, graphs=None, eps=0.1, verbose=True)
        d_tot_0p1 += d_val_0p1/num_avgs

        print(f"Done generating benefits, solving distributed 0.01...")
        _, d_val_0p01, _ = solve_w_haal(benefits, init_assignment, lambda_, L, distributed=True, graphs=None, eps=0.01, verbose=True)
        d_tot_0p01 += d_val_0p01/num_avgs

        print(f"Done generating benefits, solving distributed 0.001...")
        _, d_val_0p001, _ = solve_w_haal(benefits, init_assignment, lambda_, L, distributed=True, graphs=None, eps=0.001, verbose=True)
        d_tot_0p001 += d_val_0p001/num_avgs

        #Centralized
        print(f"Done solving distributed, solving centralized...")
        _, c_val, _ = solve_w_haal(benefits, init_assignment, lambda_, L, distributed=False)
        c_tot += c_val/num_avgs

        #CBBA
        print(f"Done solving centralized, solving CBBA...")
        _, cbba_val, _ = solve_w_centralized_CBBA(benefits, init_assignment, lambda_, verbose=True)
        cbba_tot += cbba_val/num_avgs

        #Naive
        print(f"Done solving CBBA, solving no_handover...")
        _, no_handover_val, _ = solve_wout_handover(benefits, init_assignment, lambda_)
        no_handover_tot += no_handover_val/num_avgs

    print([no_handover_tot, cbba_tot, c_tot, d_tot_0p1, d_tot_0p01, d_tot_0p001])
    plt.bar(range(6), [no_handover_tot, cbba_tot, c_tot, d_tot_0p1, d_tot_0p01, d_tot_0p001], tick_label=["Naive", "CBBA", "Centralized", "Distributed 0.1", "Distributed 0.01", "Distributed 0.001"])
    plt.title(f"Realistic Constellation Sim with complete graphs over {num_avgs} runs, n={n}, m={m}, T={T}, L={L}, lambda={lambda_}")
    plt.savefig("epsilon_effect.png")
    plt.show()

def lookahead_optimality_testing():
    """
    Do any problems violate the new bounds (11/7)?
    """
    n = 3
    m = 3
    T = 5

    for _ in range(1000):
        print(_,end='\r')
        benefit = np.random.random((n,m,T))
        _, opt_val, _ = solve_optimally(benefit, None, 0.5)
        for lookahead in range(T, T+1):
            lower_bd = (0.5+0.5*(lookahead-1)/T)*opt_val
            _, val, _ = solve_w_haal(benefit, None, 0.5, lookahead, distributed=False)
            if val < opt_val:
                print("Bound violation!!")
                print(benefit)
                print(f"Lookahead {lookahead} value: {val}, lower bound: {lower_bd}, opt_val: {opt_val}")
                return

def mha_vs_naive_counterexample():
    init_assign = np.eye(4)
    lambda_ = 1
    benefits = np.zeros((4,4,2))

    benefits[:,:,0] = np.array([[100, 1, 1, 1],
                                [1, 100, 1, 1],
                                [1, 1, 1, 1.1],
                                [1, 1, 1.1, 1]])

    benefits[:,:,1] = np.array([[100, 1, 1, 1],
                                [1, 100, 1, 1],
                                [1, 1, 1, 100],
                                [1, 1, 100, 1]])
    
    _, mv, _ = solve_w_haal(benefits, init_assign, lambda_, 1)
    _, nv, _ = solve_wout_handover(benefits, init_assign, lambda_)
    print(mv, nv)

def performance_v_num_agents_line_chart():
    ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    T = 10
    L = 5
    lambda_ = 0.5
    num_avgs = 5
    init_assign = None

    no_handover_vals = []
    sga_vals = []
    mha_vals = []
    haal_vals = []
    haald_vals = []

    no_handover_nhs = []
    sga_nhs = []
    mha_nhs = []
    haal_nhs = []
    haald_nhs = []

    for n in ns:
        print(f"Solving for {n} agents...")
        m = n

        no_handover_total_val = 0
        sga_total_val = 0
        mha_total_val = 0
        haal_total_vals = 0
        haald_total_vals = 0

        no_handover_total_nhs = 0
        sga_total_nhs = 0
        mha_total_nhs = 0
        haal_total_nhs = 0
        haald_total_nhs = 0
        
        for _ in range(num_avgs):
            print(_)
            benefits = np.random.random((n, m, T))

            _, no_handover_val, no_handover_nh = solve_wout_handover(benefits, init_assign, lambda_)
            no_handover_total_val += no_handover_val/num_avgs
            no_handover_total_nhs += no_handover_nh/num_avgs

            _, sga_val, sga_nh = solve_w_centralized_CBBA(benefits, init_assign, lambda_)
            sga_total_val += sga_val/num_avgs
            sga_total_nhs += sga_nh/num_avgs

            _, mha_val, mha_nh = solve_w_haal(benefits, init_assign, lambda_, 1)
            mha_total_val += mha_val/num_avgs
            mha_total_nhs += mha_nh/num_avgs

            _, haal_val, haal_nh = solve_w_haal(benefits, init_assign, lambda_, L)
            haal_total_vals += haal_val/num_avgs
            haal_total_nhs += haal_nh/num_avgs

            _, haald_val, haald_nh = solve_w_haal(benefits, init_assign, lambda_, L, distributed=True)
            haald_total_vals += haald_val/num_avgs
            haald_total_nhs += haald_nh/num_avgs

        no_handover_vals.append(no_handover_total_val/n)
        sga_vals.append(sga_total_val/n)
        mha_vals.append(mha_total_val/n)
        haal_vals.append(haal_total_vals/n)
        haald_vals.append(haald_total_vals/n)

        no_handover_nhs.append(no_handover_total_nhs/n)
        sga_nhs.append(sga_total_nhs/n)
        mha_nhs.append(mha_total_nhs/n)
        haal_nhs.append(haal_total_nhs/n)
        haald_nhs.append(haald_total_nhs/n)

    fig, axes = plt.subplots(2,1, sharex=True)
    fig.suptitle(f"Performance vs. number of agents over {num_avgs} runs, m=n, T={T}, L={L}, lambda={lambda_}")
    axes[0].plot(ns, no_handover_vals, label="Naive")
    axes[0].plot(ns, sga_vals, label="SGA")
    axes[0].plot(ns, mha_vals, label="MHA")
    axes[0].plot(ns, haal_vals, label=f"HAAL (L={L})")
    axes[0].plot(ns, haald_vals, label=f"HAAL-D (L={L})")
    axes[0].set_ylabel("Average benefit per agent")
    
    axes[1].plot(ns, no_handover_nhs, label="Naive")
    axes[1].plot(ns, sga_nhs, label="SGA")
    axes[1].plot(ns, mha_nhs, label="MHA")
    axes[1].plot(ns, haal_nhs, label=f"HAAL (L={L})")
    axes[1].plot(ns, haald_nhs, label=f"HAAL-D (L={L})")
    axes[1].set_ylabel("Average number of handovers per agent")
    
    axes[1].set_xlabel("Number of agents")

    axes[1].legend()

    print(no_handover_vals)
    print(sga_vals)
    print(mha_vals)
    print(haal_vals)
    print(haald_vals)

    print(no_handover_nhs)
    print(sga_nhs)
    print(mha_nhs)
    print(haal_nhs)
    print(haald_nhs)
    plt.savefig("performance_v_num_agents.png")
    plt.show()

def tasking_history_plot():
    """
    Tracks the history of task allocations in a system over time,
    with and without HAAL
    """
    n = 150
    m = 300
    num_planes = 15
    num_sats_per_plane = 10
    altitude=550
    T = 20
    lambda_ = 0.75

    # benefits, graphs = get_constellation_proxs_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_based_proximities)

    # benefits, graphs = get_constellation_proxs_and_graphs_coverage(num_planes,num_sats_per_plane,T,5, altitude=altitude, benefit_func=calc_fov_based_proximities)

    # with open('bens.pkl', 'wb') as f:
    #     pickle.dump(benefits, f)

    # with open('bens.pkl', 'rb') as f:
    #     benefits = pickle.load(f)

    with open("paper_exp1_bens.pkl", 'rb') as f:
        benefits = pickle.load(f)
    with open("paper_exp1_graphs.pkl", 'rb') as f:
        graphs = pickle.load(f)

    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]
    init_assign = np.eye(n, m)

    # benefits = np.random.random((n, m, T))

    no_handover_ass, no_handover_val, _ = solve_wout_handover(benefits, init_assign, lambda_)

    haal_ass, haal_val, _ = solve_w_haal(benefits, init_assign, lambda_, 1, graphs=None)

    haal5_ass, haal5_val, _ = solve_w_haal(benefits, init_assign, lambda_, 6, graphs=None)

    greedy_ass, greedy_val, _ = solve_greedily(benefits, init_assign, lambda_)

    print(no_handover_val, haal_val, haal5_val, greedy_val)

    #~~~~~~~~~~~~~~~~~~~~~~ PLOT OF TASKING HISTORY ~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(4,1, sharex=True)
    agent1_no_handover_ass = [np.argmax(no_handover_a[0,:]) for no_handover_a in no_handover_ass]

    agent1_haal_ass = [np.argmax(haal_a[0,:]) for haal_a in haal_ass]

    agent1_haal5_ass = [np.argmax(haal5_a[0,:]) for haal5_a in haal5_ass]

    agent1_greedy_ass = [np.argmax(greedy_a[0,:]) for greedy_a in greedy_ass]
    agent2_greedy_ass = [np.argmax(greedy_a[1,:]) for greedy_a in greedy_ass]
    agent3_greedy_ass = [np.argmax(greedy_a[2,:]) for greedy_a in greedy_ass]

    axes[0].plot(range(T), agent1_no_handover_ass, label="Not Considering Handover")
    axes[1].plot(range(T), agent1_haal_ass, label="HAAL 1")
    axes[2].plot(range(T), agent1_haal5_ass, label="HAAL 5")
    axes[3].plot(range(T), agent1_greedy_ass, label="Greedy1")
    axes[3].plot(range(T), agent2_greedy_ass, label="Greedy2")
    axes[3].plot(range(T), agent3_greedy_ass, label="Greedy3")
    axes[3].set_xlabel("Time (min.)")
    axes[0].set_ylabel("Task assignment")
    axes[1].set_ylabel("Task assignment")
    axes[2].set_ylabel("Task assignment")
    axes[3].set_ylabel("Task assignment")

    axes[0].set_title("Satellite 0 tasking, Naive")
    axes[1].set_title("Satellite 0 tasking, MHA")
    axes[2].set_title("Satellite 0 tasking, HAAL (L=5)")
    axes[3].set_title("Satellite 0 tasking, Greedy")
    plt.show(block=False)

    #~~~~~~~~~~~~~~~~~~~~ PLOT OF PRODUCTIVE TASKS COMPLETED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    no_handover_valid_tasks = []
    haal_valid_tasks = []
    haal5_valid_tasks = []
    for k in range(T):
        no_handover_assigned_benefits = no_handover_ass[k]*benefits[:,:,k]
        num_no_handover_valid_tasks = np.sum(np.where(no_handover_assigned_benefits, 1, 0))

        no_handover_valid_tasks.append(num_no_handover_valid_tasks)

        haal_assigned_benefits = haal_ass[k]*benefits[:,:,k]
        num_haal_valid_tasks = np.sum(np.where(haal_assigned_benefits, 1, 0))

        haal_valid_tasks.append(num_haal_valid_tasks)

        haal5_assigned_benefits = haal5_ass[k]*benefits[:,:,k]
        num_haal5_valid_tasks = np.sum(np.where(haal5_assigned_benefits, 1, 0))

        haal5_valid_tasks.append(num_haal5_valid_tasks)

    fig = plt.figure()
    plt.plot(range(T), no_handover_valid_tasks, label="Not Considering Handover")
    plt.plot(range(T), haal_valid_tasks, label="MHA")
    plt.plot(range(T), haal5_valid_tasks, label="HAAL")
    plt.legend()
    plt.show(block=False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT OF BENEFITS CAPTURED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure()
    gs = fig.add_gridspec(3,2)
    no_handover_ax = fig.add_subplot(gs[0,0])
    haal_ax = fig.add_subplot(gs[1,0])
    haal5_ax = fig.add_subplot(gs[2,0])
    val_ax = fig.add_subplot(gs[:,1])

    prev_no_handover = 0
    prev_haal = 0
    prev_haal5 = 0

    no_handover_ben_line = []
    haal_ben_line = []
    haal5_ben_line = []

    no_handover_val_line = []
    haal_val_line = []
    haal5_val_line = []
    greedy_val_line = []
    
    for k in range(T):
        no_handover_choice = np.argmax(no_handover_ass[k][0,:])
        haal_choice = np.argmax(haal_ass[k][0,:])
        haal5_choice = np.argmax(haal5_ass[k][0,:])

        if prev_no_handover != no_handover_choice:
            no_handover_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                # no_handover_ben_line.append(no_handover_ben_line[-1])
                if len(no_handover_ben_line) > 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r')
                elif len(no_handover_ben_line) == 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r.', markersize=1)
            no_handover_ben_line = [benefits[0,no_handover_choice, k]]
        else:
            no_handover_ben_line.append(benefits[0, no_handover_choice, k])

        if prev_haal != haal_choice:
            haal_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(haal_ben_line) > 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'b')
                elif len(haal_ben_line) == 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'b.', markersize=1)
            haal_ben_line = [benefits[0,haal_choice, k]]
        else:
            haal_ben_line.append(benefits[0,haal_choice, k])

        if prev_haal5 != haal5_choice:
            haal5_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(haal5_ben_line) > 1:
                    haal5_ax.plot(range(k-len(haal5_ben_line), k), haal5_ben_line,'g')
                elif len(haal5_ben_line) == 1:
                    haal5_ax.plot(range(k-len(haal5_ben_line), k), haal5_ben_line,'g.', markersize=1)

            haal5_ben_line = [benefits[0,haal5_choice, k]]
        else:
            haal5_ben_line.append(benefits[0,haal5_choice, k])

        no_handover_val_so_far, _ = calc_value_and_num_handovers(no_handover_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        no_handover_val_line.append(no_handover_val_so_far)

        haal_val_so_far, _ = calc_value_and_num_handovers(haal_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        haal_val_line.append(haal_val_so_far)

        haal5_val_so_far, _ = calc_value_and_num_handovers(haal5_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        haal5_val_line.append(haal5_val_so_far)

        greedy_val_so_far, _ = calc_value_and_num_handovers(greedy_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        greedy_val_line.append(greedy_val_so_far)

        prev_no_handover = no_handover_choice
        prev_haal = haal_choice
        prev_haal5 = haal5_choice

    #plot last interval
    no_handover_ax.plot(range(k+1-len(no_handover_ben_line), k+1), no_handover_ben_line, 'r')
    haal_ax.plot(range(k+1-len(haal_ben_line), k+1), haal_ben_line,'b')
    haal5_ax.plot(range(k+1-len(haal5_ben_line), k+1), haal5_ben_line,'g')

    #plot value over time
    val_ax.plot(range(T), no_handover_val_line, 'r', label='Not Considering Handover')
    val_ax.plot(range(T), haal_val_line, 'b', label='HAAL')
    val_ax.plot(range(T), haal5_val_line, 'g', label='HAAL 5')
    val_ax.plot(range(T), greedy_val_line, 'k', label='Greedy')
    val_ax.legend()

    plt.show()

def test_optimal_L(timestep=1*u.min, altitude=550, fov=60):
    a = Earth.R.to(u.km) + altitude*u.km
    sat = Satellite(Orbit.from_classical(Earth, a, 0*u.one, 0*u.deg, 0*u.deg, 0*u.deg, 0*u.deg), [], [], fov=fov)
    
    L = generate_max_L(timestep, sat)
    return L

def cbba_testing():
    np.random.seed(42)
    val = 0
    cval = 0
    for _ in range(50):
        benefits = np.random.random((30,30,10))

        # benefits = np.zeros((3,3,2))
        # benefits[:,:,0] = np.array([[10, 1, 1],
        #                             [1, 10, 1],
        #                             [1, 1, 10]])
        # benefits[:,:,1] = np.array([[1, 10, 1],
        #                             [1, 11, 10],
        #                             [10, 1, 1]])

        init_assign = None

        lambda_ = 0.5

        ass, vall, _ = solve_w_centralized_CBBA(benefits, init_assign, lambda_, benefits.shape[-1])
        val += vall
        cass, cvall, _ = solve_w_CBBA(benefits, init_assign, lambda_, benefits.shape[-1])
        cval += cvall
    print(val, cval)

def connectivity_testing():
    # def f(isl_dist, nm):
    #     T = 93
    #     _, graphs = get_constellation_proxs_and_graphs_random_tasks(nm, nm, 1, T, isl_dist=isl_dist)

    #     pct_connected = sum([nx.is_connected(graph) for graph in graphs])/T
    #     return pct_connected
    
    # isl_dist = np.linspace(1500, 7500, 12)
    # nm = np.linspace(5,30,25)
    # x, y = np.meshgrid(isl_dist,nm)
    # z = f(x,y)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(x, y, z, cmap='viridis')

    # plt.show()

    _, graphs = get_constellation_proxs_and_graphs_random_tasks(10, 10, 1, 93, isl_dist=4000)
    pct_connected = sum([nx.is_connected(graph) for graph in graphs])/93
    print("\n",pct_connected)

def paper_experiment2_compute_assigns():
    lambda_ = 0.5

    with open("haal_experiment2/paper_exp2_bens.pkl", 'rb') as f:
        symmetric_benefits = pickle.load(f)
    with open("haal_experiment2/paper_exp2_graphs.pkl", 'rb') as f:
        graphs = pickle.load(f)

    nohand_assigns, nohand_val, nohand_nh = solve_wout_handover(symmetric_benefits, None, lambda_)
    with open("haal_experiment2/paper_exp2_nohand_assigns.pkl", 'wb') as f:
        pickle.dump(nohand_assigns, f)

    greedy_assigns, greedy_val, greedy_nh = solve_greedily(symmetric_benefits, None, lambda_)
    with open("haal_experiment2/paper_exp2_greedy_assigns.pkl", 'wb') as f:
        pickle.dump(greedy_assigns, f)
    
    mha_assigns, mha_val, mha_nh = solve_w_haal(symmetric_benefits, None, lambda_, 6, verbose=True)
    with open("haal_experiment2/paper_exp2_haalc_assigns.pkl", 'wb') as f:
        pickle.dump(mha_assigns, f)

    with open("haal_experiment2/other_alg_results.txt", 'w') as f:
        f.write(f"No handover value: {nohand_val}\n")
        f.write(f"No handover handovers: {nohand_nh}\n")

        f.write(f"Greedy value: {greedy_val}\n")
        f.write(f"Greedy handovers: {greedy_nh}\n")

        f.write(f"HAAL Centralized value: {mha_val}\n")
        f.write(f"MHL Centralized handovers: {mha_nh}\n")

def paper_experiment2_tasking_history():
    with open('haal_experiment2/paper_exp2_greedy_assigns.pkl', 'rb') as f:
        greedy_assigns = pickle.load(f)
    with open('haal_experiment2/paper_exp2_haalc_assigns.pkl', 'rb') as f:
        haalc_assigns = pickle.load(f)
    with open('haal_experiment2/paper_exp2_nohand_assigns.pkl', 'rb') as f:
        nohand_assigns = pickle.load(f)

    with open("haal_experiment2/paper_exp2_bens.pkl", 'rb') as f:
        benefits = pickle.load(f)

    # greedy_assigns, _, _ = solve_greedily(benefits, None, 0.5)

    # with open("haal_experiment2/paper_exp2_greedy_assigns.pkl", 'wb') as f:
    #     pickle.dump(greedy_assigns, f)

    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[-1]

    lambda_ = 0.5
    init_assign = None

    nohand_val, nohand_nh = calc_value_and_num_handovers(nohand_assigns, benefits, init_assign, lambda_)
    greedy_val, greedy_nh = calc_value_and_num_handovers(greedy_assigns, benefits, init_assign, lambda_)
    haalc_val, haalc_nh = calc_value_and_num_handovers(haalc_assigns, benefits, init_assign, lambda_)

    # haal_coverages = []
    # greedy_coverages = []
    # nohand_coverages = []
    # for k in range(T):
    #     haal_productive_assigns = np.where(haalc_assigns[k][:,:813]*benefits[:,:813,k] > 0, 1, 0)
    #     greedy_productive_assigns = np.where(greedy_assigns[k][:,:813]*benefits[:,:813,k] > 0, 1, 0)
    #     nohand_productive_assigns = np.where(nohand_assigns[k][:,:813]*benefits[:,:813,k] > 0, 1, 0)
    #     print(np.sum(haal_productive_assigns), np.sum(greedy_productive_assigns), np.sum(nohand_productive_assigns))
    #     haal_coverages.append(np.sum(haal_productive_assigns)/812)
    #     greedy_coverages.append(np.sum(greedy_productive_assigns)/812)
    #     nohand_coverages.append(np.sum(nohand_productive_assigns)/812)
    # haal_avg_cover = sum(haal_coverages) / len(haal_coverages)
    # greedy_avg_cover = sum(greedy_coverages) / len(greedy_coverages)
    # nohand_avg_cover = sum(nohand_coverages) / len(nohand_coverages)

    # #~~~~~~~~~~~~~~~~~~~~~~ PLOT OF TASKING HISTORY ~~~~~~~~~~~~~~~~~~~~~~~~~
    # fig, axes = plt.subplots(4,1, sharex=True)
    # agent1_no_handover_ass = [np.argmax(no_handover_a[0,:]) for no_handover_a in nohand_assigns]

    # agent1_haal5_ass = [np.argmax(haal5_a[0,:]) for haal5_a in haalc_assigns]

    # agent1_greedy_ass = [np.argmax(greedy_a[0,:]) for greedy_a in greedy_assigns]

    # axes[0].plot(range(T), agent1_no_handover_ass, label="Not Considering Handover")
    # axes[1].plot(range(T), agent1_haal5_ass, label="HAAL-C")
    # axes[2].plot(range(T), agent1_greedy_ass, label="Greedy1")
    # axes[2].set_xlabel("Time (min.)")
    # axes[0].set_ylabel("Task assignment")
    # axes[1].set_ylabel("Task assignment")
    # axes[2].set_ylabel("Task assignment")

    # axes[0].set_title("Satellite 0 tasking, Naive")
    # axes[1].set_title("Satellite 0 tasking, HAAL-C")
    # axes[2].set_title("Satellite 0 tasking, Greedy")
    # plt.show(block=False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT OF BENEFITS CAPTURED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # fig = plt.figure()
    # gs = fig.add_gridspec(3,1)
    # no_handover_ax = fig.add_subplot(gs[0,0])
    # greedy_ax = fig.add_subplot(gs[1,0])
    # haal_ax = fig.add_subplot(gs[2,0])
    # val_ax = fig.add_subplot(gs[:,1])

    fig, axes = plt.subplots(3,1)
    no_handover_ax = axes[0]
    greedy_ax = axes[1]
    haal_ax = axes[2]

    prev_no_handover = 0
    prev_haal = 0
    prev_greedy = 0

    no_handover_ben_line = []
    haal_ben_line = []
    greedy_ben_line = []
    
    agent_to_investigate = 900
    for k in range(T):
        no_handover_choice = np.argmax(nohand_assigns[k][agent_to_investigate,:])
        haal_choice = np.argmax(haalc_assigns[k][agent_to_investigate,:])
        greedy_choice = np.argmax(greedy_assigns[k][agent_to_investigate,:])

        if prev_no_handover != no_handover_choice:
            no_handover_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                # no_handover_ben_line.append(no_handover_ben_line[-1])
                if len(no_handover_ben_line) > 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r')
                elif len(no_handover_ben_line) == 1:
                    no_handover_ax.plot(range(k-len(no_handover_ben_line), k), no_handover_ben_line, 'r.', markersize=1)
            no_handover_ben_line = [benefits[agent_to_investigate,no_handover_choice, k]]
        else:
            no_handover_ben_line.append(benefits[agent_to_investigate, no_handover_choice, k])

        if prev_haal != haal_choice:
            vline = haal_ax.axvline(k-0.5, linestyle='--')
            vline_color = vline.get_color()
            if k != 0: 
                if len(haal_ben_line) > 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'g')
                elif len(haal_ben_line) == 1:
                    haal_ax.plot(range(k-len(haal_ben_line), k), haal_ben_line,'g.', markersize=1)
            haal_ben_line = [benefits[agent_to_investigate,haal_choice, k]]
        else:
            haal_ben_line.append(benefits[agent_to_investigate,haal_choice, k])

        if prev_greedy != greedy_choice:
            greedy_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(greedy_ben_line) > 1:
                    greedy_ax.plot(range(k-len(greedy_ben_line), k), greedy_ben_line,'b')
                elif len(greedy_ben_line) == 1:
                    greedy_ax.plot(range(k-len(greedy_ben_line), k), greedy_ben_line,'b.', markersize=1)
            greedy_ben_line = [benefits[agent_to_investigate,greedy_choice, k]]
        else:
            greedy_ben_line.append(benefits[agent_to_investigate,greedy_choice, k])

        prev_no_handover = no_handover_choice
        prev_haal = haal_choice
        prev_greedy = greedy_choice

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #plot last interval
    no_handover_ax.plot(range(k+1-len(no_handover_ben_line), k+1), no_handover_ben_line, 'r')
    haal_ax.plot(range(k+1-len(haal_ben_line), k+1), haal_ben_line,'g')
    greedy_ax.plot(range(k+1-len(greedy_ben_line), k+1), greedy_ben_line,'b')
    haal_ax.set_xlim([0, T])
    greedy_ax.set_xlim([0, T])
    no_handover_ax.set_xlim([0, T])
    haal_ax.set_xticks(range(0,T+1,10))
    greedy_ax.set_xticks(range(0,T+1,10))
    no_handover_ax.set_xticks(range(0,T+1,10))
    haal_ax.set_ylim([-0.1, 2])
    greedy_ax.set_ylim([-0.1, 2])
    no_handover_ax.set_ylim([-0.1, 2])
    haal_ax.set_ylabel("HAAL-D\nBenefit Captured")
    greedy_ax.set_ylabel("GA\nBenefit Captured")
    no_handover_ax.set_ylabel("NHA\nBenefit Captured")
    
    haal_ax.xaxis.label.set_fontsize(MEDIUM_SIZE)
    greedy_ax.xaxis.label.set_fontsize(MEDIUM_SIZE)
    no_handover_ax.xaxis.label.set_fontsize(MEDIUM_SIZE)
    haal_ax.yaxis.label.set_fontsize(MEDIUM_SIZE)
    greedy_ax.yaxis.label.set_fontsize(MEDIUM_SIZE)
    no_handover_ax.yaxis.label.set_fontsize(MEDIUM_SIZE)

    #phantom lines for legend
    # greedy_ax.plot([T+1], [0], 'r', label="No Handover")
    # greedy_ax.plot([T+1], [0], 'g', label="HAAL-D")
    haal_ax.plot([T+1], [0], color=vline_color, linestyle='--', label="Task Changes")
    haal_ax.legend(loc="upper left",bbox_to_anchor=(0,-0.15))
    # haal_ax.set_title("HAAL-D")
    # greedy_ax.set_title("Greedy")
    # no_handover_ax.set_title("No Handover")
    haal_ax.set_xlabel("Timestep")
    fig.set_figwidth(6)
    fig.set_figheight(6)
    plt.tight_layout()
    plt.savefig("haal_experiment2/paper_exp2_task_hist.pdf")
    

    #~~~~~~~~~Plot bar charts~~~~~~~~~~~~~~
    #plot value over time
    fig, axes = plt.subplots(2,1)
    labels = ("NHA", "GA", "HAAL-D")
    val_vals = (nohand_val, greedy_val, haalc_val)
    nh_vals = (nohand_nh, greedy_nh, haalc_nh)

    val_bars = axes[0].bar(labels, val_vals)
    axes[0].set_ylabel("Total Value")
    nh_bars = axes[1].bar(labels, nh_vals)
    axes[1].set_ylabel("Total Handovers")

    val_bars[0].set_color('r')
    nh_bars[0].set_color('r')
    val_bars[1].set_color('b')
    nh_bars[1].set_color('b')
    val_bars[2].set_color('g')
    nh_bars[2].set_color('g')
    fig.set_figwidth(6)
    fig.set_figheight(6)
    plt.tight_layout()
    plt.savefig("haal_experiment2/paper_exp2_bars.pdf")
    plt.show()

def lookahead_counterexample():
    benefit = np.zeros((4,4,4))
    benefit[:,:,0] = np.array([[1, 100, 1, 1],
                               [100, 1, 1, 1],
                               [1, 1, 1.99, 1],
                               [1, 1, 1, 1.99]])
    
    benefit[:,:,1] = np.array([[100, 1, 1, 1],
                               [1, 100, 1, 1],
                               [1, 1, 1.99, 1],
                               [1, 1, 1, 1.99]])
    
    benefit[:,:,2] = np.array([[1, 100, 1, 1],
                               [100, 1, 1, 1],
                               [1, 1, 1.99, 1],
                               [1, 1, 1, 1.99]])

    benefit[:,:,3] = np.array([[100, 1, 1, 1],
                               [1, 100, 1, 1],
                               [1, 1, 1.99, 1],
                               [1, 1, 1, 1.99]])

    init_assignment = np.array([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]])

    ass, opt_val, _ = solve_optimally(benefit, init_assignment, 1)
    print(opt_val)
    for a in ass:
        print(a)

    L = benefit.shape[-1]
    print("haal")
    ass, val, _ = solve_w_haal(benefit, init_assignment, 1, L)
    print(val)
    for a in ass:
        print(a)

    rat = 1/2+1/2*((L-1)/L)

    print(f"Difference: {opt_val - val}")

    print(f"Ratio: {val/opt_val}, desired rat: {rat}")

def test_accelerated():
    num_planes = 10
    num_sats = 10

    m = 100
    n = 100

    benefits, _ = get_constellation_proxs_and_graphs_random_tasks(num_planes, num_sats, m, 93)

    st = time.time()
    _, val, _ = solve_w_haal(benefits, None, 0.5, 6)
    print(f"Normal time {time.time()-st}")

    st = time.time()
    _, accel_val, _ = solve_w_accel_haal(benefits, None, 0.5, 6)
    print(f"Accelerated time {time.time()-st}")

    print(val, accel_val)

def l_compare_counterexample():
    """
    Trying to prove that a longer lookahead window is not necessarily better
    """
    benefit = np.zeros((4,4,3))
    benefit[:,:,0] = np.array([[100, 1, 1, 1],
                               [1, 100, 1, 1],
                               [1, 1, 1.5, 1],
                               [1, 1, 1, 1.5]])
    
    benefit[:,:,1] = np.array([[100, 1, 1, 1],
                               [1, 100, 1, 1],
                               [1, 1, 1.55, 1],
                               [1, 1, 1, 1.55]])
    
    benefit[:,:,2] = np.array([[100, 1, 1, 1],
                               [1, 100, 1, 1],
                               [1, 1, 1, 100],
                               [1, 1, 100, 1]])

    init_assignment = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]])

    ass, opt_val, _ = solve_optimally(benefit, init_assignment, 1)
    print("Opt",opt_val)
    for a in ass:
        print(a)

    L = benefit.shape[-1]
    ass, val1, _ = solve_w_haal(benefit, init_assignment, 1, 1)
    print("L = 1",val1)
    for a in ass:
        print(a)

    L = benefit.shape[-1]
    ass, val2, _ = solve_w_haal(benefit, init_assignment, 1, 2)
    print("L = 2",val2)
    for a in ass:
        print(a)

    rat = 1/2+1/2*((L-1)/L)

    print(opt_val, val1, val2)

def generalized_handover_fn_testing():
    with open('object_track_experiment/sat_prox_mat_large_const.pkl','rb') as f:
        sat_prox_mat = pickle.load(f)
    with open('object_track_experiment/graphs_large_const.pkl','rb') as f:
        graphs = pickle.load(f)
    with open('object_track_experiment/task_transition_scaling_large_const.pkl','rb') as f:
        task_trans_state_dep_scaling_mat = pickle.load(f)
    with open('object_track_experiment/hex_task_map_large_const.pkl','rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('object_track_experiment/const_object_large_const.pkl','rb') as f:
        const = pickle.load(f)

    lat_range = (20, 50)
    lon_range = (73, 135)
    L = 3
    lambda_ = 0.05
    T = 60
    num_objects = 50
    coverage_benefit = 1
    object_benefit = 10

    # sat_prox_mat, graphs, task_trans_state_dep_scaling_mat, hex_to_task_mapping, const = get_sat_prox_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T)
    
    np.random.seed(0)
    task_objects = init_task_objects(num_objects, const, hex_to_task_mapping, T)
    benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_prox_mat, task_objects)

    ass, tv = solve_object_track_w_dynamic_haal(sat_prox_mat, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel_approx=False,
                                                benefit_fn=timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    print(tv)

    # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_prox_mat)

    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    # ass, tv = solve_w_haal(benefits, None, lambda_, L, benefit_fn=timestep_loss_pen_benefit_fn, 
    #                        task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    # print(tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
    print(tv)
    print(is_assignment_mat_sequence_valid(ass))

    # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_prox_mat)
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    # ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
    # print(tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

def object_tracking_velocity_test():
    """
    Determine if as speed gets faster, HAAL gets more beneficial.

    VERDICT: doesn't seem to affect things dramatically.
    """
    with open('object_track_experiment/sat_prox_mat_large_const.pkl','rb') as f:
        sat_prox_mat = pickle.load(f)
    with open('object_track_experiment/graphs_large_const.pkl','rb') as f:
        graphs = pickle.load(f)
    with open('object_track_experiment/task_transition_scaling_large_const.pkl','rb') as f:
        task_trans_state_dep_scaling_mat = pickle.load(f)
    with open('object_track_experiment/hex_task_map_large_const.pkl','rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('object_track_experiment/const_object_large_const.pkl','rb') as f:
        const = pickle.load(f)
    
    lat_range = (20, 50)
    lon_range = (73, 135)
    L = 3
    lambda_ = 0.05
    T = 60
    num_objects = 50
    coverage_benefit = 1
    object_benefit = 10
    np.random.seed(0)

    haal_vals = []
    haal_pcts = []

    nohand_vals = []
    nohand_pcts = []

    greedy_vals = []
    greedy_pcts = []

    vels = range(4000, 13000, 500)
    for vel in vels:
        print(vel)
        task_objects = init_task_objects(num_objects, const, hex_to_task_mapping, T, vel*u.km/u.hr)
        benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_prox_mat, task_objects)

        # ass, tv = solve_object_track_w_dynamic_haal(sat_prox_mat, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel_approx=False,
        #                                         benefit_fn=timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
        ass, tv = solve_w_haal(benefits, None, lambda_, L, benefit_fn=timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
        pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
        haal_vals.append(tv)
        haal_pcts.append(pct)

        ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
        pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
        nohand_vals.append(tv)
        nohand_pcts.append(pct)

        ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
        pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
        greedy_vals.append(tv)
        greedy_pcts.append(pct)

    fig, axes = plt.subplots(2,1)

    axes[0].plot(vels, haal_vals, label='HAAL val')
    axes[0].plot(vels, nohand_vals, label='No Handover val')
    axes[0].plot(vels, greedy_vals, label='Greedy val')
    axes[0].legend()

    axes[1].plot(vels, haal_pcts, label='HAAL pct')
    axes[1].plot(vels, nohand_pcts, label='No Handover pct')
    axes[1].plot(vels, greedy_pcts, label='Greedy pct')
    axes[1].legend()

    plt.show()

def smaller_area_size_object_tracking():
    # with open('object_track_experiment/sat_prox_mat_highres.pkl','rb') as f:
    #     sat_prox_mat = pickle.load(f)
    # with open('object_track_experiment/graphs_highres.pkl','rb') as f:
    #     graphs = pickle.load(f)
    # with open('object_track_experiment/task_transition_scaling_highres.pkl','rb') as f:
    #     task_trans_state_dep_scaling_mat = pickle.load(f)
    # with open('object_track_experiment/hex_task_map_highres.pkl','rb') as f:
    #     hex_to_task_mapping = pickle.load(f)
    # with open('object_track_experiment/const_object_highres.pkl','rb') as f:
    #     const = pickle.load(f)

    # with open('object_track_experiment/sat_prox_mat_usa.pkl','rb') as f:
    #     sat_prox_mat = pickle.load(f)
    # with open('object_track_experiment/graphs_usa.pkl','rb') as f:
    #     graphs = pickle.load(f)
    # with open('object_track_experiment/task_transition_scaling_usa.pkl','rb') as f:
    #     task_trans_state_dep_scaling_mat = pickle.load(f)
    # with open('object_track_experiment/hex_task_map_usa.pkl','rb') as f:
    #     hex_to_task_mapping = pickle.load(f)
    # with open('object_track_experiment/const_object_usa.pkl','rb') as f:
    #     const = pickle.load(f)

    lat_range = (22, 52)
    lon_range = (-124.47, -66.87)
    L = 3
    lambda_ = 0.05
    T = 60
    num_objects = 50
    coverage_benefit = 1
    object_benefit = 10

    sat_prox_mat, graphs, task_trans_state_dep_scaling_mat, hex_to_task_mapping, const = get_sat_prox_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T)

    np.random.seed(42)
    task_objects = init_task_objects(num_objects, const, hex_to_task_mapping, T, velocity=10000*u.km/u.hr)
    benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_prox_mat, task_objects)

    benefit_info = BenefitInfo()
    benefit_info.T_trans = task_trans_state_dep_scaling_mat

    print("Dynamic HAAL, Centralized")
    ass, tv = solve_object_track_w_dynamic_haal(sat_prox_mat, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel=False,
                                                benefit_fn=timestep_loss_pen_benefit_fn, benefit_info=benefit_info)
    print("Value", tv)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct", pct)
    plot_object_track_scenario(hex_to_task_mapping, sat_prox_mat, task_objects, ass, task_trans_state_dep_scaling_mat,
                               "haal_object_track.gif", show=False)

    print("Dynamic HAAL, Distributed")
    ass, tv = solve_object_track_w_dynamic_haal(sat_prox_mat, task_objects, coverage_benefit, object_benefit, None, lambda_, L, distributed=True, graphs=graphs,
                                                benefit_fn=timestep_loss_pen_benefit_fn, benefit_info=benefit_info, verbose=True)
    print("Value", tv)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct", pct)

    # print("Normal HAAL")
    # ass, tv = solve_w_haal(benefits, None, lambda_, L, benefit_fn=timestep_loss_pen_benefit_fn, 
    #                        task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    # print("Value", tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print("Pct",pct)

    print("No handover")
    ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
    print("Value", tv)
    print(is_assignment_mat_sequence_valid(ass))
    plot_object_track_scenario(hex_to_task_mapping, sat_prox_mat, task_objects, ass, task_trans_state_dep_scaling_mat,
                               "nha_object_track.gif", show=False)

    # # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_prox_mat)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct",pct)

    # print("Greedy")
    # ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_pen_benefit_fn, task_trans_state_dep_scaling_mat)
    # print("Value", tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print("pct",pct)

def paper_experiment1():
    num_planes = 18
    num_sats_per_plane = 18
    m = 450
    T = 93

    altitude = 550
    fov = 60
    timestep = 1*u.min

    max_L = test_optimal_L(timestep, altitude, fov)
    print(max_L)
    
    lambda_ = 0.5

    # benefits, graphs = get_constellation_proxs_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_based_proximities, fov=fov, isl_dist=2500)

    # with open("haal/haal_experiment1/paper_exp1_bens.pkl", 'wb') as f:
    #     pickle.dump(benefits,f)
    # with open("haal/haal_experiment1/paper_exp1_graphs.pkl", 'wb') as f:
    #     pickle.dump(graphs,f)

    with open("haal/haal_experiment1/paper_exp1_bens.pkl", 'rb') as f:
        benefits = pickle.load(f)
    with open("haal/haal_experiment1/paper_exp1_graphs.pkl", 'rb') as f:
        graphs = pickle.load(f)

    env = SimpleAssignEnv(benefits, None, lambda_)
    _, no_handover_val = solve_wout_handover(env)
    print(no_handover_val)

    # _, cbba_val = solve_w_centralized_CBBA(benefits, None, lambda_, max_L, verbose=True)
    cbba_val = 0

    env.reset()
    _, greedy_val = solve_greedily(env)
    print(greedy_val)
    itersd_by_lookahead = []
    valued_by_lookahead = []

    iterscbba_by_lookahead = []
    valuecbba_by_lookahead = []

    valuec_by_lookahead = []
    for L in range(1,max_L+1):
        print(f"lookahead {L}")
        # _, cbba_val, avg_iters = solve_w_CBBA_track_iters(benefits, None, lambda_, L, graphs=graphs, verbose=True)
        # iterscbba_by_lookahead.append(avg_iters)
        # valuecbba_by_lookahead.append(cbba_val)
        
        env.reset()
        _, d_val, avg_iters = solve_w_haal(env, L, graphs=graphs, verbose=True, track_iters=True, distributed=True)
        print(d_val, avg_iters)
        itersd_by_lookahead.append(avg_iters)
        valued_by_lookahead.append(d_val)

        env.reset()
        _, c_val = solve_w_haal(env, L, distributed=False, verbose=True)
        print(c_val)
        valuec_by_lookahead.append(c_val)

    # #Values from 1/31, before scaling experiments
    valuecbba_by_lookahead = [4208.38020192484, 4412.873727755446, 4657.90330919782, 4717.85859678172, 4710.212483240204, 4726.329218229788]
    # valuec_by_lookahead = [6002.840671517548, 7636.731195199751, 7581.29374466441, 7435.882168254755, 7511.4534257400755, 7591.261917337481]
    # itersd_by_lookahead = [54.89247311827957, 61.17204301075269, 68.46236559139786, 72.64516129032258, 79.10752688172043, 80.02150537634408]
    iterscbba_by_lookahead = [18.50537634408602, 26.0, 29.193548387096776, 33.11827956989247, 34.806451612903224, 37.29032258064516]
    # valued_by_lookahead = [6021.705454081699, 7622.246684035546, 7585.4110847804, 7294.093230272816, 7437.211996201664, 7559.402984912062]
    greedy_val = 3650.418056196203
    no_handover_val = 4078.018608711949

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    
    fig, axes = plt.subplots(2,1,sharex=True)
    for ax in axes:
        ax.grid(True)
    axes[0].plot(range(1,max_L+1), valued_by_lookahead, 'g--', label="HAAL-D")
    axes[0].plot(range(1,max_L+1), valuec_by_lookahead, 'g', label="HAAL")
    axes[0].plot(range(1,max_L+1), valuecbba_by_lookahead, 'b', label="CBBA")
    axes[0].plot(range(1,max_L+1), [no_handover_val]*max_L, 'r', label="NHA")
    axes[0].plot(range(1,max_L+1), [greedy_val]*max_L, 'k', label="GA")
    axes[0].set_ylabel("Total Value")
    axes[0].set_xticks(range(1,max_L+1))
    axes[0].set_ylim((0, 1.1*max(valuec_by_lookahead)))
    # axes[0].legend(loc='upper left', bbox_to_anchor=(1, 0.25))
    plt.subplots_adjust(right=0.75)
    # axes[0].set_aspect('equal', adjustable='box')
    # axes[0].yaxis.label.set_fontsize(12)
    handles, labels = [], []
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.15, 0.7))
    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.7, 0.3))

    axes[1].plot(range(1,max_L+1), itersd_by_lookahead, 'g--', label="HAAL-D")
    axes[1].plot(range(1,max_L+1), iterscbba_by_lookahead, 'b', label="CBBA")
    axes[1].set_ylim((0, 1.1*max(itersd_by_lookahead)))
    axes[0].set_xticks(range(1,max_L+1))
    axes[1].set_ylabel("Average # Communications\nUntil All Auctions Converge")
    axes[1].set_xlabel("Lookahead Window (L)")
    axes[0].set_xlim(1,6)
    axes[1].set_xlim(1,6)

    with open("haal_experiment1/results.txt", 'w') as f:
        f.write(f"num_planes: {num_planes}, num_sats_per_plane: {num_sats_per_plane}, m: {m}, T: {T}, altitude: {altitude}, fov: {fov}, timestep: {timestep}, max_L: {max_L}, lambda: {lambda_}\n")
        f.write(f"~~~~~~~~~~~~~~~~~~~~~\n")
        f.write(f"No Handover Value: {no_handover_val}\n")
        f.write(f"Greedy Value: {greedy_val}\n")
        f.write(f"CBBA Values by lookahead:\n{valuecbba_by_lookahead}\n")
        f.write(f"HAAL Values by lookahead:\n{valuec_by_lookahead}\n")
        f.write(f"HAAL-D Values by lookahead:\n{valued_by_lookahead}\n")

        f.write(f"CBBA Iters by lookahead:\n{iterscbba_by_lookahead}\n")
        f.write(f"HAAL Iters by lookahead:\n{itersd_by_lookahead}\n")

    fig.set_figwidth(6)
    fig.set_figheight(6)
    fig.tight_layout()
    plt.savefig("haal_experiment1/paper_exp1.pdf")
    plt.show()

def scaling_experiment():
    with open("haal_experiment1/paper_exp1_bens.pkl", 'rb') as f:
        benefits = pickle.load(f)
    with open("haal_experiment1/paper_exp1_graphs.pkl", 'rb') as f:
        graphs = pickle.load(f)

    # lambdas = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    lambdas_over_bens = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    max_L = 6
    avg_pass_ben = 1.1062110048940414
    avg_pass_len = 4.2706495009402582

    # haal_ass_lengths = []
    # haal_benefit_captured = []
    # for L in range(1, max_L+1):
    #     ass_lengths = []
    #     ben_captured = []
    #     for lambda_over_ben in lambdas_over_bens:
    #         print(f"\nL: {L}, lambda_over_ben: {lambda_over_ben}")

    #         assigns, _ = solve_w_haal(benefits, None, lambda_over_ben/avg_pass_ben, L, verbose=True)

    #         _, _, avg_ass_len = calc_pass_statistics(benefits, assigns)
    #         print(avg_pass_len)
    #         ass_lengths.append(avg_ass_len)

    #         ben, _ = calc_value_and_num_handovers(assigns, benefits, None, 0)
    #         ben_captured.append(ben)
        
    #     haal_ass_lengths.append(ass_lengths)
    #     haal_benefit_captured.append(ben_captured)

    haal_ass_lengths = [[1.975839717147908, 2.6321243523316062, 2.9826032929481205, 3.1127060074428496, 3.167612787570959, 3.166237776634071, 3.1564245810055866, 3.136138613861386], [1.975839717147908, 2.6321243523316062, 3.1437713310580206, 3.518816067653277, 3.6521591295477727, 3.74131998748827, 3.7848474909806495, 3.7702182284980745], [1.975839717147908, 2.6321243523316062, 3.0977611940298506, 3.674305033809166, 4.029679461812425, 4.1662279052275295, 4.268018833755885, 4.348904060366511], [1.975839717147908, 2.6321243523316062, 3.0945330296127564, 3.460616438356164, 4.037037037037037, 4.361411087113031, 4.42348623853211, 4.5418569254185694], [1.975839717147908, 2.6321243523316062, 3.0975700934579438, 3.583904727535186, 3.8355240984159082, 4.2120644436118395, 4.43562066306862, 4.647544204322201], [1.975839717147908, 2.6321243523316062, 3.0907046476761617, 3.6238130021913806, 3.9195156695156697, 4.017934241115908, 4.485096870342772, 4.621862348178138]]
    haal_benefit_captured = [[13948.722593405377, 13736.335016158991, 12814.189934221762, 10482.412016961945, 7333.355671855289, 3685.7664616188877, 1248.5513620437603, 623.2249270689351], [13948.722593405377, 13736.335016158991, 13034.873746423451, 12666.10491127586, 11615.043938933559, 10115.646076312827, 8208.367489747627, 5567.898557073061], [13948.722593405377, 13736.335016158991, 13008.183567290924, 11993.167521089039, 11495.584504412045, 10923.505283370283, 10103.6032538031, 8672.164995560246], [13948.722593405377, 13736.335016158991, 13019.51058647498, 11976.573334230057, 10809.450638906772, 10206.05387952314, 9753.063501644381, 9101.618507857653], [13948.722593405377, 13736.335016158991, 13016.36301038427, 12046.386026305474, 11021.742093991927, 9642.197338315971, 9225.830354891794, 8730.66387472196], [13948.722593405377, 13736.335016158991, 13018.50355117942, 12008.943137690649, 11366.454920955044, 10226.098839775805, 8628.681268625438, 8073.396674187025]]
    # avg_pass_ben = 1.1062110048940414

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(2,1, sharex=True)
    for ax in axes:
        ax.grid(True)
    
    viridis_cmap = plt.get_cmap('viridis')
    for L_idx in range(max_L):
        color = viridis_cmap(L_idx /(max_L-1))
        axes[0].plot(lambdas_over_bens, haal_ass_lengths[L_idx], label=f"L={L_idx+1}", color=color)
        axes[1].plot(lambdas_over_bens, haal_benefit_captured[L_idx], label=f"L={L_idx+1}", color=color)
    axes[0].plot(lambdas_over_bens, [avg_pass_len]*len(lambdas_over_bens), 'k--', label="Avg. Time Task\nin View "+r'($P$)')

    axes[0].set_ylabel("Avg. Time Satellite\nAssigned to Same Task "+r'($P^{\mathbf{x}}$)')
    axes[1].set_ylabel("Total Benefit")
    axes[1].set_xlabel(r'$\frac{\lambda}{P^\beta}$')
    axes[1].xaxis.label.set_fontsize(19.5)
    axes[0].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))
    axes[1].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))

    handles, labels = [], []
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
    
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.4,0.5), ncol=2)
    # axes[0].legend(loc='upper left', bbox_to_anchor=(1, 0.3))

    with open("haal_scaling_exp/results.txt", 'w') as f:
        f.write(f"same constellation stats as exp 1")
        f.write(f"~~~~~~~~~~~~~~~~~~~~~\n")
        f.write(f"Lambdas_over_bens: {lambdas_over_bens}\n")
        f.write(f"avg pass ben: {avg_pass_ben}\n")
        f.write(f"avg pass len: {avg_pass_len}\n")
        f.write(f"HAAL assignment lengths:\n{haal_ass_lengths}\n")
        f.write(f"HAAL benefit captured:\n{haal_benefit_captured}\n")

    fig.set_figwidth(6)
    fig.set_figheight(6)
    # plt.subplots_adjust(right=0.75)
    plt.tight_layout()
    plt.savefig("haal_scaling_exp/paper_scaling.pdf")
    plt.show()

# def auction_speedup_test():
#     """
#     Was gonna test if we could speed up auctions by using edited
#     prices from previous timesteps, but decided it was kinda a stupid
#     idea in the end.
#     """
#     with open("haal_experiment1/paper_exp1_bens.pkl", 'rb') as f:
#         benefits = pickle.load(f)
#     with open("haal_experiment1/paper_exp1_graphs.pkl", 'rb') as f:
#         graphs = pickle.load(f)

#     n = benefits.shape[0]
#     m = benefits.shape[1]
#     T = benefits.shape[2]
#     lambda_ = 0.5

#     auction = Auction(n, m, benefits=benefits[:,:,0], graph=graphs[0])
#     auction.run_auction()
#     init_assigns = convert_agents_to_assignment_matrix(auction.agents)
#     timestep_1_prices = auction.agents[0].public_prices
    
#     benefits1 = add_handover_pen_to_benefit_matrix(np.expand_dims(benefits[:,:,1],-1), init_assigns, lambda_)
#     benefits1 = np.squeeze(benefits1)
#     #Run default auction for second timestep
#     def_auction2 = Auction(n, m, benefits=benefits1, graph=graphs[1])
#     def_auction2.run_auction()

#     def_iters = def_auction2.n_iterations
#     print(f"Iterations without speedup {def_iters}")

#     #Run default auction for second timestep
#     def_auction2 = Auction(n, m, benefits=benefits1, graph=graphs[1], prices=timestep_1_prices)
#     def_auction2.run_auction()

#     def_iters = def_auction2.n_iterations
#     print(f"Iterations with old prices speedup {def_iters}")

#     #Run sped up auction
#     for j in range(m):
#         prev_benefit = np.sum(benefits[:,j,0])
#         total_benefit = np.sum(benefits1[:,j])
#         print(total_benefit - prev_benefit)

def multi_task_test():
    lat_range = (20, 50)
    lon_range = (73, 135)
    T = 60
    # full_benefit_matrix_w_synthetic_sats, graphs, T_trans, A_eqiv, \
    #     _, _, tracked_lats, tracked_lons = get_benefit_matrix_and_graphs_multitask_area(lat_range, lon_range, T)

    with open('multitask_experiment/benefit_matrix.pkl','rb') as f:
        full_benefit_matrix_w_synthetic_sats = pickle.load(f)
    with open('multitask_experiment/graphs.pkl','rb') as f:
        graphs = pickle.load(f)
    with open('multitask_experiment/hex_task_map.pkl','rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('multitask_experiment/const_object.pkl','rb') as f:
        const = pickle.load(f)
    with open('multitask_experiment/T_trans.pkl','rb') as f:
        T_trans = pickle.load(f)
    with open('multitask_experiment/A_eqiv.pkl','rb') as f:
        A_eqiv = pickle.load(f)
    with open('multitask_experiment/sat_to_track.pkl','rb') as f:
        sat_to_track = pickle.load(f)   

    benefit_info = BenefitInfo()
    benefit_info.T_trans = T_trans
    benefit_info.A_eqiv = A_eqiv

    lambda_ = 0.2
    print("lambda", lambda_)

    ass, tv = solve_multitask_w_haal(full_benefit_matrix_w_synthetic_sats, None, lambda_, 3, distributed=False, verbose=True, benefit_info=benefit_info)
    print("haal",tv)

    plot_multitask_scenario(hex_to_task_mapping, ass, A_eqiv, "haal_multitask.gif", show=False)

    #Pad benefit matrices
    n = full_benefit_matrix_w_synthetic_sats.shape[0]
    m = full_benefit_matrix_w_synthetic_sats.shape[1]
    T = full_benefit_matrix_w_synthetic_sats.shape[2]
    padded_size = max(n,m)
    padded_benefits = np.zeros((n,padded_size, T))
    padded_benefits[:,:m,:] = full_benefit_matrix_w_synthetic_sats

    ass, tv = solve_wout_handover(padded_benefits, None, lambda_, benefit_fn=calc_multiassign_benefit_fn, benefit_info=benefit_info)
    print("nha",tv)

    plot_multitask_scenario(hex_to_task_mapping, ass, A_eqiv, "nha_multitask.gif", show=False)

    # ass, tv = solve_greedily(padded_benefits, None, lambda_, benefit_fn=calc_multiassign_benefit_fn, benefit_info=benefit_info)
    # print("greedy",tv)

def proof_verification_with_full_information():
    L = 5
    T = 5
    n = 50
    m = 50
    lambda_ = 0.5

    total_sequences = 0
    observed_equal = 0
    for _ in range(100):
        print(f"Experiment {_}")
        benefits = np.random.rand(n,m,T)

        base_ass, _ = solve_w_haal(benefits, None, lambda_, T)
        curr_ass = base_ass
        for k in range(1,T):
            new_ass, _ = solve_w_haal(benefits[:,:,k:], curr_ass.pop(0), lambda_, T-k)
            
            total_sequences += len(new_ass)
            equal_array = [int(np.array_equal(new_ass[i], curr_ass[i])) for i in range(len(new_ass))]
            observed_equal += sum(equal_array)

            curr_ass = new_ass

    print(f"Sequences which were identical: {observed_equal}/{total_sequences}")

def num_tasks_per_satellite():
    with open("haal/haal_experiment2/paper_exp2_bens.pkl", 'rb') as f:
        benefits = pickle.load(f)

    n = benefits.shape[0]
    m = benefits.shape[1]

    total_tasks = 0
    for i in range(n):
        ben_slice = benefits[i,:,:6]

        total_bens = np.sum(ben_slice, axis=1)
        
        available_tasks = np.where(total_bens > 0, 1, 0)
        total_tasks += np.sum(available_tasks)
    
    print(total_tasks/n)

if __name__ == "__main__":
    paper_experiment1()