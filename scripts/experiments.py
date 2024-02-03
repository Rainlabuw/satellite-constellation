import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from tqdm import tqdm
import time
import pickle

from methods import *

from solve_optimally import solve_optimally
from solve_wout_handover import solve_wout_handover
from solve_w_haal import solve_w_haal
from solve_w_accelerated_haal import solve_w_accel_haal
from solve_w_centralized_CBBA import solve_w_centralized_CBBA
from solve_w_CBBA import solve_w_CBBA, solve_w_CBBA_track_iters
from solve_greedily import solve_greedily
from classic_auction import Auction
from object_track_scenario import timestep_loss_state_dep_fn, init_task_objects, get_benefits_from_task_objects, solve_object_track_w_dynamic_haal, get_sat_coverage_matrix_and_graphs_object_tracking_area
from object_track_utils import calc_pct_objects_tracked, object_tracking_history
from plotting_utils import plot_object_track_scenario

from constellation_sim.ConstellationSim import get_constellation_bens_and_graphs_random_tasks, get_constellation_bens_and_graphs_coverage, ConstellationSim
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

    num_avgs = 50
    for _ in tqdm(range(num_avgs)):
        benefit = np.random.rand(n,m,T)

        #SMHGL centralized, lookahead = 3
        _, haal_ben, _ = solve_w_haal(benefit, init_ass, lambda_, L)
        avg_haal += haal_ben/num_avgs

        #MHA
        _, mha_ben, _ = solve_w_haal(benefit, init_ass, lambda_, 1)
        avg_mha += mha_ben/num_avgs

        #Naive
        _, ben, _ = solve_wout_handover(benefit, init_ass, lambda_)
        avg_no_handover += ben/num_avgs

        #Optimal
        _, ben, _ = solve_optimally(benefit, init_ass, lambda_)
        avg_best += ben/num_avgs

    fig = plt.figure()
    plt.bar(["Standard Assignment\nProblem Solution","HAA", f"HAAL (L={L})", "Optimal"], [avg_no_handover, avg_mha, avg_haal, avg_best])
    plt.ylabel("Value")
    print(["No Handover","HAA", f"HAAL (L={L})", "Optimal"])
    print([avg_no_handover, avg_mha, avg_haal, avg_best])
    plt.savefig("opt_comparison.png")
    plt.show()

def MHA_unit_testing():
    """
    Tests MHA in various cases where solutions are known.
    """
    # Case where no solutions are the best.
    benefits = np.zeros((4,4,2))
    benefits[:,:,0] = np.array([[100, 1, 0, 0],
                                [1, 100, 0, 0],
                                [0, 0, 0.2, 0.1],
                                [0, 0, 0.1, 0.2]])
    
    benefits[:,:,1] = np.array([[1, 1000, 0, 0],
                                [1000, 1, 0, 0],
                                [0, 0, 0.1, 0.3],
                                [0, 0, 0.3, 0.1]])

    print("Expect no solution to be optimal (2198.8) but them to be same for all lookaheads")
    for lookahead in range(1,benefits.shape[-1]+1):
        multi_auction = HAAL_Auction(benefits, None, lookahead)
        multi_auction.run_auctions()
        ben = multi_auction.calc_value_and_num_handovers()
        print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

    #Case where a combined solution is the best.
    benefits = np.zeros((3,3,3))
    benefits[:,:,0] = np.array([[0.1, 0, 0],
                                [0, 0.1, 0],
                                [0, 0, 0.1]])
    benefits[:,:,1] = np.array([[0, 0, 0.1],
                                [0.1, 0, 0],
                                [0, 0.1, 0]])
    benefits[:,:,2] = np.array([[0.1, 1000, 0],
                                [0, 0.1, 1000],
                                [1000, 0, 0.1]])

    print("Expect combined solution to be optimal (3000) only at lookahead of 3")
    for lookahead in range(1,benefits.shape[-1]+1):
        multi_auction = HAAL_Auction(benefits, None, lookahead)
        multi_auction.run_auctions()
        ben,_ = multi_auction.calc_value_and_num_handovers()
        print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

def compare_MHA_to_other_algs():
    #Case where we expect solutions to get increasingly better as lookahead window increases
    n = 50
    m = 50
    T = 95
    lambda_ = 1

    print("Comparing performance of HAAL to other algorithms")
    num_avgs = 10

    haal2_ben = 0
    haal2_nh = 0

    haal5_ben = 0
    haal5_nh = 0

    mha_ben = 0
    mha_nh = 0

    no_handover_ben = 0
    no_handover_nh = 0

    sga_ben = 0
    sga_nh = 0
    for _ in range(num_avgs):
        print(f"Run {_}/{num_avgs}")
        # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
        benefits, _ = get_benefits_and_graphs_from_constellation(n, m, T)
        print("Generated realistic benefits")

        #SMHGL centralized, lookahead = 2
        multi_auction = HAAL_Auction(benefits, None, 2, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        haal2_ben += ben/num_avgs
        haal2_nh += nh/num_avgs

        #SMHGL centralized, lookahead = 5
        multi_auction = HAAL_Auction(benefits, None, 5, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        haal5_ben += ben/num_avgs
        haal5_nh += nh/num_avgs

        #MHA
        multi_auction = HAAL_Auction(benefits, None, 1, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        mha_ben += ben/num_avgs
        mha_nh += nh/num_avgs

        #Naive
        _, ben, nh = solve_wout_handover(benefits, lambda_)
        no_handover_ben += ben/num_avgs
        no_handover_nh += nh/num_avgs

        #SGA/CBBA
        handover_ben, _, handover_nh = solve_w_centralized_CBBA(benefits, lambda_)

        sga_ben += handover_ben/num_avgs
        sga_nh += handover_nh/num_avgs

    fig, axes = plt.subplots(2,1)
    axes[0].bar(["Naive", "SGA", "MHA", "HAAL2", "HAAL5"],[no_handover_ben, sga_ben, mha_ben, haal2_ben, haal5_ben])
    axes[0].set_title("Average benefit across 10 runs")
    # axes[0].set_xlabel("Lookahead timesteps")

    axes[1].bar(["Naive", "SGA", "MHA", "HAAL2", "HAAL5"],[no_handover_nh, sga_nh, mha_nh, haal2_nh, haal5_nh])
    
    axes[1].set_title("Average number of handovers across 10 runs")
    fig.suptitle(f"Test with n={n}, m={m}, T={T}, lambda={lambda_}, realistic-ish benefits")

    plt.show()

def test_MHA_lookahead_performance():
    """
    Test performance of MHA as lookahead window increases.

    Hopefully, general trend is that performance increases as lookahead increases.
    """
    print("Expect performance to generally increase as lookahead increases")
    n = 36*15
    m = 300
    T = 93
    lambda_ = 1

    max_lookahead = 6
    num_avgs = 1

    resulting_bens = []
    resulting_approx_bens = []
    handovers = []
    benefits, graphs = get_constellation_bens_and_graphs_random_tasks(36, 15, m, T, altitude=500)
    m = benefits.shape[1]

    init_assignment = np.eye(n,m)
    for lookahead in range(1,max_lookahead+1):
        avg_ben = 0
        avg_nh = 0
        avg_approx_ben = 0
        for _ in range(num_avgs):
            print(f"Lookahead {lookahead} ({_}/{num_avgs})", end='\r')
            # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
            # benefits, graphs = get_benefits_and_graphs_from_constellation(10, 10, m, T, altitude=500)
            # benefits = np.random.rand(n,m,T)

            #HAAL with true lookaheads
            _, ben, nh = solve_w_haal(benefits, init_assignment, lambda_, lookahead, distributed=False, verbose=True)
            avg_ben += ben/num_avgs
            avg_nh += nh/num_avgs
            
            # #HAAL (distributed)
            # _, ben, nh = solve_w_haal(benefits, init_assignment, lambda_, lookahead, distributed=True)
            # avg_approx_ben += ben/num_avgs

        resulting_bens.append(avg_ben)
        # resulting_approx_bens.append(avg_approx_ben)
        handovers.append(avg_nh)

    plt.plot(range(1,max_lookahead+1), resulting_bens, label="HAAL (Centralized)")
    # plt.plot(range(1,max_lookahead+1), resulting_approx_bens, label="HAAL (Distributed)")
    plt.title(f"Lookahead vs. accuracy, n={n}, m={m}, T={T}")
    plt.xlabel("Lookahead timesteps")
    plt.ylabel(f"Average benefit across {num_avgs} runs")
    plt.legend()
    plt.savefig("lookahead_vs_benefit.png")
    plt.show()

    plt.figure()
    plt.plot(range(1,max_lookahead+1), handovers)
    plt.title("Num handovers vs. lookahead")
    plt.show()

def compare_alg_benefits_and_handover():
    """
    As number of agents increase, how do the benefits
    and number of handovers increase?
    """
    ns = [10, 25]
    # ns = [10, 15, 20]
    t_final = 50
    T = 25
    lambda_ = 0.5

    no_handover_benefits = []
    no_handover_handover_benefits = []
    no_handover_handover_violations = []

    sequential_benefits = []
    sequential_handover_benefits = []
    sequential_handover_violations = []

    sga_benefits = []
    sga_handover_benefits = []
    sga_handover_violations = []
    for n in ns:
        print(f"AGENT {n}")
        m = n
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        # np.random.seed(29)
        print(f"Seed {seed}")
        print(f"n: {n}, m: {m}, T: {T}, lambda: {lambda_}")
        graph = nx.complete_graph(n)
        benefit_mats_over_time = np.random.rand(n,m,T)
        
        # benefit_mats_over_time = generate_benefits_over_time(n, m, T, t_final)
        #Add 2 lambda_+eps to the benefit matrix to ensure that it's always positive to complete
        #a task.
        # benefit_mats_over_time += 2*lambda_ + 0.01


        #solve each timestep independently
        assignment_mats = []
        benefits = []
        for k in range(T):
            print(k, end='\r')
            a = Auction(n, m, benefits=benefit_mats_over_time[:,:,k], graph=graph)
            benefit = a.run_auction()

            assignment_mat = convert_agents_to_assignment_matrix(a.agents)
            assignment_mats.append(assignment_mat)

            benefits.append(benefit)
        
        handover_ben = sum(benefits) + calc_assign_seq_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep independently")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        no_handover_benefits.append(sum(benefits))
        no_handover_handover_benefits.append(handover_ben)
        no_handover_handover_violations.append(calc_assign_seq_handover_penalty(assignment_mats, lambda_))

        #solve each timestep sequentially
        assignment_mats = []
        benefits = []
        
        #solve first timestep separately
        a = Auction(n, m, benefits=benefit_mats_over_time[:,:,0], graph=graph)
        benefit = a.run_auction()

        assignment_mat = convert_agents_to_assignment_matrix(a.agents)
        assignment_mats.append(assignment_mat)
        benefits.append(benefit)

        prev_assignment_mat = assignment_mat
        for k in range(1, T):
            print(k, end='\r')
            #Generate assignment for the task minimizing handover
            benefit_mat_w_handover = add_handover_pen_to_benefit_matrix(benefit_mats_over_time[:,:,k], prev_assignment_mat, lambda_)

            a = Auction(n, m, benefits=benefit_mat_w_handover, graph=graph)
            a.run_auction()
            choices = [ag.choice for ag in a.agents]

            assignment_mat = convert_agents_to_assignment_matrix(a.agents)
            assignment_mats.append(assignment_mat)

            prev_assignment_mat = assignment_mat

            #Calculate the benefits from a task with the normal benefit matrix
            benefit = benefit_mats_over_time[:,:,k]*assignment_mat

            benefits.append(benefit.sum())

        handover_ben = sum(benefits) + calc_assign_seq_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep considering the last one")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        sequential_benefits.append(sum(benefits))
        sequential_handover_benefits.append(handover_ben)

        #solve each timestep sequentially with greedy
        _, sg_assignment_mats, _ = solve_w_centralized_CBBA(benefit_mats_over_time, lambda_)
        sg_benefit = 0
        for k, sg_assignment_mat in enumerate(sg_assignment_mats):
            sg_benefit += (benefit_mats_over_time[:,:,k]*sg_assignment_mat).sum()

        handover_ben = sg_benefit + calc_assign_seq_handover_penalty(sg_assignment_mats, lambda_)

        print("Solving with greedy algorithm")
        print(f"\tBenefit without considering handover: {sg_benefit}")
        print(f"\tBenefit with handover penalty: {handover_ben}")
    
        sga_benefits.append(sg_benefit)
        sga_handover_benefits.append(handover_ben)

    print("done")
    print(no_handover_benefits)
    print(no_handover_handover_benefits)
    print(sequential_benefits)
    print(sequential_handover_benefits)
    print(sga_benefits)
    print(sga_handover_benefits)


    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Top subplot
    axs[0].set_title('Benefits without handover penalty')
    axs[0].set_xlabel('Number of agents')
    axs[0].set_ylabel('Total benefit')
    axs[0].bar(np.arange(len(no_handover_benefits)), no_handover_benefits, width=0.2, label='Naive')
    axs[0].bar(np.arange(len(sga_benefits))+0.2, sga_benefits, width=0.2, label='SGA')
    axs[0].bar(np.arange(len(sequential_benefits))+0.4, sequential_benefits, width=0.2, label='MHA (Ours)')
    axs[0].set_xticks(np.arange(len(no_handover_benefits)))
    axs[0].set_xticklabels([str(n) for n in ns])
    axs[0].legend(loc='lower center')

    # Bottom subplot
    axs[1].set_title('Total Benefits, including handover penalty')
    axs[1].set_xlabel('Number of agents')
    axs[1].set_ylabel('Average Benefit')
    axs[1].bar(np.arange(len(no_handover_handover_benefits)), no_handover_handover_benefits, width=0.2, label='Naive')
    axs[1].bar(np.arange(len(sga_handover_benefits))+0.2, sga_handover_benefits, width=0.2, label='CBBA')
    axs[1].bar(np.arange(len(sequential_handover_benefits))+0.4, sequential_handover_benefits, width=0.2, label='MHA (Ours)')
    axs[1].set_xticks(np.arange(len(no_handover_handover_benefits)))
    axs[1].set_xticklabels([str(n) for n in ns])
    #add a legend to the bottom middle of the subplot
    axs[1].legend(loc='lower center')

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
        # benefits, graphs = get_constellation_bens_and_graphs_random_tasks(num_planes,num_sats_per_plane,m,T, benefit_func=calc_distance_based_benefits)
        benefits, graphs = get_constellation_bens_and_graphs_coverage(num_planes,num_sats_per_plane,T,70,benefit_func=calc_distance_based_benefits)

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
            _, d_val, _, avg_iters = solve_w_haald_track_iters(benefits, init_assignment, lambda_, L, graphs=graphs, verbose=True)

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
        benefits, graphs = get_constellation_bens_and_graphs_random_tasks(10,5,m,T)

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

    # benefits, graphs = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_benefits)

    # benefits, graphs = get_constellation_bens_and_graphs_coverage(num_planes,num_sats_per_plane,T,5, altitude=altitude, benefit_func=calc_fov_benefits)

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
    
    L = generate_safe_L(timestep, sat)
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
    #     _, graphs = get_constellation_bens_and_graphs_random_tasks(nm, nm, 1, T, isl_dist=isl_dist)

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

    _, graphs = get_constellation_bens_and_graphs_random_tasks(18, 18, 1, 93, isl_dist=2500)
    pct_connected = sum([nx.is_connected(graph) for graph in graphs])/93
    print(pct_connected)

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

    benefits, _ = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats, m, 93)

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
    with open('object_track_experiment/sat_cover_matrix_large_const.pkl','rb') as f:
        sat_cover_matrix = pickle.load(f)
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

    # sat_cover_matrix, graphs, task_trans_state_dep_scaling_mat, hex_to_task_mapping, const = get_sat_coverage_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T)
    
    np.random.seed(0)
    task_objects = init_task_objects(num_objects, const, hex_to_task_mapping, T)
    benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_cover_matrix, task_objects)

    ass, tv = solve_object_track_w_dynamic_haal(sat_cover_matrix, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel_approx=False,
                                                state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    print(tv)

    # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_cover_matrix)

    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    # ass, tv = solve_w_haal(benefits, None, lambda_, L, state_dep_fn=timestep_loss_state_dep_fn, 
    #                        task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    # print(tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
    print(tv)
    print(is_assignment_mat_sequence_valid(ass))

    # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_cover_matrix)
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

    # ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
    # print(tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print(pct)

def object_tracking_velocity_test():
    """
    Determine if as speed gets faster, HAAL gets more beneficial.

    VERDICT: doesn't seem to affect things dramatically.
    """
    with open('object_track_experiment/sat_cover_matrix_large_const.pkl','rb') as f:
        sat_cover_matrix = pickle.load(f)
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
        benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_cover_matrix, task_objects)

        # ass, tv = solve_object_track_w_dynamic_haal(sat_cover_matrix, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel_approx=False,
        #                                         state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
        ass, tv = solve_w_haal(benefits, None, lambda_, L, state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
        pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
        haal_vals.append(tv)
        haal_pcts.append(pct)

        ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
        pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
        nohand_vals.append(tv)
        nohand_pcts.append(pct)

        ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
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
    with open('object_track_experiment/sat_cover_matrix_highres.pkl','rb') as f:
        sat_cover_matrix = pickle.load(f)
    with open('object_track_experiment/graphs_highres.pkl','rb') as f:
        graphs = pickle.load(f)
    with open('object_track_experiment/task_transition_scaling_highres.pkl','rb') as f:
        task_trans_state_dep_scaling_mat = pickle.load(f)
    with open('object_track_experiment/hex_task_map_highres.pkl','rb') as f:
        hex_to_task_mapping = pickle.load(f)
    with open('object_track_experiment/const_object_highres.pkl','rb') as f:
        const = pickle.load(f)

    # with open('object_track_experiment/sat_cover_matrix_highres_neigh.pkl','rb') as f:
    #     sat_cover_matrix = pickle.load(f)
    # with open('object_track_experiment/graphs_highres_neigh.pkl','rb') as f:
    #     graphs = pickle.load(f)
    # with open('object_track_experiment/task_transition_scaling_highres_neigh.pkl','rb') as f:
    #     task_trans_state_dep_scaling_mat = pickle.load(f)
    # with open('object_track_experiment/hex_task_map_highres_neigh.pkl','rb') as f:
    #     hex_to_task_mapping = pickle.load(f)
    # with open('object_track_experiment/const_object_highres_neigh.pkl','rb') as f:
    #     const = pickle.load(f)

    lat_range = (20, 50)
    lon_range = (73, 135)
    L = 3
    lambda_ = 0.05
    T = 60
    num_objects = 50
    coverage_benefit = 1
    object_benefit = 10

    # sat_cover_matrix, graphs, task_trans_state_dep_scaling_mat, hex_to_task_mapping, const = get_sat_coverage_matrix_and_graphs_object_tracking_area(lat_range, lon_range, T)

    np.random.seed(42)
    task_objects = init_task_objects(num_objects, const, hex_to_task_mapping, T, velocity=10000*u.km/u.hr)
    benefits = get_benefits_from_task_objects(coverage_benefit, object_benefit, sat_cover_matrix, task_objects)

    print("Dynamic HAAL, Centralized")
    ass, tv = solve_object_track_w_dynamic_haal(sat_cover_matrix, task_objects, coverage_benefit, object_benefit, None, lambda_, L, parallel=False,
                                                state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    print("Value", tv)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct", pct)
    # plot_object_track_scenario(hex_to_task_mapping, sat_cover_matrix, task_objects, ass, task_trans_state_dep_scaling_mat,
    #                            "haal_no_neighbors.gif", show=False)

    print("Dynamic HAAL, Distributed")
    ass, tv = solve_object_track_w_dynamic_haal(sat_cover_matrix, task_objects, coverage_benefit, object_benefit, None, lambda_, L, distributed=True, graphs=graphs,
                                                state_dep_fn=timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat, verbose=True)
    print("Value", tv)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct", pct)

    # print("Normal HAAL")
    # ass, tv = solve_w_haal(benefits, None, lambda_, L, state_dep_fn=timestep_loss_state_dep_fn, 
    #                        task_trans_state_dep_scaling_mat=task_trans_state_dep_scaling_mat)
    # print("Value", tv)
    # print(is_assignment_mat_sequence_valid(ass))
    # pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    # print("Pct",pct)

    print("No handover")
    ass, tv = solve_wout_handover(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
    print("Value", tv)
    print(is_assignment_mat_sequence_valid(ass))
    # plot_object_track_scenario(hex_to_task_mapping, sat_cover_matrix, task_objects, ass, task_trans_state_dep_scaling_mat,
    #                            "nha_no_neighbors.gif", show=False)

    # # object_tracking_history(ass, task_objects, task_trans_state_dep_scaling_mat, sat_cover_matrix)
    pct = calc_pct_objects_tracked(ass, task_objects, task_trans_state_dep_scaling_mat)
    print("pct",pct)

    # print("Greedy")
    # ass, tv = solve_greedily(benefits, None, lambda_, timestep_loss_state_dep_fn, task_trans_state_dep_scaling_mat)
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
    
    lambda_ = 1.5

    # benefits, graphs = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_benefits, fov=fov, isl_dist=2500)

    # with open("haal_experiment1/paper_exp1_bens.pkl", 'wb') as f:
    #     pickle.dump(benefits,f)
    # with open("haal_experiment1/paper_exp1_graphs.pkl", 'wb') as f:
    #     pickle.dump(graphs,f)

    # with open("haal_experiment1/paper_exp1_bens.pkl", 'rb') as f:
    #     benefits = pickle.load(f)
    # with open("haal_experiment1/paper_exp1_graphs.pkl", 'rb') as f:
    #     graphs = pickle.load(f)

    # _, no_handover_val = solve_wout_handover(benefits, None, lambda_)

    # # _, cbba_val = solve_w_centralized_CBBA(benefits, None, lambda_, max_L, verbose=True)
    # cbba_val = 0

    # _, greedy_val = solve_greedily(benefits, None, lambda_)
    # print(greedy_val)
    # itersd_by_lookahead = []
    # valued_by_lookahead = []

    # iterscbba_by_lookahead = []
    # valuecbba_by_lookahead = []

    # valuec_by_lookahead = []
    # for L in range(1,max_L+1):
    #     print(f"lookahead {L}")
    #     # _, cbba_val, avg_iters = solve_w_CBBA_track_iters(benefits, None, lambda_, L, graphs=graphs, verbose=True)
    #     # iterscbba_by_lookahead.append(avg_iters)
    #     # valuecbba_by_lookahead.append(cbba_val)
        
    #     # _, d_val, avg_iters = solve_w_haal(benefits, None, lambda_, L, graphs=graphs, verbose=True, track_iters=True, distributed=True)
    #     # itersd_by_lookahead.append(avg_iters)
    #     # valued_by_lookahead.append(d_val)

    #     _, c_val = solve_w_haal(benefits, None, lambda_, L, distributed=False, verbose=True)
    #     valuec_by_lookahead.append(c_val)

    # #Values from 1/31, before scaling experiments
    valuecbba_by_lookahead = [4208.38020192484, 4412.873727755446, 4657.90330919782, 4717.85859678172, 4710.212483240204, 4726.329218229788]
    valuec_by_lookahead = [6002.840671517548, 7636.731195199751, 7581.29374466441, 7435.882168254755, 7511.4534257400755, 7591.261917337481]
    itersd_by_lookahead = [54.89247311827957, 61.17204301075269, 68.46236559139786, 72.64516129032258, 79.10752688172043, 80.02150537634408]
    iterscbba_by_lookahead = [18.50537634408602, 26.0, 29.193548387096776, 33.11827956989247, 34.806451612903224, 37.29032258064516]
    valued_by_lookahead = [6021.705454081699, 7622.246684035546, 7585.4110847804, 7294.093230272816, 7437.211996201664, 7559.402984912062]
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
    axes[1].set_ylabel("Average Iterations")
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

def calc_pass_statistics(benefits, assigns):
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    pass_lens = []
    pass_bens = []
    task_assign_len = []
    for j in range(m):
        for i in range(n):
            pass_started = False
            task_assigned = False
            assign_len = 0
            pass_len = 0
            pass_ben = 0
            for k in range(T):
                this_pass_assign_lens = []
                if benefits[i,j,k] > 0:
                    if not pass_started:
                        pass_started = True
                    pass_len += 1
                    pass_ben += benefits[i,j,k]

                    if assigns[k][i,j] == 1:
                        if not task_assigned: task_assigned = True
                        assign_len += 1
                    #If there are benefits and the task was previously assigned,
                    #but is no longer, end the streak
                    elif task_assigned:
                        task_assigned = False
                        this_pass_assign_lens.append(assign_len)
                        assign_len = 0

                elif pass_started and benefits[i,j,k] == 0:
                    if task_assigned:
                        this_pass_assign_lens.append(assign_len)
                    pass_started = False
                    task_assigned = False
                    for ass_len in this_pass_assign_lens:
                        task_assign_len.append(ass_len)
                    this_pass_assign_lens = []
                    pass_lens.append(pass_len)
                    pass_bens.append(pass_ben)
                    pass_len = 0
                    pass_ben = 0
                    assign_len = 0
    
    avg_pass_len = sum(pass_lens) / len(pass_lens)
    avg_pass_ben = sum(pass_bens) / len(pass_bens)
    avg_ass_len = sum(task_assign_len) / len(task_assign_len)
    
    return avg_pass_len, avg_pass_ben, avg_ass_len

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

    #         avg_pass_len, avg_pass_ben, avg_ass_len = calc_pass_statistics(benefits, assigns)
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
    axes[0].plot(lambdas_over_bens, [avg_pass_len]*len(lambdas_over_bens), 'k--', label="Avg. time\ntask in view")

    axes[0].set_ylabel("Avg. Length Satellite\nAssigned to Same Task")
    axes[1].set_ylabel("Total Benefit")
    axes[1].set_xlabel(r'$\frac{\lambda}{\beta_{pass, avg}}$')
    axes[1].xaxis.label.set_fontsize(16)
    axes[0].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))
    axes[1].set_xlim(min(lambdas_over_bens), max(lambdas_over_bens))

    handles, labels = [], []
    for handle, label in zip(*axes[0].get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)
    
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.45,0.5), ncol=2)
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


if __name__ == "__main__":
    scaling_experiment()
    # paper_experiment1()
    # paper_experiment2_tasking_history()