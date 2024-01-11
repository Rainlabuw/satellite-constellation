import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from tqdm import tqdm
import time
import pickle

from methods import *

from solve_optimally import solve_optimally
from solve_wout_handover import solve_wout_handover
from solve_w_haal import solve_w_haal, solve_w_haald_track_iters
from solve_w_accelerated_haal import solve_w_accel_haal, solve_w_accel_haald_track_iters
from solve_w_centralized_CBBA import solve_w_centralized_CBBA
from solve_w_CBBA import solve_w_CBBA, solve_w_CBBA_track_iters
from solve_greedily import solve_greedily
from classic_auction import Auction

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

    # benefits, graphs = get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m, T, altitude=altitude, benefit_func=calc_fov_benefits, fov=fov, isl_dist=2500)

    # with open("haal_experiment1/paper_exp1_bens.pkl", 'wb') as f:
    #     pickle.dump(benefits,f)
    # with open("haal_experiment1/paper_exp1_graphs.pkl", 'wb') as f:
    #     pickle.dump(graphs,f)

    # # with open("haal_experiment1/paper_exp1_bens.pkl", 'rb') as f:
    # #     benefits = pickle.load(f)
    # # with open("haal_experiment1/paper_exp1_graphs.pkl", 'rb') as f:
    # #     graphs = pickle.load(f)

    # _, no_handover_val, _ = solve_wout_handover(benefits, None, lambda_)

    # # _, cbba_val, _ = solve_w_centralized_CBBA(benefits, None, lambda_)

    # _, greedy_val, _ = solve_greedily(benefits, None, lambda_)
    # print(greedy_val)
    # itersd_by_lookahead = []
    # valued_by_lookahead = []

    # iterscbba_by_lookahead = []
    # valuecbba_by_lookahead = []

    # valuec_by_lookahead = []
    # for L in range(1,max_L+1):
    #     print(f"lookahead {L}")
    #     _, cbba_val, _, avg_iters = solve_w_CBBA_track_iters(benefits, None, lambda_, L, graphs=graphs, verbose=True)
    #     iterscbba_by_lookahead.append(avg_iters)
    #     valuecbba_by_lookahead.append(cbba_val)
        
    #     _, d_val, _, avg_iters = solve_w_haald_track_iters(benefits, None, lambda_, L, graphs=graphs, verbose=True)
    #     itersd_by_lookahead.append(avg_iters)
    #     valued_by_lookahead.append(d_val)

    #     _, c_val, _ = solve_w_haal(benefits, None, lambda_, L, distributed=False, verbose=True)
    #     valuec_by_lookahead.append(c_val)

    valuecbba_by_lookahead = [4208.38020192484, 4412.873727755446, 4657.90330919782, 4717.85859678172, 4710.212483240204, 4726.329218229788]
    valuec_by_lookahead = [6002.840671517548, 7636.731195199751, 7581.29374466441, 7435.882168254755, 7511.4534257400755, 7591.261917337481]
    itersd_by_lookahead = [54.89247311827957, 61.17204301075269, 68.46236559139786, 72.64516129032258, 79.10752688172043, 80.02150537634408]
    iterscbba_by_lookahead = [18.50537634408602, 26.0, 29.193548387096776, 33.11827956989247, 34.806451612903224, 37.29032258064516]
    valued_by_lookahead = [6021.705454081699, 7622.246684035546, 7585.4110847804, 7294.093230272816, 7437.211996201664, 7559.402984912062]
    greedy_val = 3650.418056196203
    no_handover_val = 4078.018608711949


    fig, axes = plt.subplots(2,1)
    axes[0].plot(range(1,max_L+1), valued_by_lookahead, 'g--', label="HAAL-D")
    axes[0].plot(range(1,max_L+1), valuec_by_lookahead, 'g', label="HAAL")
    axes[0].plot(range(1,max_L+1), valuecbba_by_lookahead, 'b', label="CBBA")
    axes[0].plot(range(1,max_L+1), [no_handover_val]*max_L, 'r', label="NHA")
    axes[0].plot(range(1,max_L+1), [greedy_val]*max_L, 'k', label="GA")
    axes[0].set_ylabel("Total value")
    axes[0].set_xticks(range(1,max_L+1))
    axes[0].set_ylim((0, 1.1*max(valuec_by_lookahead)))
    axes[1].set_xlabel("Lookahead window L")
    axes[0].legend(loc='lower right')

    axes[1].plot(range(1,max_L+1), itersd_by_lookahead, 'g--', label="HAAL-D")
    axes[1].plot(range(1,max_L+1), iterscbba_by_lookahead, 'b', label="CBBA")
    axes[1].set_ylim((0, 1.1*max(itersd_by_lookahead)))
    axes[0].set_xticks(range(1,max_L+1))
    axes[1].set_ylabel("Average iterations")
    axes[1].set_xlabel("Lookahead window")

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

    fig.set_figwidth(8)
    fig.set_figheight(5)
    plt.savefig("haal_experiment1/paper_exp1.pdf")
    plt.show()

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

    #phantom lines for legend
    # greedy_ax.plot([T+1], [0], 'r', label="No Handover")
    # greedy_ax.plot([T+1], [0], 'g', label="HAAL-D")
    haal_ax.plot([T+1], [0], color=vline_color, linestyle='--', label="Task Changes")
    haal_ax.legend(loc="upper left",bbox_to_anchor=(0,-0.15))
    # haal_ax.set_title("HAAL-D")
    # greedy_ax.set_title("Greedy")
    # no_handover_ax.set_title("No Handover")
    haal_ax.set_xlabel("Timestep")
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
    ass, val1, _ = solve_w_mhal(benefit, init_assignment, 1, 1)
    print("L = 1",val1)
    for a in ass:
        print(a)

    L = benefit.shape[-1]
    ass, val2, _ = solve_w_mhal(benefit, init_assignment, 1, 2)
    print("L = 2",val2)
    for a in ass:
        print(a)

    rat = 1/2+1/2*((L-1)/L)

    print(opt_val, val1, val2)


if __name__ == "__main__":
    l_compare_counterexample()