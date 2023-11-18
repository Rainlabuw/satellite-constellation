import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import time

from methods import *

from solve_optimally import solve_optimally
from solve_naively import solve_naively
from solve_w_mhal import solve_w_mhal, solve_w_mhald_track_iters
from solve_w_centralized_CBBA import solve_w_centralized_CBBA
from classic_auction import Auction

from constellation_sim.ConstellationSim import get_benefits_and_graphs_from_constellation, ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

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
    avg_mhal = 0
    avg_mha = 0
    avg_naive = 0

    num_avgs = 50
    for _ in tqdm(range(num_avgs)):
        benefit = np.random.rand(n,m,T)

        #SMHGL centralized, lookahead = 3
        _, mhal_ben, _ = solve_w_mhal(benefit, L, init_ass, lambda_=lambda_)
        avg_mhal += mhal_ben/num_avgs

        #MHA
        _, mha_ben, _ = solve_w_mhal(benefit, 1, init_ass, lambda_=lambda_)
        avg_mha += mha_ben/num_avgs

        #Naive
        _, ben, _ = solve_naively(benefit, init_ass, lambda_)
        avg_naive += ben/num_avgs

        #Optimal
        _, ben, _ = solve_optimally(benefit, init_ass, lambda_)
        avg_best += ben/num_avgs

    fig = plt.figure()
    plt.bar(["Naive","MHA", f"MHAL (L={L})", "Optimal"], [avg_naive, avg_mha, avg_mhal, avg_best])
    plt.title(f"Average benefit across {num_avgs} runs, n={n}, m={m}, T={T}")
    print(["Naive","MHA", f"MHAL (L={L})", "Optimal"])
    print([avg_naive, avg_mha, avg_mhal, avg_best])
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
        multi_auction = MHAL_Auction(benefits, None, lookahead)
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
        multi_auction = MHAL_Auction(benefits, None, lookahead)
        multi_auction.run_auctions()
        ben,_ = multi_auction.calc_value_and_num_handovers()
        print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

def compare_MHA_to_other_algs():
    #Case where we expect solutions to get increasingly better as lookahead window increases
    n = 50
    m = 50
    T = 95
    lambda_ = 1

    print("Comparing performance of MHAL to other algorithms")
    num_avgs = 10

    mhal2_ben = 0
    mhal2_nh = 0

    mhal5_ben = 0
    mhal5_nh = 0

    mha_ben = 0
    mha_nh = 0

    naive_ben = 0
    naive_nh = 0

    sga_ben = 0
    sga_nh = 0
    for _ in range(num_avgs):
        print(f"Run {_}/{num_avgs}")
        # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
        benefits, _ = get_benefits_and_graphs_from_constellation(n, m, T)
        print("Generated realistic benefits")

        #SMHGL centralized, lookahead = 2
        multi_auction = MHAL_Auction(benefits, None, 2, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        mhal2_ben += ben/num_avgs
        mhal2_nh += nh/num_avgs

        #SMHGL centralized, lookahead = 5
        multi_auction = MHAL_Auction(benefits, None, 5, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        mhal5_ben += ben/num_avgs
        mhal5_nh += nh/num_avgs

        #MHA
        multi_auction = MHAL_Auction(benefits, None, 1, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_value_and_num_handovers()
        mha_ben += ben/num_avgs
        mha_nh += nh/num_avgs

        #Naive
        _, ben, nh = solve_naively(benefits, lambda_)
        naive_ben += ben/num_avgs
        naive_nh += nh/num_avgs

        #SGA/CBBA
        handover_ben, _, handover_nh = solve_w_centralized_CBBA(benefits, lambda_)

        sga_ben += handover_ben/num_avgs
        sga_nh += handover_nh/num_avgs

    fig, axes = plt.subplots(2,1)
    axes[0].bar(["Naive", "SGA", "MHA", "MHAL2", "MHAL5"],[naive_ben, sga_ben, mha_ben, mhal2_ben, mhal5_ben])
    axes[0].set_title("Average benefit across 10 runs")
    # axes[0].set_xlabel("Lookahead timesteps")

    axes[1].bar(["Naive", "SGA", "MHA", "MHAL2", "MHAL5"],[naive_nh, sga_nh, mha_nh, mhal2_nh, mhal5_nh])
    
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
    benefits, graphs = get_benefits_and_graphs_from_constellation(36, 15, m, T, altitude=500)
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

            #MHAL with true lookaheads
            _, ben, nh = solve_w_mhal(benefits, lookahead, init_assignment, distributed=False, lambda_=lambda_, verbose=True)
            avg_ben += ben/num_avgs
            avg_nh += nh/num_avgs
            
            # #MHAL (distributed)
            # _, ben, nh = solve_w_mhal(benefits, lookahead, init_assignment, distributed=True, lambda_=lambda_)
            # avg_approx_ben += ben/num_avgs

        resulting_bens.append(avg_ben)
        # resulting_approx_bens.append(avg_approx_ben)
        handovers.append(avg_nh)

    plt.plot(range(1,max_lookahead+1), resulting_bens, label="MHAL (Centralized)")
    # plt.plot(range(1,max_lookahead+1), resulting_approx_bens, label="MHAL (Distributed)")
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

    naive_benefits = []
    naive_handover_benefits = []
    naive_handover_violations = []

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

        naive_benefits.append(sum(benefits))
        naive_handover_benefits.append(handover_ben)
        naive_handover_violations.append(calc_assign_seq_handover_penalty(assignment_mats, lambda_))

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
    print(naive_benefits)
    print(naive_handover_benefits)
    print(sequential_benefits)
    print(sequential_handover_benefits)
    print(sga_benefits)
    print(sga_handover_benefits)


    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Top subplot
    axs[0].set_title('Benefits without handover penalty')
    axs[0].set_xlabel('Number of agents')
    axs[0].set_ylabel('Total benefit')
    axs[0].bar(np.arange(len(naive_benefits)), naive_benefits, width=0.2, label='Naive')
    axs[0].bar(np.arange(len(sga_benefits))+0.2, sga_benefits, width=0.2, label='SGA')
    axs[0].bar(np.arange(len(sequential_benefits))+0.4, sequential_benefits, width=0.2, label='MHA (Ours)')
    axs[0].set_xticks(np.arange(len(naive_benefits)))
    axs[0].set_xticklabels([str(n) for n in ns])
    axs[0].legend(loc='lower center')

    # Bottom subplot
    axs[1].set_title('Total Benefits, including handover penalty')
    axs[1].set_xlabel('Number of agents')
    axs[1].set_ylabel('Average Benefit')
    axs[1].bar(np.arange(len(naive_handover_benefits)), naive_handover_benefits, width=0.2, label='Naive')
    axs[1].bar(np.arange(len(sga_handover_benefits))+0.2, sga_handover_benefits, width=0.2, label='CBBA')
    axs[1].bar(np.arange(len(sequential_handover_benefits))+0.4, sequential_handover_benefits, width=0.2, label='MHA (Ours)')
    axs[1].set_xticks(np.arange(len(naive_handover_benefits)))
    axs[1].set_xticklabels([str(n) for n in ns])
    #add a legend to the bottom middle of the subplot
    axs[1].legend(loc='lower center')

    plt.show()

def distributed_comparison():
    """
    Compare the performance of the distributed MHAL algorithm
    to other algorithms (including centralized MHAL)
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

    naive_tot = 0
    cbba_tot = 0

    n_tests = 10
    for _ in tqdm(range(n_tests)):
        benefits = np.random.rand(n, m, T)
        init_assignment = np.eye(n,m)

        # st = time.time()
        _, d_val, _ = solve_w_mhal(benefits, L, init_assignment, distributed=True, lambda_=lambda_)

        _, c_val, _ = solve_w_mhal(benefits, L, init_assignment, distributed=False, lambda_=lambda_)

        _, ca_val, _ = solve_w_mhal(benefits, L, init_assignment, distributed=False, central_approx=True, lambda_=lambda_)

        _, naive_val, _ = solve_naively(benefits, init_assignment, lambda_)
        _, cbba_val, _ = solve_w_centralized_CBBA(benefits, init_assignment, lambda_)

        ctot += c_val/n_tests
        catot += ca_val/n_tests
        dtot += d_val/n_tests

        naive_tot += naive_val/n_tests
        cbba_tot += cbba_val/n_tests

    print([naive_tot, cbba_tot, ctot, catot, dtot])
    plt.bar(range(6), [naive_tot, cbba_tot, ctot, catot, dtot], tick_label=["Naive", "CBBA", "Centralized", "Centralized Approx", "Distributed"])
    plt.show()

def realistic_orbital_simulation():
    """
    Simulate a realistic orbital mechanics case
    using distributed and centralized MHAL.

    Compute this over several lookahead windows.
    """
    n = 100
    m = 100
    T = 93
    max_L = 3
    lambda_ = 1.5
    init_assignment = None

    cbba_tot = 0
    naive_tot = 0

    tot_iters_by_lookahead = np.zeros(max_L)
    tot_value_by_lookahead = np.zeros(max_L)

    num_avgs = 5
    for _ in range(num_avgs):
        print(f"\nNum trial {_}")
        num_planes = 10
        num_sats_per_plane = 10
        if n != num_planes*num_sats_per_plane: raise Exception("Make sure n = num planes * num sats per plane")
        benefits, graphs = get_benefits_and_graphs_from_constellation(num_planes,num_sats_per_plane,m,T)

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
        print(f"Done solving CBBA, solving naive...")
        _, naive_val, _ = solve_naively(benefits, init_assignment, lambda_)
        naive_tot += naive_val/num_avgs

        iters_by_lookahead = []
        value_by_lookahead = []
        for L in range(1,max_L+1):
            print(f"lookahead {L}")
            _, d_val, _, avg_iters = solve_w_mhald_track_iters(benefits, L, init_assignment, graphs=graphs, lambda_=lambda_, verbose=True)

            iters_by_lookahead.append(avg_iters)
            value_by_lookahead.append(d_val)

        tot_iters_by_lookahead += np.array(iters_by_lookahead)/num_avgs
        tot_value_by_lookahead += np.array(value_by_lookahead)/num_avgs

    fig, axes = plt.subplots(2,1)
    
    axes[0].plot(range(1,max_L+1), tot_value_by_lookahead, 'g', label="MHAL-D")
    # axes[0].plot(range(1,max_L+1), [cbba_tot]*max_L, 'b--', label="CBBA")
    axes[0].plot(range(1,max_L+1), [naive_tot]*max_L, 'r--', label="Naive")
    axes[0].set_ylabel("Total value")
    axes[0].set_xticks(range(1,max_L+1))
    axes[0].set_ylim((0, 1.1*max(tot_value_by_lookahead)))
    axes[0].legend()

    axes[1].plot(range(1,max_L+1), tot_iters_by_lookahead, 'g', label="MHAL-D")
    axes[1].set_ylim((0, 1.1*max(tot_iters_by_lookahead)))
    axes[1].set_ylabel("Average iterations")
    axes[1].set_xlabel("Lookahead window")
    axes[1].set_xticks(range(1,max_L+1))

    plt.savefig('real_const.png')
    plt.show()

def epsilon_effect():
    """
    Simulate a realistic orbital mechanics case
    using distributed and centralized MHAL.

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
    naive_tot = 0

    num_avgs = 1
    for _ in tqdm(range(num_avgs)):
        benefits, graphs = get_benefits_and_graphs_from_constellation(10,5,n,T)

        #Distributed
        print(f"Done generating benefits, solving distributed 0.1...")
        _, d_val_0p1, _ = solve_w_mhal(benefits, L, init_assignment, distributed=True, graphs=None, lambda_=lambda_, eps=0.1, verbose=True)
        d_tot_0p1 += d_val_0p1/num_avgs

        print(f"Done generating benefits, solving distributed 0.01...")
        _, d_val_0p01, _ = solve_w_mhal(benefits, L, init_assignment, distributed=True, graphs=None, lambda_=lambda_, eps=0.01, verbose=True)
        d_tot_0p01 += d_val_0p01/num_avgs

        print(f"Done generating benefits, solving distributed 0.001...")
        _, d_val_0p001, _ = solve_w_mhal(benefits, L, init_assignment, distributed=True, graphs=None, lambda_=lambda_, eps=0.001, verbose=True)
        d_tot_0p001 += d_val_0p001/num_avgs

        #Centralized
        print(f"Done solving distributed, solving centralized...")
        _, c_val, _ = solve_w_mhal(benefits, L, init_assignment, distributed=False, lambda_=lambda_)
        c_tot += c_val/num_avgs

        #CBBA
        print(f"Done solving centralized, solving CBBA...")
        _, cbba_val, _ = solve_w_centralized_CBBA(benefits, init_assignment, lambda_, verbose=True)
        cbba_tot += cbba_val/num_avgs

        #Naive
        print(f"Done solving CBBA, solving naive...")
        _, naive_val, _ = solve_naively(benefits, init_assignment, lambda_)
        naive_tot += naive_val/num_avgs

    print([naive_tot, cbba_tot, c_tot, d_tot_0p1, d_tot_0p01, d_tot_0p001])
    plt.bar(range(6), [naive_tot, cbba_tot, c_tot, d_tot_0p1, d_tot_0p01, d_tot_0p001], tick_label=["Naive", "CBBA", "Centralized", "Distributed 0.1", "Distributed 0.01", "Distributed 0.001"])
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
            _, val, _ = solve_w_mhal(benefit, lookahead, None, distributed=False, lambda_=0.5)
            if val < opt_val:
                print("Bound violation!!")
                print(benefit)
                print(f"Lookahead {lookahead} value: {val}, lower bound: {lower_bd}, opt_val: {opt_val}")
                return
            
def lookahead_counterexample():
    benefit = np.zeros((5,5,3))
    benefit[:,:,0] = np.array([[100, 1, 1, 1, 1],
                               [1, 100, 1, 1, 1],
                               [1, 1, 1.01, 1, 1],
                               [1, 1, 1, 1.01, 1],
                               [1, 1, 1, 1, 1.01]])
    
    benefit[:,:,1] = np.array([[100, 1, 1, 1, 1],
                               [1, 100, 1, 1, 1],
                               [1, 1, 1.01, 1, 1],
                               [1, 1, 1, 1.01, 1],
                               [1, 1, 1, 1, 1.01]])
    
    benefit[:,:,2] = np.array([[1, 100, 1, 1, 1],
                               [100, 1, 1, 1, 1],
                               [1, 1, 1, 2.01, 1],
                               [1, 1, 1, 1, 2.01],
                               [1, 1, 2.01, 1, 1]])
    
    benefit[:,:,3] = np.array([[100, 1, 1, 1, 1],
                               [1, 100, 1, 1, 1],
                               [1, 1, 1, 1, 2.01],
                               [1, 1, 2.01, 1, 1],
                               [1, 1, 1, 2.01, 1]])

    init_assignment = np.array([[0, 0, 0, 0, 1],
                               [0, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0]])

    ass, opt_val, _ = solve_optimally(benefit, init_assignment, 1)
    print(opt_val)
    for a in ass:
        print(a)

    L = benefit.shape[-1]
    print("mhal")
    ass, val, _ = solve_w_mhal(benefit, L, init_assignment)
    print(val)
    for a in ass:
        print(a)

    rat = 1/2+1/2*((L-1)/3)

    print(f"Ratio: {val/opt_val}, desired rat: {rat}")

def performance_v_num_agents_line_chart():
    ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    T = 10
    L = 5
    lambda_ = 0.5
    num_avgs = 5
    init_assign = None

    naive_vals = []
    sga_vals = []
    mha_vals = []
    mhal_vals = []
    mhald_vals = []

    naive_nhs = []
    sga_nhs = []
    mha_nhs = []
    mhal_nhs = []
    mhald_nhs = []

    for n in ns:
        print(f"Solving for {n} agents...")
        m = n

        naive_total_val = 0
        sga_total_val = 0
        mha_total_val = 0
        mhal_total_vals = 0
        mhald_total_vals = 0

        naive_total_nhs = 0
        sga_total_nhs = 0
        mha_total_nhs = 0
        mhal_total_nhs = 0
        mhald_total_nhs = 0
        
        for _ in range(num_avgs):
            print(_)
            benefits = np.random.random((n, m, T))

            _, naive_val, naive_nh = solve_naively(benefits, init_assign, lambda_)
            naive_total_val += naive_val/num_avgs
            naive_total_nhs += naive_nh/num_avgs

            _, sga_val, sga_nh = solve_w_centralized_CBBA(benefits, init_assign, lambda_)
            sga_total_val += sga_val/num_avgs
            sga_total_nhs += sga_nh/num_avgs

            _, mha_val, mha_nh = solve_w_mhal(benefits, 1, init_assign, lambda_=lambda_)
            mha_total_val += mha_val/num_avgs
            mha_total_nhs += mha_nh/num_avgs

            _, mhal_val, mhal_nh = solve_w_mhal(benefits, L, init_assign, lambda_=lambda_)
            mhal_total_vals += mhal_val/num_avgs
            mhal_total_nhs += mhal_nh/num_avgs

            _, mhald_val, mhald_nh = solve_w_mhal(benefits, L, init_assign, distributed=True, lambda_=lambda_)
            mhald_total_vals += mhald_val/num_avgs
            mhald_total_nhs += mhald_nh/num_avgs

        naive_vals.append(naive_total_val/n)
        sga_vals.append(sga_total_val/n)
        mha_vals.append(mha_total_val/n)
        mhal_vals.append(mhal_total_vals/n)
        mhald_vals.append(mhald_total_vals/n)

        naive_nhs.append(naive_total_nhs/n)
        sga_nhs.append(sga_total_nhs/n)
        mha_nhs.append(mha_total_nhs/n)
        mhal_nhs.append(mhal_total_nhs/n)
        mhald_nhs.append(mhald_total_nhs/n)

    fig, axes = plt.subplots(2,1, sharex=True)
    fig.suptitle(f"Performance vs. number of agents over {num_avgs} runs, m=n, T={T}, L={L}, lambda={lambda_}")
    axes[0].plot(ns, naive_vals, label="Naive")
    axes[0].plot(ns, sga_vals, label="SGA")
    axes[0].plot(ns, mha_vals, label="MHA")
    axes[0].plot(ns, mhal_vals, label=f"MHAL (L={L})")
    axes[0].plot(ns, mhald_vals, label=f"MHAL-D (L={L})")
    axes[0].set_ylabel("Average benefit per agent")
    
    axes[1].plot(ns, naive_nhs, label="Naive")
    axes[1].plot(ns, sga_nhs, label="SGA")
    axes[1].plot(ns, mha_nhs, label="MHA")
    axes[1].plot(ns, mhal_nhs, label=f"MHAL (L={L})")
    axes[1].plot(ns, mhald_nhs, label=f"MHAL-D (L={L})")
    axes[1].set_ylabel("Average number of handovers per agent")
    
    axes[1].set_xlabel("Number of agents")

    axes[1].legend()

    print(naive_vals)
    print(sga_vals)
    print(mha_vals)
    print(mhal_vals)
    print(mhald_vals)

    print(naive_nhs)
    print(sga_nhs)
    print(mha_nhs)
    print(mhal_nhs)
    print(mhald_nhs)
    plt.savefig("performance_v_num_agents.png")
    plt.show()

def tasking_history_plot():
    """
    Tracks the history of task allocations in a system over time,
    with and without MHAL
    """
    n = 100
    m = 300
    num_planes = 10
    num_sats_per_plane = 10
    altitude=550
    T = 50
    lambda_ = 0.5

    init_assign = np.eye(n, m)

    benefits, graphs = get_benefits_and_graphs_from_constellation(num_planes,num_sats_per_plane,m,T, altitude=altitude, benefit_func=calc_fov_benefits)

    # benefits = np.random.random((n, m, T))

    naive_ass, naive_val, _ = solve_naively(benefits, init_assign, lambda_)

    mhal_ass, mhal_val, _ = solve_w_mhal(benefits, 1, init_assign, None, lambda_)

    mhal5_ass, mhal5_val, _ = solve_w_mhal(benefits, 5, init_assign, None, lambda_)

    print(naive_val, mhal_val, mhal5_val)

    #~~~~~~~~~~~~~~~~~~~~~~ PLOT OF TASKING HISTORY ~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(3,1, sharex=True)
    agent1_naive_ass = [np.argmax(naive_a[0,:]) for naive_a in naive_ass]

    agent1_mhal_ass = [np.argmax(mhal_a[0,:]) for mhal_a in mhal_ass]

    agent1_mhal5_ass = [np.argmax(mhal5_a[0,:]) for mhal5_a in mhal5_ass]

    axes[0].plot(range(T), agent1_naive_ass, label="Naive")
    axes[1].plot(range(T), agent1_mhal_ass, label="MHAL 1")
    axes[2].plot(range(T), agent1_mhal5_ass, label="MHAL 5")
    axes[2].set_xlabel("Time (min.)")
    axes[0].set_ylabel("Task assignment")
    axes[1].set_ylabel("Task assignment")
    axes[2].set_ylabel("Task assignment")

    axes[0].set_title("Satellite 0 tasking, Naive")
    axes[1].set_title("Satellite 0 tasking, MHA")
    axes[2].set_title("Satellite 0 tasking, MHAL (L=5)")
    plt.show(block=False)

    #~~~~~~~~~~~~~~~~~~~~ PLOT OF PRODUCTIVE TASKS COMPLETED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    naive_valid_tasks = []
    mhal_valid_tasks = []
    mhal5_valid_tasks = []
    for k in range(T):
        naive_assigned_benefits = naive_ass[k]*benefits[:,:,k]
        num_naive_valid_tasks = np.sum(np.where(naive_assigned_benefits, 1, 0))

        naive_valid_tasks.append(num_naive_valid_tasks)

        mhal_assigned_benefits = mhal_ass[k]*benefits[:,:,k]
        num_mhal_valid_tasks = np.sum(np.where(mhal_assigned_benefits, 1, 0))

        mhal_valid_tasks.append(num_mhal_valid_tasks)

        mhal5_assigned_benefits = mhal5_ass[k]*benefits[:,:,k]
        num_mhal5_valid_tasks = np.sum(np.where(mhal5_assigned_benefits, 1, 0))

        mhal5_valid_tasks.append(num_mhal5_valid_tasks)

    fig = plt.figure()
    plt.plot(range(T), naive_valid_tasks, label="Naive")
    plt.plot(range(T), mhal_valid_tasks, label="MHA")
    plt.plot(range(T), mhal5_valid_tasks, label="MHAL")
    plt.legend()
    plt.show(block=False)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT OF BENEFITS CAPTURED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure()
    gs = fig.add_gridspec(3,2)
    naive_ax = fig.add_subplot(gs[0,0])
    mhal_ax = fig.add_subplot(gs[1,0])
    mhal5_ax = fig.add_subplot(gs[2,0])
    val_ax = fig.add_subplot(gs[:,1])

    prev_naive = 0
    prev_mhal = 0
    prev_mhal5 = 0

    naive_ben_line = []
    mhal_ben_line = []
    mhal5_ben_line = []

    naive_val_line = []
    mhal_val_line = []
    mhal5_val_line = []
    
    for k in range(T):
        naive_choice = np.argmax(naive_ass[k][0,:])
        mhal_choice = np.argmax(mhal_ass[k][0,:])
        mhal5_choice = np.argmax(mhal5_ass[k][0,:])

        if prev_naive != naive_choice:
            naive_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                # naive_ben_line.append(naive_ben_line[-1])
                if len(naive_ben_line) > 1:
                    naive_ax.plot(range(k-len(naive_ben_line), k), naive_ben_line, 'r')
                elif len(naive_ben_line) == 1:
                    naive_ax.plot(range(k-len(naive_ben_line), k), naive_ben_line, 'r.', markersize=1)
            naive_ben_line = [benefits[0,naive_choice, k]]
        else:
            naive_ben_line.append(benefits[0, naive_choice, k])

        if prev_mhal != mhal_choice:
            mhal_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(mhal_ben_line) > 1:
                    mhal_ax.plot(range(k-len(mhal_ben_line), k), mhal_ben_line,'b')
                elif len(mhal_ben_line) == 1:
                    mhal_ax.plot(range(k-len(mhal_ben_line), k), mhal_ben_line,'b.', markersize=1)
            mhal_ben_line = [benefits[0,mhal_choice, k]]
        else:
            mhal_ben_line.append(benefits[0,mhal_choice, k])

        if prev_mhal5 != mhal5_choice:
            mhal5_ax.axvline(k-0.5, linestyle='--')
            if k != 0: 
                if len(mhal5_ben_line) > 1:
                    mhal5_ax.plot(range(k-len(mhal5_ben_line), k), mhal5_ben_line,'g')
                elif len(mhal5_ben_line) == 1:
                    mhal5_ax.plot(range(k-len(mhal5_ben_line), k), mhal5_ben_line,'g.', markersize=1)

            mhal5_ben_line = [benefits[0,mhal5_choice, k]]
        else:
            mhal5_ben_line.append(benefits[0,mhal5_choice, k])

        naive_val_so_far, _ = calc_value_and_num_handovers(naive_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        naive_val_line.append(naive_val_so_far)

        mhal_val_so_far, _ = calc_value_and_num_handovers(mhal_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        mhal_val_line.append(mhal_val_so_far)

        mhal5_val_so_far, _ = calc_value_and_num_handovers(mhal5_ass[:k+1], benefits[:,:,:k+1], init_assign, lambda_)
        mhal5_val_line.append(mhal5_val_so_far)

        prev_naive = naive_choice
        prev_mhal = mhal_choice
        prev_mhal5 = mhal5_choice

    #plot last interval
    naive_ax.plot(range(k+1-len(naive_ben_line), k+1), naive_ben_line, 'r')
    mhal_ax.plot(range(k+1-len(mhal_ben_line), k+1), mhal_ben_line,'b')
    mhal5_ax.plot(range(k+1-len(mhal5_ben_line), k+1), mhal5_ben_line,'g')

    #plot value over time
    val_ax.plot(range(T), naive_val_line, 'r', label='Naive')
    val_ax.plot(range(T), mhal_val_line, 'b', label='MHAL')
    val_ax.plot(range(T), mhal5_val_line, 'g', label='MHAL 5')

    plt.show()

if __name__ == "__main__":
    lookahead_counterexample()