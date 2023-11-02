import matplotlib.pyplot as plt
import numpy as np

from methods import *

from solve_optimally import solve_optimally
from solve_naively import solve_naively
from solve_w_sgmh import SMGHAuction
from solve_w_centralized_CBBA import solve_w_centralized_CBBA
from classic_auction import Auction

from constellation_sim.ConstellationSim import get_benefit_matrix_from_constellation, ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

def optimal_baseline_comparison():
    """
    Compare various solutions types against the true optimal
    """
    n = 5
    m = 5
    T = 3

    init_ass = None
    
    lambda_ = 1

    avg_best = 0
    avg_smghl = 0
    avg_smgh = 0
    avg_naive = 0

    num_avgs = 50
    for _ in range(num_avgs):
        benefit = np.random.rand(n,m,T)

        print(f"Run {_}/{num_avgs}", end='\r')

        #SMHGL centralized, lookahead = 2
        multi_auction = SMGHAuction(benefit, None, T, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        avg_smghl += ben/num_avgs

        #SMGH
        multi_auction = SMGHAuction(benefit, None, 1, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        avg_smgh += ben/num_avgs

        #Naive
        ben, nh = solve_naively(benefit, lambda_)
        avg_naive += ben/num_avgs

        #Optimal
        ben, _ = solve_optimally(benefit, init_ass, lambda_)
        avg_best += ben/num_avgs

    fig = plt.figure()
    plt.bar(["Naive","SMGH", "SMGHL", "Optimal"], [avg_naive, avg_smgh, avg_smghl, avg_best])
    plt.title(f"Average benefit across {num_avgs} runs, n={n}, m={m}, T={T}")

    plt.show()

def SMGH_unit_testing():
    """
    Tests SMGH in various cases where solutions are known.
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
        multi_auction = SMGHAuction(benefits, None, lookahead)
        multi_auction.run_auctions()
        ben = multi_auction.calc_benefit()
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
        multi_auction = SMGHAuction(benefits, None, lookahead)
        multi_auction.run_auctions()
        ben,_ = multi_auction.calc_benefit()
        print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

def compare_SMGH_to_other_algs():
    #Case where we expect solutions to get increasingly better as lookahead window increases
    n = 50
    m = 50
    T = 95
    lambda_ = 1

    print("Comparing performance of SMGHL to other algorithms")
    num_avgs = 10

    smghl2_ben = 0
    smghl2_nh = 0

    smghl5_ben = 0
    smghl5_nh = 0

    smgh_ben = 0
    smgh_nh = 0

    naive_ben = 0
    naive_nh = 0

    sga_ben = 0
    sga_nh = 0
    for _ in range(num_avgs):
        print(f"Run {_}/{num_avgs}")
        # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
        benefits = get_benefit_matrix_from_constellation(n, m, T)
        print("Generated realistic benefits")

        #SMHGL centralized, lookahead = 2
        multi_auction = SMGHAuction(benefits, None, 2, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        smghl2_ben += ben/num_avgs
        smghl2_nh += nh/num_avgs

        #SMHGL centralized, lookahead = 5
        multi_auction = SMGHAuction(benefits, None, 5, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        smghl5_ben += ben/num_avgs
        smghl5_nh += nh/num_avgs

        #SMGH
        multi_auction = SMGHAuction(benefits, None, 1, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        smgh_ben += ben/num_avgs
        smgh_nh += nh/num_avgs

        #Naive
        ben, nh = solve_naively(benefits, lambda_)
        naive_ben += ben/num_avgs
        naive_nh += nh/num_avgs

        #SGA/CBBA
        sg_assignment_mats = solve_w_centralized_CBBA(benefits, lambda_)
        sg_benefit = 0
        for k, sg_assignment_mat in enumerate(sg_assignment_mats):
            sg_benefit += (benefits[:,:,k]*sg_assignment_mat).sum()

        handover_ben = sg_benefit - calc_handover_penalty(None, sg_assignment_mats, lambda_)
        sga_ben += handover_ben/num_avgs
        sga_nh += calc_handover_penalty(None, sg_assignment_mats, lambda_)/lambda_/num_avgs

    fig, axes = plt.subplots(2,1)
    axes[0].bar(["Naive", "SGA", "SMGH", "SMGHL2", "SMGHL5"],[naive_ben, sga_ben, smgh_ben, smghl2_ben, smghl5_ben])
    axes[0].set_title("Average benefit across 10 runs")
    # axes[0].set_xlabel("Lookahead timesteps")

    axes[1].bar(["Naive", "SGA", "SMGH", "SMGHL2", "SMGHL5"],[naive_nh, sga_nh, smgh_nh, smghl2_nh, smghl5_nh])
    
    axes[1].set_title("Average number of handovers across 10 runs")
    fig.suptitle(f"Test with n={n}, m={m}, T={T}, lambda={lambda_}, realistic-ish benefits")

    plt.show()

def test_SMGH_lookahead_performance():
    """
    Test performance of SMGH as lookahead window increases.

    Hopefully, general trend is that performance increases as lookahead increases.
    """
    print("Expect performance to generally increase as lookahead increases")
    n = 50
    m = 50
    T = 25
    lambda_ = 1

    max_lookahead = 5
    num_avgs = 100

    resulting_bens = []
    resulting_approx_bens = []
    handovers = []
    for lookahead in range(1,max_lookahead+1):
        avg_ben = 0
        avg_nh = 0
        avg_approx_ben = 0
        for _ in range(num_avgs):
            print(f"Lookahead {lookahead} ({_}/{num_avgs})", end='\r')
            # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
            benefits = get_benefit_matrix_from_constellation(n, m, T)
            # benefits = np.random.rand(n,m,T)

            #SMGHL with true lookaheads
            multi_auction = SMGHAuction(benefits, None, lookahead, lambda_=lambda_)
            multi_auction.run_auctions()
            ben, nh = multi_auction.calc_benefit()
            avg_ben += ben/num_avgs
            avg_nh += nh/num_avgs
            
            #SMGHL (distributed)
            multi_auction = SMGHAuction(benefits, None, lookahead, lambda_=lambda_, approximate=True)
            multi_auction.run_auctions()
            ben, _ = multi_auction.calc_benefit()
            avg_approx_ben += ben/num_avgs

        resulting_bens.append(avg_ben)
        resulting_approx_bens.append(avg_approx_ben)
        handovers.append(avg_nh)

    plt.plot(range(1,max_lookahead+1), resulting_bens, label="SMGHL (Centralized)")
    plt.plot(range(1,max_lookahead+1), resulting_approx_bens, label="SMGHL (Distributed)")
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
        
        handover_ben = sum(benefits) + calc_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep independently")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        naive_benefits.append(sum(benefits))
        naive_handover_benefits.append(handover_ben)
        naive_handover_violations.append(calc_handover_penalty(assignment_mats, lambda_))

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

        handover_ben = sum(benefits) + calc_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep considering the last one")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        sequential_benefits.append(sum(benefits))
        sequential_handover_benefits.append(handover_ben)

        #solve each timestep sequentially with greedy
        sg_assignment_mats = solve_w_centralized_CBBA(benefit_mats_over_time, lambda_)
        sg_benefit = 0
        for k, sg_assignment_mat in enumerate(sg_assignment_mats):
            sg_benefit += (benefit_mats_over_time[:,:,k]*sg_assignment_mat).sum()

        handover_ben = sg_benefit + calc_handover_penalty(sg_assignment_mats, lambda_)

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
    axs[0].bar(np.arange(len(sequential_benefits))+0.4, sequential_benefits, width=0.2, label='SMGH (Ours)')
    axs[0].set_xticks(np.arange(len(naive_benefits)))
    axs[0].set_xticklabels([str(n) for n in ns])
    axs[0].legend(loc='lower center')

    # Bottom subplot
    axs[1].set_title('Total Benefits, including handover penalty')
    axs[1].set_xlabel('Number of agents')
    axs[1].set_ylabel('Average Benefit')
    axs[1].bar(np.arange(len(naive_handover_benefits)), naive_handover_benefits, width=0.2, label='Naive')
    axs[1].bar(np.arange(len(sga_handover_benefits))+0.2, sga_handover_benefits, width=0.2, label='CBBA')
    axs[1].bar(np.arange(len(sequential_handover_benefits))+0.4, sequential_handover_benefits, width=0.2, label='SMGH (Ours)')
    axs[1].set_xticks(np.arange(len(naive_handover_benefits)))
    axs[1].set_xticklabels([str(n) for n in ns])
    #add a legend to the bottom middle of the subplot
    axs[1].legend(loc='lower center')

    plt.show()

if __name__ == "__main__":
    optimal_baseline_comparison()