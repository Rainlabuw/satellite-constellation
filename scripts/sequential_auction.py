import numpy as np
from methods import *
import networkx as nx
from dist_auction_algo_josh import Auction
from handover_test import *

from constellation_sim.ConstellationSim import get_benefit_matrix_from_constellation, ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

class MultiAuction(object):
    def __init__(self, benefits, init_assignment, max_tstep_lookahead, graph=None, prices=None, verbose=False, approximate=False, lambda_=1):
        # benefit matrix for the next few timesteps
        self.benefits = benefits

        self.n_agents = benefits.shape[0]
        self.n_tasks = benefits.shape[1]
        self.n_timesteps = benefits.shape[2]

        self.max_tstep_lookahead = max_tstep_lookahead

        #price matrix for the next few timesteps
        self.prices = prices
        
        self.graph = graph

        # current assignment
        self.init_assignment = init_assignment
        self.curr_assignment = init_assignment
        
        self.chosen_assignments = []

        self.lambda_ = lambda_

        #If true, constrains each auction solution from being too far from the initial
        #assignment, not the actual assignment resulting from the previous auction.
        self.approximate = approximate

    def calc_benefit(self):
        total_benefit = 0
        for i, chosen_ass in enumerate(self.chosen_assignments):
            curr_benefit = self.benefits[:,:,i]
            total_benefit += (curr_benefit * chosen_ass).sum()
        handover_pen = calc_handover_penalty(self.init_assignment, self.chosen_assignments, self.lambda_)
        total_benefit -= handover_pen

        return total_benefit, handover_pen/self.lambda_

    def run_auctions(self):
        """
        Runs auctions for the next few timesteps, and returns the best assignment.
        """
        while len(self.chosen_assignments) < self.n_timesteps:
            curr_tstep = len(self.chosen_assignments)
            tstep_end = min(curr_tstep+self.max_tstep_lookahead, self.n_timesteps)
            benefit_mat_window = self.benefits[:,:,curr_tstep:tstep_end]

            len_window = benefit_mat_window.shape[-1]

            combined_sols = self.generate_combined_benefit_sols(benefit_mat_window)
            #Evaluate all possible combinations of combined benefit matrix solutions
            global max_benefit, max_sol_key_seq
            max_benefit = -np.inf
            max_sol_key_seq = None

            self.stack_assignments_together(combined_sols, [], len_window)
            
            first_chosen_assignment = combined_sols[max_sol_key_seq[0]].assignment
            self.chosen_assignments.append(first_chosen_assignment)
            self.curr_assignment = first_chosen_assignment

    def stack_assignments_together(self, combined_sols, sol_key_sequence, len_window):
        """
        Recursively constructs all the possible solutions out of the assignment elements we have.
        i.e. if we have solutions to the auctions for timesteps 1, 2, and 12, this method stacks
        computes the benefit from executing assignment (1->2) - the penalty, and compares it to an
        assignment using simply the combined 12 benefit matrix.

        INPUTS:
        combined_sols: dictionary mapping (i,j) to BenefitSolution objects (which contain ben mat, assignment, and computed benefit.)
        sol_key_sequence: list of tuples, where each tuple is an interval (i,j) which corresponds to generating a greedy solution
            for beta^ij.
        """
        global max_benefit, max_sol_key_seq
        if sol_key_sequence == []:
            most_recent_timestep = -1
        else:
            most_recent_timestep = sol_key_sequence[-1][-1]

        #When we have an assignment which ends at the last timestep, we can compute the total benefit
        if most_recent_timestep == (len_window-1):
            #Compute the series of optimal solutions
            if not self.approximate:
                combined_sols = self.calc_sols_from_sol_key_sequence(combined_sols, sol_key_sequence)

            total_benefit = sum([combined_sols[ass].benefit for ass in sol_key_sequence])

            #create list of assignments to use in calc_handover_penalty()
            assignment_sequence = [combined_sols[ass].assignment for ass in sol_key_sequence]

            #take into account losses from handover
            total_benefit -= calc_handover_penalty(self.curr_assignment, assignment_sequence, self.lambda_)
            
            # print(sol_key_sequence, total_benefit)

            if total_benefit > max_benefit:
                max_sol_key_seq = sol_key_sequence
                max_benefit = total_benefit
        else:
            #Iterate through all of the solutions, looking for assignments that start where this one ended
            for sol_key in combined_sols.keys():
                if most_recent_timestep == sol_key[0]-1:
                    self.stack_assignments_together(combined_sols, sol_key_sequence + [sol_key], len_window)

    def calc_sols_from_sol_key_sequence(self, combined_sols, sol_key_sequence):
        """
        Given a list of (i,j) tuples, calculates the benefit and assignment
        for the combined benefit matrix from the benefit matrices at each (i,j) timestep.
        """
        curr_assignment = None
        for sol_key in sol_key_sequence:
            combined_sol = combined_sols[sol_key]
            if (combined_sol.assignment is None) or (combined_sol.benefit is None):
                adj_combined_benefit_mat = self.adjust_benefit_mat_based_on_curr_assign(curr_assignment, combined_sol.benefit_mat)

                #Solve auction and generate assignment for combined benefit matrix
                a = Auction(self.n_agents, self.n_tasks, benefits=adj_combined_benefit_mat)
                _, c, _ = a.solve_centralized()
                combined_sol.assignment = convert_central_sol_to_assignment_mat(self.n_agents, self.n_tasks, c)
                
                combined_sol.benefit = (combined_sol.assignment * combined_sol.benefit_mat).sum()
                
                combined_sols[sol_key] = combined_sol

            #Set the current assignment to be the assignment from this timestep
            curr_assignment = combined_sol.assignment

        return combined_sols
    
    def generate_combined_benefit_sols(self, benefit_mat_window):
        """
        Generates all possible combined benefit matrices from the next few timesteps.

        If self.approximate is True, goes ahead and also computes the assignment and benefit
        by constraining the assignment in each timestep to be near the initial assignment.

        If self.approximate is False, does not solve for the optimal assignment and benefit yet:
        instead waits to recalculate this after we know what assignment it follows, and constrains
        the distance to this assignment.
        """
        curr_tstep_lookahead = benefit_mat_window.shape[-1]
        combined_benefit_sols = {}
        for i in range(curr_tstep_lookahead):
            for j in range(i, curr_tstep_lookahead):
                key = (i,j)
                combined_benefit_mat = benefit_mat_window[:,:,i:j+1].sum(axis=-1)

                #If this is the first timestep or we're approximating, precalculate
                #assignment and benefits by constraining to be near init assignment
                if i == 0 or self.approximate:
                    adj_combined_benefit_mat = self.adjust_benefit_mat_based_on_curr_assign(self.curr_assignment, combined_benefit_mat)

                    #Solve auction and generate assignment for combined benefit matrix
                    graph = nx.complete_graph(self.n_agents)
                    a = Auction(self.n_agents, self.n_tasks, benefits=adj_combined_benefit_mat, graph=graph)
                    _, c, _ = a.solve_centralized()
                    combined_assignment = convert_central_sol_to_assignment_mat(self.n_agents, self.n_tasks, c)

                    benefit_score = (combined_assignment * combined_benefit_mat).sum()
                    # print(benefit_score, (combined_assignment * adj_combined_benefit_mat).sum())
                
                #Otherwise, set benefit_score and combined_assignment to None so we can calculate it later
                else:
                    benefit_score = None
                    combined_assignment = None

                combined_benefit_sols[key] = BenefitSolution(combined_benefit_mat, combined_assignment, benefit_score)
            
        return combined_benefit_sols
    
    def adjust_benefit_mat_based_on_curr_assign(self, curr_assignment, benefit_mat):
        """
        Adjust benefit matrix to incentivize agents being assigned the same task twice.
        """
        if curr_assignment is None:
            return benefit_mat
        else:
            adjusted_benefit_mat = np.where(curr_assignment == 1, benefit_mat, benefit_mat - self.lambda_)
            return adjusted_benefit_mat

class BenefitSolution(object):
    def __init__(self, benefit_mat, assignment, benefit):
        self.benefit_mat = benefit_mat
        self.assignment = assignment
        self.benefit = benefit

    def __repr__(self) -> str:
        return f"Benefit: {self.benefit},\nAssignment:\n{self.assignment},\nCombined benefit matrix:\n{self.benefit_mat}"

if __name__ == "__main__":
    # # Case where no solutions are the best.
    # benefits = np.zeros((4,4,2))
    # benefits[:,:,0] = np.array([[100, 1, 0, 0],
    #                             [1, 100, 0, 0],
    #                             [0, 0, 0.2, 0.1],
    #                             [0, 0, 0.1, 0.2]])
    
    # benefits[:,:,1] = np.array([[1, 1000, 0, 0],
    #                             [1000, 1, 0, 0],
    #                             [0, 0, 0.1, 0.3],
    #                             [0, 0, 0.3, 0.1]])

    # print("Expect no solution to be optimal (2198.8) but them to be same for all lookaheads")
    # for lookahead in range(1,benefits.shape[-1]+1):
    #     multi_auction = MultiAuction(benefits, None, lookahead)
    #     multi_auction.run_auctions()
    #     ben = multi_auction.calc_benefit()
    #     print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

    # #Case where a combined solution is the best.
    # benefits = np.zeros((3,3,3))
    # benefits[:,:,0] = np.array([[0.1, 0, 0],
    #                             [0, 0.1, 0],
    #                             [0, 0, 0.1]])
    # benefits[:,:,1] = np.array([[0, 0, 0.1],
    #                             [0.1, 0, 0],
    #                             [0, 0.1, 0]])
    # benefits[:,:,2] = np.array([[0.1, 1000, 0],
    #                             [0, 0.1, 1000],
    #                             [1000, 0, 0.1]])

    # print("Expect combined solution to be optimal (3000) only at lookahead of 3")
    # for lookahead in range(1,benefits.shape[-1]+1):
    #     multi_auction = MultiAuction(benefits, None, lookahead)
    #     multi_auction.run_auctions()
    #     ben,_ = multi_auction.calc_benefit()
    #     print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

    # seed = np.random.randint(0, 1000)
    # # np.random.seed(624)
    # n = 10
    # m = 10
    # T = 5
    # i = 0
    # np.random.seed(42)
    # while True:
    #     i += 1
    #     print(i, end='\r')
    #     bens = []
    #     benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=1.01)
    #     for lookahead in range(1,benefits.shape[-1]+1):
    #         multi_auction = MultiAuction(benefits, None, lookahead)
    #         multi_auction.run_auctions()
    #         ben, nh = multi_auction.calc_benefit()
    #         bens.append(ben)
    


    # #Case where we expect solutions to get increasingly better as lookahead window increases
    # n = 50
    # m = 50
    # T = 95
    # lambda_ = 1
    # # np.random.seed(42)

    # resulting_bens = []
    # resulting_approx_bens = []
    # naive_benefits = []
    # smgh_benefits = []
    # handovers = []
    # print("Comparing performance of SMGHL to other algorithms")
    # max_lookahead = 5
    # num_avgs = 10

    # smghl2_ben = 0
    # smghl2_nh = 0

    # smghl5_ben = 0
    # smghl5_nh = 0

    # smgh_ben = 0
    # smgh_nh = 0

    # naive_ben = 0
    # naive_nh = 0

    # sga_ben = 0
    # sga_nh = 0
    # for _ in range(num_avgs):
    #     print(f"Run {_}/{num_avgs}")
    #     # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
    #     benefits = get_benefit_matrix_from_constellation(n, m, T)
    #     print("Generated realistic benefits")

    #     #SMHGL centralized, lookahead = 2
    #     multi_auction = MultiAuction(benefits, None, 2, lambda_=lambda_)
    #     multi_auction.run_auctions()
    #     ben, nh = multi_auction.calc_benefit()
    #     smghl2_ben += ben/num_avgs
    #     smghl2_nh += nh/num_avgs

    #     #SMHGL centralized, lookahead = 5
    #     multi_auction = MultiAuction(benefits, None, 5, lambda_=lambda_)
    #     multi_auction.run_auctions()
    #     ben, nh = multi_auction.calc_benefit()
    #     smghl5_ben += ben/num_avgs
    #     smghl5_nh += nh/num_avgs

    #     #SMGH
    #     multi_auction = MultiAuction(benefits, None, 1, lambda_=lambda_)
    #     multi_auction.run_auctions()
    #     ben, nh = multi_auction.calc_benefit()
    #     smgh_ben += ben/num_avgs
    #     smgh_nh += nh/num_avgs

    #     #Naive
    #     ben, nh = solve_naively(benefits, lambda_)
    #     naive_ben += ben/num_avgs
    #     naive_nh += nh/num_avgs

    #     # #SGA/CBBA
    #     # sg_assignment_mats = sequential_greedy(benefits, lambda_)
    #     # sg_benefit = 0
    #     # for k, sg_assignment_mat in enumerate(sg_assignment_mats):
    #     #     sg_benefit += (benefits[:,:,k]*sg_assignment_mat).sum()

    #     # handover_ben = sg_benefit - calc_handover_penalty(None, sg_assignment_mats, lambda_)
    #     # sga_ben += handover_ben/num_avgs
    #     # sga_nh += calc_handover_penalty(None, sg_assignment_mats, lambda_)/lambda_/num_avgs

    # fig, axes = plt.subplots(2,1)
    # axes[0].bar(["Naive", "SGA", "SMGH", "SMGHL2", "SMGHL5"],[naive_ben, sga_ben, smgh_ben, smghl2_ben, smghl5_ben])
    # axes[0].set_title("Average benefit across 10 runs")
    # # axes[0].set_xlabel("Lookahead timesteps")

    # axes[1].bar(["Naive", "SGA", "SMGH", "SMGHL2", "SMGHL5"],[naive_nh, sga_nh, smgh_nh, smghl2_nh, smghl5_nh])
    
    # axes[1].set_title("Average number of handovers across 10 runs")
    # fig.suptitle(f"Test with n={n}, m={m}, T={T}, lambda={lambda_}, realistic-ish benefits")

    # plt.show()

    # #TESTING FOR HOW PERFORMANCE INCREASES AS LOOKAHAED INCREASES
    # print("Expect performance to generally increase as lookahead increases")

    # for lookahead in range(1,max_lookahead+1):
    #     avg_ben = 0
    #     avg_nh = 0
    #     avg_approx_ben = 0
    #     for _ in range(num_avgs):
    #         print(f"Lookahead {lookahead} ({_}/{num_avgs})", end='\r')
    #         # benefits = generate_benefits_over_time(n, m, 10, T, scale_min=1, scale_max=2)
    #         benefits = get_benefit_matrix_from_constellation(n, m, T)
    #         # benefits = np.random.rand(n,m,T)

    #         #SMGHL with true lookaheads
    #         multi_auction = MultiAuction(benefits, None, lookahead, lambda_=lambda_)
    #         multi_auction.run_auctions()
    #         ben, nh = multi_auction.calc_benefit()
    #         avg_ben += ben/num_avgs
    #         avg_nh += nh/num_avgs
            
    #         #SMGHL (distributed)
    #         multi_auction = MultiAuction(benefits, None, lookahead, lambda_=lambda_, approximate=True)
    #         multi_auction.run_auctions()
    #         ben, _ = multi_auction.calc_benefit()
    #         avg_approx_ben += ben/num_avgs

    #     resulting_bens.append(avg_ben)
    #     resulting_approx_bens.append(avg_approx_ben)
    #     handovers.append(avg_nh)

    # plt.plot(range(1,max_lookahead+1), resulting_bens, label="SMGHL (Centralized)")
    # plt.plot(range(1,max_lookahead+1), resulting_approx_bens, label="SMGHL (Distributed)")
    # plt.title(f"Lookahead vs. accuracy, n={n}, m={m}, T={T}")
    # plt.xlabel("Lookahead timesteps")
    # plt.ylabel(f"Average benefit across {num_avgs} runs")
    # plt.legend()
    # plt.savefig("lookahead_vs_benefit.png")
    # plt.show()

    # plt.figure()
    # plt.plot(range(1,max_lookahead+1), handovers)
    # plt.title("Num handovers vs. lookahead")
    # plt.show()

    # Testing plotting
    const = ConstellationSim(dt=1*u.min)

    T = int(95 // const.dt.to_value(u.min)) #simulate enough timesteps for ~1 orbit
    T = 25
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #5 evenly spaced planes of satellites, each with 10 satellites per plane
    a = earth.R.to(u.km) + 400*u.km
    ecc = 0.01*u.one
    inc = 58*u.deg
    argp = 0*u.deg

    num_planes = 10
    num_sats_per_plane = 5
    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num)
            const.add_sat(sat)

    #~~~~~~~~~Generate 5 random tasks on the surface of earth~~~~~~~~~~~~~
    num_tasks = 50

    for i in range(num_tasks):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-50, 50)
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1,2)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    const.propagate_orbits(T)

    const.assign_over_time = [np.eye(const.n, const.m) for i in range(T)]

    multi_auction = MultiAuction(const.benefits_over_time, None, 5, lambda_=1)
    multi_auction.run_auctions()

    const.assign_over_time = multi_auction.chosen_assignments
    print(const.assign_over_time)
    const.run_animation(frames=T)