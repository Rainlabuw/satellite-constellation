import numpy as np
from methods import *
import networkx as nx
from dist_auction_algo_josh import Auction
from handover_test import *

class MultiAuction(object):
    def __init__(self, benefits, init_assignment, max_tstep_lookahead, graph=None, prices=None, verbose=False, lambda_=0.5):
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

    def calc_benefit(self):
        total_benefit = 0
        for i, chosen_ass in enumerate(self.chosen_assignments):
            curr_benefit = self.benefits[:,:,i]
            total_benefit += (curr_benefit * chosen_ass).sum()
        total_benefit += calc_handover_penalty([self.init_assignment] + self.chosen_assignments, self.lambda_)

        return total_benefit


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
            # print(f"final {max_sol_key_seq}")
            
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
        sol_key_sequence: list of tuples, where each tuple is an assignment (i,j) which is a solution to the auction for timestep i.
        """
        global max_benefit, max_sol_key_seq
        if sol_key_sequence == []:
            most_recent_timestep = -1
        else:
            most_recent_timestep = sol_key_sequence[-1][-1]

        #When we have an assignment which ends at the last timestep, we can compute the total benefit
        if most_recent_timestep == (len_window-1):
            #compute sum of the benefits
            total_benefit = sum([combined_sols[ass].benefit for ass in sol_key_sequence])
            
            if self.curr_assignment is None:
                init_ass = combined_sols[sol_key_sequence[0]].assignment
            else: init_ass = self.curr_assignment

            assignment_sequence = [init_ass] + [combined_sols[ass].assignment for ass in sol_key_sequence]

            #take into account losses from handover
            total_benefit += calc_handover_penalty(assignment_sequence, self.lambda_)
            
            # print(sol_key_sequence, total_benefit)

            if total_benefit > max_benefit:
                max_sol_key_seq = sol_key_sequence
                max_benefit = total_benefit
        else:
            #Iterate through all of the solutions, looking for assignments that start where this one ended
            for sol_key in combined_sols.keys():
                if most_recent_timestep == sol_key[0]-1:
                    self.stack_assignments_together(combined_sols, sol_key_sequence + [sol_key], len_window)

    def generate_combined_benefit_sols(self, benefit_mat_window):
        """
        Generates all possible combined benefit matrices from the next few timesteps.
        """
        curr_tstep_lookahead = benefit_mat_window.shape[-1]
        combined_benefit_sols = {}
        for i in range(curr_tstep_lookahead):
            for j in range(i, curr_tstep_lookahead):
                key = (i,j)
                combined_benefit_mat = benefit_mat_window[:,:,i:j+1].sum(axis=-1)
                adj_combined_benefit_mat = self.adjust_benefit_mat_based_on_curr_assign(combined_benefit_mat)


                #Solve auction and generate assignment for combined benefit matrix
                graph = nx.complete_graph(self.n_agents)
                a = Auction(self.n_agents, self.n_tasks, benefits=adj_combined_benefit_mat, graph=graph)
                _, c, _ = a.solve_centralized()
                combined_assignment = convert_central_sol_to_assignment_mat(self.n_agents, self.n_tasks, c)
                
                benefit_score = (combined_assignment * combined_benefit_mat).sum()
                # print(benefit_score, (combined_assignment * adj_combined_benefit_mat).sum())

                combined_benefit_sols[key] = BenefitSolution(adj_combined_benefit_mat, combined_assignment, benefit_score)
            
        return combined_benefit_sols
    
    def adjust_benefit_mat_based_on_curr_assign(self, benefit_mat):
        """
        Adjusts the benefit matrix based on the current assignment, so that the agents
        don't get assigned the same task twice.
        """
        if self.curr_assignment is None:
            return benefit_mat
        else:
            adjusted_benefit_mat = np.where(self.curr_assignment == 1, benefit_mat, benefit_mat - self.lambda_*2)
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
    # benefits[:,:,1] = np.array([[0.1, 0, 0],
    #                             [0, 0.1, 0],
    #                             [0, 0, 0.1]])
    # benefits[:,:,2] = np.array([[0.1, 1000, 0],
    #                             [0, 0.1, 1000],
    #                             [1000, 0, 0.1]])

    # print("Expect combined solution to be optimal (3000) only at lookahead of 3")
    # for lookahead in range(1,benefits.shape[-1]+1):
    #     multi_auction = MultiAuction(benefits, None, lookahead)
    #     multi_auction.run_auctions()
    #     ben = multi_auction.calc_benefit()
    #     print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")
    # np.random.seed(45)
    # n = 50
    # m = 50
    # T = 25
    # benefits = np.random.rand(n,m,T)
    
    # for lookahead in range(1,benefits.shape[-1]+1):
    #     multi_auction = MultiAuction(benefits, None, lookahead)
    #     multi_auction.run_auctions()
    #     ben = multi_auction.calc_benefit()
    #     print(f"~~~~Benefit from combined solution, lookahead {lookahead}: {ben}~~~~")

    #Case where we expect solutions to get icnreasingly better as lookahead window increases
    n = 50
    m = 50
    T = 25
    np.random.seed(42)

    resulting_bens = []
    print("Expect combined solutions to get better as lookahead increases")
    max_lookahead = 10
    num_avgs = 100

    for lookahead in range(1,max_lookahead+1):
        avg_ben = 0
        for _ in range(num_avgs):
            print(f"Lookahead {lookahead} ({_}/{num_avgs})", end='\r')
            benefits = generate_benefits_over_time(n, m, 10, T)
            multi_auction = MultiAuction(benefits, None, lookahead, lambda_=0.5)
            multi_auction.run_auctions()
            ben = multi_auction.calc_benefit()
            avg_ben += ben/num_avgs
            # print(f"\tBenefit from combined solution, lookahead {lookahead}: {ben}")

        resulting_bens.append(avg_ben)

    plt.plot(range(1,max_lookahead+1), resulting_bens)
    plt.title(f"Lookahead vs. accuracy, n={n}, m={m}, T={T}")
    plt.xlabel("Lookahead timesteps")
    plt.ylabel(f"Average benefit across {num_avgs} runs")
    plt.savefig("lookahead_vs_benefit.png")
    plt.show()


    #solve each timestep sequentially
    assignment_mats = []
    benefits_hist = []
    lambda_ = 0.5
    
    #solve first timestep separately
    graph = nx.complete_graph(n)
    a = Auction(n, m, benefits=benefits[:,:,0], graph=graph)
    benefit = a.run_auction()

    assignment_mat = convert_agents_to_assignment_matrix(a.agents)
    assignment_mats.append(assignment_mat)
    benefits_hist.append(benefit)

    prev_assignment_mat = assignment_mat
    for k in range(1, T):
        print(k, end='\r')
        #Generate assignment for the task minimizing handover
        benefit_mat_w_handover = add_handover_pen_to_benefit_matrix(benefits[:,:,k], prev_assignment_mat, lambda_)

        a = Auction(n, m, benefits=benefit_mat_w_handover, graph=graph)
        a.run_auction()
        choices = [ag.choice for ag in a.agents]

        assignment_mat = convert_agents_to_assignment_matrix(a.agents)
        assignment_mats.append(assignment_mat)

        prev_assignment_mat = assignment_mat

        #Calculate the benefits from a task with the normal benefit matrix
        benefit = benefits[:,:,k]*assignment_mat

        benefits_hist.append(benefit.sum())

    handover_ben = sum(benefits_hist) + calc_handover_penalty(assignment_mats, lambda_)
    print("Solving sequentially, each timestep considering the last one")
    print(f"\tBenefit without considering handover: {sum(benefits_hist)}")
    print(f"\tBenefit with handover penalty: {handover_ben}")
