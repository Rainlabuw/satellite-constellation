import numpy as np
from methods import *
import networkx as nx
from classic_auction import Auction, AuctionAgent
from solve_w_mhal import MHAL_Auction

from tqdm import tqdm
import time

class MHAL_D_Auction(object):
    def __init__(self, benefits, curr_assignment, all_time_intervals, all_time_interval_sequences, prices=None, eps=0.01, graph=None, lambda_=1, verbose=False):
        # benefit matrix for the next few timesteps
        self.benefits = benefits

        if prices is None:
            self.prices = np.zeros(benefits.shape)
        else:
            self.prices = prices

        self.n = benefits.shape[0]
        self.m = benefits.shape[1]
        self.T = benefits.shape[2]

        if graph is None:
            self.graph = nx.complete_graph(self.n)
        else:
            self.graph = graph

        self.curr_assignment = curr_assignment
        
        self.chosen_assignments = []

        self.lambda_ = lambda_

        self.eps = eps
        self.verbose = verbose

        self.agents = [MHAL_D_Agent(self, i, all_time_intervals, all_time_interval_sequences, \
                                    self.benefits[i,:,:], self.prices[i,:,:], list(self.graph.neighbors(i)), self.eps, nx.diameter(self.graph)) for i in range(self.n)]

    def run_auction(self):
        self.n_iterations = 0
        while sum([agent.converged for agent in self.agents]) < self.n:
            for agent in self.agents:
                agent.update_agent_prices_bids()

            for agent in self.agents:
                agent.publish_agent_prices_bids()
            
            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")

class MHAL_D_Agent(object):
    def __init__(self, auction, id, all_time_intervals, all_time_interval_sequences, benefits, prices, neighbors, eps, max_steps_since_last_update):
        self.id = id
        self.auction = auction

        self.all_time_intervals = all_time_intervals
        self.all_time_interval_sequences = all_time_interval_sequences

        #Benefits and prices are mxL matrices.
        self.benefits = benefits
        self.init_time_interval_benefits()
        self.prices = prices
        self.eps = eps

        self.m = benefits.shape[0]
        self.T = benefits.shape[1]

        self._high_bidders = {}
        self.public_high_bidders = {}
        self._prices = {}
        self.public_prices = {}
        self.choice_by_ti = {}
        for time_interval in all_time_intervals:
            self._high_bidders[time_interval] = -1*np.ones(self.m)
            self._high_bidders[time_interval][0] = self.id
            self.public_high_bidders[time_interval] = -1*np.ones(self.m)
            self.public_high_bidders[time_interval][0] = self.id

            self._prices[time_interval] = np.zeros(self.m)
            self.public_prices[time_interval] = np.zeros(self.m)

            self.choice_by_ti[time_interval] = 0

        #Stores the benefit yielded by time interval sequences
        self._benefits_from_time_interval_seq = {}
        self.public_benefits_from_time_interval_seq = {}
        for time_interval_sequence in all_time_interval_sequences:
            self._benefits_from_time_interval_seq[time_interval_sequence] = 0
            self.public_benefits_from_time_interval_seq[time_interval_sequence] = 0

        self.neighbors = neighbors

        self.steps_since_last_update = 0
        self.max_steps_since_last_update = max_steps_since_last_update
        self.converged = False

        #Final task choice selected by the algorithm
        self.choice = None

    def init_time_interval_benefits(self):
        """
        Generate dictionary which contains combined benefits for each time interval.

        i.e. adds up the benefits over the entire time interval.
        """
        self.time_interval_benefits = {}
        for ti in self.all_time_intervals:
            combined_benefits = self.benefits[:,ti[0]:ti[1]+1].sum(axis=-1)

            self.time_interval_benefits[ti] = combined_benefits

    def compute_time_interval_sequence_benefits(self):
        for time_interval_sequence in self.all_time_interval_sequences:
            tis_benefit = 0
            curr_assignment = np.argmax(self.auction.curr_assignment[self.id,:])
            for time_interval in time_interval_sequence:
                tis_benefit += self.time_interval_benefits[time_interval][self.choice_by_ti[time_interval]]
                
                if curr_assignment != self.choice_by_ti[time_interval]:
                    tis_benefit -= self.auction.lambda_

                curr_assignment = self.choice_by_ti[time_interval]
            
            self._benefits_from_time_interval_seq[time_interval_sequence] = tis_benefit

    def update_agent_prices_bids(self):
        """
        Updates the agent's prices and bids.
        """
        if self.steps_since_last_update < self.max_steps_since_last_update:
            #Update the agent's prices and bids
            for ti in self.all_time_intervals:
                neighbor_prices = np.array(self.public_prices[ti])
                highest_bidders = np.array(self.public_high_bidders[ti])
                for n in self.neighbors:
                    neighbor_prices = np.vstack((neighbor_prices, self.auction.agents[n].public_prices[ti]))
                    highest_bidders = np.vstack((highest_bidders, self.auction.agents[n].public_high_bidders[ti]))

                max_prices = np.max(neighbor_prices, axis=0)
                
                # Filter the high bidders by the ones that have the max price, and set the rest to -1.
                # Grab the highest index max bidder to break ties.
                max_price_bidders = np.where(neighbor_prices == max_prices, highest_bidders, -1)
                self._high_bidders[ti] = np.max(max_price_bidders, axis=0)

                if max_prices[self.choice_by_ti[ti]] >= self.public_prices[ti][self.choice_by_ti[ti]] and self._high_bidders[ti][self.choice_by_ti[ti]] != self.id:
                    #Adjust the combined benefits to enforce handover penalty 
                    indices_where_agent_not_prev_assigned = np.where(self.auction.curr_assignment[self.id,:]==1, 0, 1)
                    benefit_hat = self.time_interval_benefits[ti] - self.auction.lambda_ * indices_where_agent_not_prev_assigned

                    best_net_value = np.max(benefit_hat - max_prices)
                    second_best_net_value = np.partition(benefit_hat - max_prices, -2)[-2] #https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix

                    self.choice_by_ti[ti] = np.argmax(benefit_hat-max_prices) #choose the task with the highest benefit to the agent

                    self._high_bidders[ti][self.choice_by_ti[ti]] = self.id
                    
                    inc = best_net_value - second_best_net_value + self.eps

                    self._prices[ti] = max_prices
                    self._prices[ti][self.choice_by_ti[ti]] = max_prices[self.choice_by_ti[ti]] + inc
                else:
                    #Otherwise, don't change anything and just update prices
                    #based on the new info from other agents.
                    self._prices[ti] = max_prices

            #Based on the new choices for each time interval, calculate the time interval sequence benefits
            self.compute_time_interval_sequence_benefits()
        
        #Otherwise, the algorithm has converged
        else:
            self.converged = True

            best_tis_benefit = -np.inf
            best_tis = None

            #NOTE: Need to get this information in a distributed way
            for tis in self.all_time_interval_sequences:
                tis_benefit = 0
                for agent in self.auction.agents:
                    tis_benefit += agent.public_benefits_from_time_interval_seq[tis]

                if tis_benefit > best_tis_benefit:
                    best_tis_benefit = tis_benefit
                    best_tis = tis

            self.choice = self.choice_by_ti[best_tis[0]]

    def publish_agent_prices_bids(self):
        if self.steps_since_last_update < self.max_steps_since_last_update:
            updated = False
            for ti in self.all_time_intervals:
                if not np.array_equal(self._prices[ti], self.public_prices[ti]) or \
                    not np.array_equal(self._high_bidders[ti], self.public_high_bidders[ti]):
                    updated = True

                self.public_prices[ti] = np.copy(self._prices[ti])
                self.public_high_bidders[ti] = np.copy(self._high_bidders[ti])

            for tis in self.all_time_interval_sequences:
                self.public_benefits_from_time_interval_seq[tis] = self._benefits_from_time_interval_seq[tis]

            if not updated:
                self.steps_since_last_update += 1
            else:
                self.steps_since_last_update = 0
        
        #Otherwise, the algorithm has converged
        else:
            self.converged = True


def choose_time_interval_sequence_centralized(time_interval_sequences, prev_assignment, benefit_mat_window, lambda_, approx=False):
    """
    Chooses the best time interval sequence from a list of time interval sequences,
    and return the corresponding assignment.
    """
    n = benefit_mat_window.shape[0]
    m = benefit_mat_window.shape[1]

    best_benefit = -np.inf
    best_assignment = None
    best_time_interval = None

    for time_interval_sequence in time_interval_sequences:
        total_tis_benefit = 0
        tis_assignment_curr = prev_assignment
        tis_first_assignment = None

        for i, time_interval in enumerate(time_interval_sequence):
            #Calculate combined benefit matrix from this time interval
            combined_benefit_mat = benefit_mat_window[:,:,time_interval[0]:time_interval[1]+1].sum(axis=-1)

            #Adjust the benefit matrix to incentivize agents being assigned the same task twice.
            #Note that if we're not approximating, we incentivize staying close to the previous assignment calculated during this
            #time interval sequence, not the actual assignment that the agents currently have (i.e. prev_assignment)
            benefit_hat = add_handover_pen_to_benefit_matrix(combined_benefit_mat, tis_assignment_curr, lambda_)
            if approx: benefit_hat = add_handover_pen_to_benefit_matrix(combined_benefit_mat, prev_assignment, lambda_)

            #Generate an assignment using a centralized solution.
            central_assignments = solve_centralized(benefit_hat)
            tis_assignment = convert_central_sol_to_assignment_mat(n, m, central_assignments)

            #Calculate the benefit from this time interval sequence
            total_tis_benefit += (tis_assignment * combined_benefit_mat).sum()
            total_tis_benefit -= calc_assign_seq_handover_penalty(tis_assignment_curr, [tis_assignment], lambda_)

            tis_assignment_curr = tis_assignment
            #If this is the first assignment in the time interval sequence, save it
            if i == 0:
                tis_first_assignment = tis_assignment_curr

        #If this time interval sequence is better than the best one we've seen so far, save it 
        if total_tis_benefit > best_benefit:
            best_benefit = total_tis_benefit
            best_assignment = tis_first_assignment
            best_time_interval = time_interval_sequence

    return best_assignment

def build_time_interval_sequences(all_time_intervals, time_interval_sequence, len_window):
    """
    Recursively constructs all the possible solutions out of the assignment elements we have.
    i.e. if we have solutions to the auctions for timesteps 1, 2, and 12, this method stacks
    computes the benefit from executing assignment (1->2) - the penalty, and compares it to an
    assignment using simply the combined 12 benefit matrix.

    INPUTS:
    all_time_intervals: list of (i,j) tuples, where i is the start timestep and j is the end timestep
    """
    global all_time_interval_sequences
    #Grab the most recent timestep from the end of the current sol
    if time_interval_sequence == []:
        most_recent_timestep = -1 #set it to -1 so that time intervals starting w 0 will be selected
    else:
        most_recent_timestep = time_interval_sequence[-1][-1]

    #When we have an assignment which ends at the last timestep, we can compute the total benefit
    if most_recent_timestep == (len_window-1):
        #If we're not approximating, we still need to calculate the assignment and benefit for each
        #(if we are approximating, we already calculated this up front)
        all_time_interval_sequences.append(tuple(time_interval_sequence))
    else:
        #Iterate through all of the solutions, looking for assignments that start where this one ended
        for time_interval in all_time_intervals:
            if most_recent_timestep == time_interval[0]-1:
                build_time_interval_sequences(all_time_intervals, time_interval_sequence + [time_interval], len_window)

def generate_all_time_intervals(L):
    """
    Generates all possible combined benefit matrices from the next few timesteps.
    """
    all_time_intervals = []
    for i in range(L):
        for j in range(i,L):
            all_time_intervals.append((i,j))
        
    return all_time_intervals

def solve_w_mhal(benefits, L, init_assignment, graphs=None, lambda_=1, distributed=False, central_approx=False):
    """
    Sequentially solves the problem using the MHAL algorithm.

    When distributed = True, computes the solution using the fully distributed method.
    When central_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as would be done in the distributed version.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    if graphs is None:
        graphs = [nx.complete_graph(n) for i in range(T)]

    curr_assignment = init_assignment
    
    chosen_assignments = []

    while len(chosen_assignments) < T:
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        global all_time_interval_sequences
        all_time_interval_sequences = []
        build_time_interval_sequences(all_time_intervals, [], len_window)

        if not distributed:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, benefit_mat_window, lambda_, approx=central_approx)
        else:
            mhal_d_auction = MHAL_D_Auction(benefit_mat_window, curr_assignment, all_time_intervals, all_time_interval_sequences, eps=0.01, graph=graphs[curr_tstep], lambda_=lambda_)
            mhal_d_auction.run_auction()

            chosen_assignment = convert_agents_to_assignment_matrix(mhal_d_auction.agents)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
            
    return chosen_assignments

if __name__ == "__main__":
    n = 20
    m = 20
    T = 5
    L = 4

    graphs = [rand_connected_graph(n) for i in range(T)]

    ctot = 0
    catot = 0
    dtot = 0
    n_tests = 100
    for _ in tqdm(range(n_tests)):
        benefits = np.random.rand(n, m, T)
        init_assignment = np.eye(n,m)

        # st = time.time()
        dcas = solve_w_mhal(benefits, L, init_assignment, distributed=True, lambda_=0.5, graphs=graphs)
        d_val = calc_value_and_num_handovers(dcas, benefits, init_assignment, 0.5)[0]
        # print(f"Time elapsed (distributed) = {time.time()-st}")

        cas = solve_w_mhal(benefits, L, init_assignment, distributed=False, lambda_=0.5)
        c_val = calc_value_and_num_handovers(cas, benefits, init_assignment, 0.5)[0]

        caas = solve_w_mhal(benefits, L, init_assignment, distributed=False, central_approx=True, lambda_=0.5)
        ca_val = calc_value_and_num_handovers(caas, benefits, init_assignment, 0.5)[0]

        ctot += c_val/n_tests
        catot += ca_val/n_tests
        dtot += d_val/n_tests
        # st = time.time()
        # caas = solve_w_mhal(benefits, L, init_assignment, distributed=False, central_approx=True, lambda_=0.5)
        # ca_val = calc_value_and_num_handovers(caas, benefits, init_assignment, 0.5)[0]
        # print(f"Time elapsed (centralized approx) = {time.time()-st}")

        # for k, (dca, ca) in enumerate(zip(ma.chosen_assignments, cas)):
        #     print(f"Timestep {k}")
        #     print(f"\tMA valid = {check_assign_matrix_validity(dca)}, CA valid = {check_assign_matrix_validity(ca)}")
        #     if not check_assign_matrix_validity(dca) or not check_assign_matrix_validity(ca):
        #         break
        #     for i in range(n):
        #         print(f"\tAgent {i}, MA = {np.argmax(dca[i,:])}, CA = {np.argmax(ca[i,:])}")

        # print(f"DA value = {d_val}, CA value = {c_val}, CAA value = {ca_val}")
        # print(f"Diff = {c_val-ca_val}, n\epsilon = {n*0.01}")

    print(ctot, catot, dtot)

    # chosen_assignments = solve_w_mhal(benefits, 4, init_assignment)
    # print(calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, 1))