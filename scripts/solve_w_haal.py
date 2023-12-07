import numpy as np
from .utils import *
import networkx as nx
import time

class HAAL_D_Auction(object):
    """
    This class runs an assignment auction for all possible time interval sequences
    in a single timestep, all in parallel.
    
    The algorithm stores the assigned task for each agent in their .choice attribute.
    """
    def __init__(self, benefits, curr_assignment, all_time_intervals, all_time_interval_sequences, eps=0.01, graph=None, lambda_=1, verbose=False):
        # benefit matrix for the next L timesteps
        self.benefits = benefits
        self.n = benefits.shape[0]
        self.m = benefits.shape[1]
        self.T = benefits.shape[2]

        #If no graph is provided, assume the communication graph is complete.
        if graph is None:
            self.graph = nx.complete_graph(self.n)
        else:
            self.graph = graph

        self.curr_assignment = curr_assignment
        
        self.chosen_assignments = []

        self.lambda_ = lambda_

        self.eps = eps
        self.verbose = verbose

        #The number of steps without an update before the algorithm converges
        self.max_steps_since_last_update = nx.diameter(self.graph)

        #Build the list of agents participating in the auction, providing them only
        #the benefits they recieve for completing each task and a list of their neighbors.
        self.agents = [HAAL_D_Agent(self, i, all_time_intervals, all_time_interval_sequences, \
                                    self.benefits[i,:,:], list(self.graph.neighbors(i))) for i in range(self.n)]

    def run_auction(self):
        """
        Run the auction to completion. Alternate between calculating updated prices and bids
        for each agent, and formulating and sending packets of information that each agent recieves
        from it's neighbors containing this updated information.
        """
        self.n_iterations = 0
        while sum([agent.converged for agent in self.agents]) < self.n:
            #Send the appropriate communication packets to each agent
            self.update_communication_packets()

            #Have each agent calculate it's prices, bids, and values
            #based on the communication packet it currently has
            for agent in self.agents:
                agent.perform_auction_iteration_for_agent()

            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")

    def update_communication_packets(self):
        """
        Compiles price, high bidder, value (and time value info recieved)
        information for all of each agent's neighbors and stores it in an array.

        Then, update the agents variables accordingly so it can access that information
        during the auction.

        By precomputing these packets, we can parallelize the auction.
        """
        for agent in self.agents:
            price_packets = {}
            high_bidder_packets = {}
            for ti in agent.all_time_intervals:
                price_packet = np.zeros((len(agent.neighbors)+1,self.m), dtype=np.float16)
                high_bidder_packet = np.zeros((len(agent.neighbors)+1,self.m), dtype=np.int16)
                timestep_value_info_recieved_packet = np.zeros((len(agent.neighbors)+1,self.n), dtype=np.int16)

                price_packet[0,:] = agent.prices[ti]
                high_bidder_packet[0,:] = agent.high_bidders[ti]
                
                for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                    price_packet[neighbor_num+1,:] = self.agents[neighbor_idx].prices[ti]
                    high_bidder_packet[neighbor_num+1,:] = self.agents[neighbor_idx].high_bidders[ti]

                price_packets[ti] = price_packet
                high_bidder_packets[ti] = high_bidder_packet

            timestep_value_info_recieved_packet[0,:] = agent.timestep_value_info_recieved
            for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                timestep_value_info_recieved_packet[neighbor_num+1,:] = self.agents[neighbor_idx].timestep_value_info_recieved

            value_from_time_interval_seq_packets = {}
            for time_interval_sequence in agent.all_time_interval_sequences:
                value_from_time_interval_seq_packets[time_interval_sequence] = np.zeros((len(agent.neighbors)+1,self.n), dtype=np.float16)

                value_from_time_interval_seq_packets[time_interval_sequence][0,:] = agent.values_from_time_interval_seq[time_interval_sequence]
                for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                    value_from_time_interval_seq_packets[time_interval_sequence][neighbor_num+1,:] = self.agents[neighbor_idx].values_from_time_interval_seq[time_interval_sequence]

            agent.price_comm_packets = price_packets
            agent.high_bidder_comm_packets = high_bidder_packets
            agent.timestep_value_info_recieved_comm_packet = timestep_value_info_recieved_packet
            agent.value_from_time_interval_seq_comm_packets = value_from_time_interval_seq_packets


class HAAL_D_Agent(object):
    def __init__(self, auction, id, all_time_intervals, all_time_interval_sequences, benefits, neighbors):
        self.id = id

        #Grab info from auction
        self.init_assignment = auction.curr_assignment
        self.lambda_ = auction.lambda_
        self.n = auction.n
        self.eps = auction.eps
        self.max_steps_since_last_update = auction.max_steps_since_last_update

        self.all_time_intervals = all_time_intervals
        self.all_time_interval_sequences = all_time_interval_sequences

        #Benefits and prices are mxL matrices.
        self.benefits = benefits
        self.init_time_interval_benefits()

        self.m = benefits.shape[0]
        self.T = benefits.shape[1]

        #~~~~~~~Attributes which the agent uses to run the auction, and which it publishes to other agents~~~~~~~~
        self.high_bidders = {}
        self.prices = {}
        self.choice_by_ti = {}
        #Generate bids and prices for every time interval you're running an auction for
        for time_interval in all_time_intervals:
            #Initialize yourself as the highest bidder on the first task
            self.high_bidders[time_interval] = -1*np.ones(self.m, dtype=np.int16)
            self.high_bidders[time_interval][0] = self.id

            self.prices[time_interval] = np.zeros(self.m, dtype=np.float16)

            self.choice_by_ti[time_interval] = 0

        #Stores the value yielded by time interval sequences
        self.values_from_time_interval_seq = {}
        for time_interval_sequence in all_time_interval_sequences:
            self.values_from_time_interval_seq[time_interval_sequence] = np.zeros(self.n, dtype=np.float16)

        self.timestep_value_info_recieved = np.zeros(self.n, dtype=np.int16)

        #~~~~~~~~Communication packet related attributes~~~~~~~~~~
        self.neighbors = neighbors

        #price and high bidder packets are (num neighbor x m) matrices,
        #one for each different time interval
        self.price_comm_packets = None
        self.high_bidder_comm_packets = None

        #timestep_value_info_recieved is a (num neighbor x n) matrix
        self.timestep_value_info_recieved_comm_packet = None

        #value_from_time_interval_seq_comm_packets is a dictionary of (num neighbor x n) matrices,
        #each key corresponding to a different time interval sequence
        self.value_from_time_interval_seq_comm_packets = None

        #~~~~~~~~~~~~~~Convergence related attributes~~~~~~~~~~~~~~~~
        self.steps_since_last_update = 0
        self.converged = False

        self.n_iters = 0

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

    def perform_auction_iteration_for_agent(self):
        """
        After recieving an updated communication packet, runs a single iteration
        of the auctions for the agents. Consists of the following steps:

        1. 
        """
        self.update_prices_bids()

        self.update_time_interval_sequence_values()

        #Set the agents current choice to the choice for the best time interval sequence
        best_tis_value = -np.inf
        best_tis = None
        for tis in self.all_time_interval_sequences:
            tis_value = np.sum(self.values_from_time_interval_seq[tis])

            if tis_value > best_tis_value:
                best_tis_value = tis_value
                best_tis = tis

        self.choice = self.choice_by_ti[best_tis[0]]

        #Determine if anything has been updated. If so, increment the counter
        updated = False
        for ti in self.all_time_intervals:
            #The information in the 0th index of the comm packets is the information on this agent
            #as of last cycle which. Thus we compare the current prices to this data to measure change
            if not np.array_equal(self.prices[ti], self.price_comm_packets[ti][0,:]) or \
                not np.array_equal(self.high_bidders[ti], self.high_bidder_comm_packets[ti][0,:]):
                updated = True

        if not updated:
            self.steps_since_last_update += 1
        else:
            self.steps_since_last_update = 0

        self.n_iters += 1
        self.timestep_value_info_recieved[self.id] = self.n_iters

        self.converged = self.steps_since_last_update >= self.max_steps_since_last_update

    def update_prices_bids(self):
        """
        Updates the agent's prices and bids based on the price and high bidder communication packet.
        """
        for ti in self.all_time_intervals:
            max_prices = np.max(self.price_comm_packets[ti], axis=0)
            
            # Filter the high bidders by the ones that have the max price, and set the rest to -1.
            # Grab the highest index max bidder to break ties.
            max_price_bidders = np.where(self.price_comm_packets[ti] == max_prices, self.high_bidder_comm_packets[ti], -1)
            self.high_bidders[ti] = np.max(max_price_bidders, axis=0)

            if max_prices[self.choice_by_ti[ti]] >= self.prices[ti][self.choice_by_ti[ti]] and self.high_bidders[ti][self.choice_by_ti[ti]] != self.id:
                #Adjust the combined benefits to enforce handover penalty 
                if self.init_assignment is None:
                    #If initial assignment is None, then you shouldn't add a penalty at any index
                    indices_where_agent_not_prev_assigned = np.zeros(self.m)
                else:
                    indices_where_agent_not_prev_assigned = np.where(self.init_assignment[self.id,:]==1, 0, 1)
                benefit_hat = self.time_interval_benefits[ti] - self.lambda_ * indices_where_agent_not_prev_assigned

                best_net_value = np.max(benefit_hat - max_prices)
                second_best_net_value = np.partition(benefit_hat - max_prices, -2)[-2] #https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix

                self.choice_by_ti[ti] = np.argmax(benefit_hat-max_prices) #choose the task with the highest benefit to the agent

                self.high_bidders[ti][self.choice_by_ti[ti]] = self.id
                
                inc = best_net_value - second_best_net_value + self.eps

                self.prices[ti] = max_prices
                self.prices[ti][self.choice_by_ti[ti]] = max_prices[self.choice_by_ti[ti]] + inc
            else:
                #Otherwise, don't change anything and just update prices
                #based on the new info from other agents.
                self.prices[ti] = max_prices

    def update_time_interval_sequence_values(self):
        """
        Updates the values it has for all other time interval sequences for all other agents,
        in order to be able to choose the TIS which is optimal across the entire constellation.
        """
        agents_w_most_updated_value_info = np.argmax(self.timestep_value_info_recieved_comm_packet, axis=0)
        self.timestep_value_info_recieved = np.max(self.timestep_value_info_recieved_comm_packet, axis=0)

        for time_interval_sequence in self.all_time_interval_sequences:
            #This line takes the most updated benefit info from the neighbors and puts it into the agent's own benefit info
            self.values_from_time_interval_seq[time_interval_sequence] = self.value_from_time_interval_seq_comm_packets[time_interval_sequence][agents_w_most_updated_value_info, np.arange(self.n)]

        #Based on the new choices for each time interval, calculate the time interval sequence benefits for the agent itself
        for time_interval_sequence in self.all_time_interval_sequences:
            tis_value = 0
            if self.init_assignment is None:
                curr_assignment = None
            else: curr_assignment = np.argmax(self.init_assignment[self.id,:])
            
            for time_interval in time_interval_sequence:
                tis_value += self.time_interval_benefits[time_interval][self.choice_by_ti[time_interval]]
                
                if curr_assignment != self.choice_by_ti[time_interval] and curr_assignment is not None:
                    tis_value -= self.lambda_

                curr_assignment = self.choice_by_ti[time_interval]
            
            self.values_from_time_interval_seq[time_interval_sequence][self.id] = tis_value

def choose_time_interval_sequence_centralized(time_interval_sequences, prev_assignment, benefit_mat_window, lambda_, approx=False):
    """
    Chooses the best time interval sequence from a list of time interval sequences,
    and return the corresponding assignment.
    """
    n = benefit_mat_window.shape[0]
    m = benefit_mat_window.shape[1]

    best_value = -np.inf
    best_assignment = None
    best_time_interval = None

    for time_interval_sequence in time_interval_sequences:
        total_tis_value = 0
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

            #Calculate the value from this time interval sequence
            total_tis_value += (tis_assignment * combined_benefit_mat).sum()
            total_tis_value -= calc_assign_seq_handover_penalty(tis_assignment_curr, [tis_assignment], lambda_)

            tis_assignment_curr = tis_assignment
            #If this is the first assignment in the time interval sequence, save it
            if i == 0:
                tis_first_assignment = tis_assignment_curr

        #If this time interval sequence is better than the best one we've seen so far, save it 
        if total_tis_value > best_value:
            best_value = total_tis_value
            best_assignment = tis_first_assignment
            best_time_interval = time_interval_sequence[0]

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

def solve_w_haal(benefits, init_assignment, lambda_, L, graphs=None, distributed=False, central_approx=False, verbose=False, eps=0.01):
    """
    Sequentially solves the problem using the HAAL algorithm.

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
        if verbose: 
            if distributed: print(f"Solving w distributed HAAL, {len(chosen_assignments)}/{T}", end='\r')
            else: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
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
            if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Auction(benefit_mat_window, curr_assignment, all_time_intervals, all_time_interval_sequences, eps=eps, graph=graphs[curr_tstep], lambda_=lambda_)
            haal_d_auction.run_auction()

            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value, nh = calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_)
    
    return chosen_assignments, total_value, nh

def solve_w_haald_track_iters(benefits, init_assignment, lambda_, L, graphs=None, verbose=False, eps=0.01):
    """
    Sequentially solves the problem using the HAAL algorithm.

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

    total_iterations = 0

    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w distributed HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        global all_time_interval_sequences
        all_time_interval_sequences = []
        build_time_interval_sequences(all_time_intervals, [], len_window)

        haal_d_auction = HAAL_D_Auction(benefit_mat_window, curr_assignment, all_time_intervals, all_time_interval_sequences, eps=eps, graph=graphs[curr_tstep], lambda_=lambda_)
        haal_d_auction.run_auction()

        chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment

        total_iterations += haal_d_auction.n_iterations
    
    total_value, nh = calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_)
    
    return chosen_assignments, total_value, nh, total_iterations/T

if __name__ == "__main__":
    np.random.seed(42)
    benefits = 2*np.random.random((50, 50, 10))
    s = time.time()
    chosen_assignments, val, _ = solve_w_haal(benefits, 4, None, distributed=True, verbose=False, graphs=[rand_connected_graph(50) for _ in range(10)])
    print(val,time.time()-s)
    # print(calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, 1))