import numpy as np
from common.methods import *
import networkx as nx
import time

def solve_w_haal(sat_proximities, init_assignment, lambda_, L, distributed=False, parallel=None, verbose=False,
                 eps=0.01, graphs=None, track_iters=False, 
                 benefit_fn=generic_handover_pen_benefit_fn, benefit_info=None):
    """
    Sequentially solves the problem using the HAAL algorithm.

    When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")
    
    n = sat_proximities.shape[0]
    m = sat_proximities.shape[1]
    T = sat_proximities.shape[2]

    if graphs is None and distributed:
        graphs = [nx.complete_graph(n) for i in range(T)]

    curr_assignment = init_assignment
    
    total_iterations = 0 if distributed else None
    chosen_assignments = []
    ass_lens = []
    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        prox_mat_window = sat_proximities[:,:,curr_tstep:tstep_end]

        len_window = prox_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Parallel_Auction(prox_mat_window, curr_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=graphs[curr_tstep], lambda_=lambda_, 
                                                     benefit_fn=benefit_fn, benefit_info=benefit_info)
            haal_d_auction.run_auction()
            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, curr_assignment, prox_mat_window, 
                                                                      lambda_, parallel_approx=parallel, benefit_fn=benefit_fn,
                                                                      benefit_info=benefit_info)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value = calc_assign_seq_state_dependent_value(init_assignment, chosen_assignments, sat_proximities, lambda_, 
                                                        benefit_fn=benefit_fn, benefit_info=benefit_info)
    
    if not track_iters or not distributed:
        return chosen_assignments, total_value
    else:
        return chosen_assignments, total_value, total_iterations/T

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CENTRALIZED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def choose_time_interval_sequence_centralized(time_interval_sequences, prev_assignment, prox_mat_window, lambda_, parallel_approx=False, 
                                              benefit_fn=generic_handover_pen_benefit_fn, benefit_info=None):
    """
    Chooses the best time interval sequence from a list of time interval sequences,
    and return the corresponding assignment.
    """
    n = prox_mat_window.shape[0]
    m = prox_mat_window.shape[1]

    best_value = -np.inf
    best_assignment = None
    best_time_interval = None

    for time_interval_sequence in time_interval_sequences:
        total_tis_value = 0
        tis_assignment_curr = prev_assignment
        tis_first_assignment = None

        for i, time_interval in enumerate(time_interval_sequence):
            #Grab benefit matrices from this time interval
            ti_prox_mats = prox_mat_window[:,:,time_interval[0]:time_interval[1]+1]
            if ti_prox_mats.ndim == 2: #make sure ti_prox_mats is 3D
                ti_prox_mats = np.expand_dims(ti_prox_mats, axis=2) 

            #From proximity info, compute the benefit matrix using benefit_fn.
            #(By default, this incentivizes agents to be assigned the same task twice.
            # Note that if we're not approximating, we incentivize staying close to the previous assignment calculated during this
            # time interval sequence, not the actual assignment that the agents currently have (i.e. prev_assignment) )
            if not parallel_approx: 
                benefit_hat = benefit_fn(ti_prox_mats, tis_assignment_curr, lambda_, benefit_info)
            else: 
                benefit_hat = benefit_fn(ti_prox_mats, prev_assignment, lambda_, benefit_info)
            benefit_hat = benefit_hat.sum(axis=-1) #add up the benefits from the entire time window to assign with

            #Generate an assignment using a centralized solution.
            central_assignments = solve_centralized(benefit_hat)
            tis_assignment = convert_central_sol_to_assignment_mat(n, m, central_assignments)

            #Calculate the value from this time interval sequence
            total_tis_value += (benefit_hat*tis_assignment).sum()

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DISTRIBUTED FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class HAAL_D_Parallel_Auction(object):
    """
    This class runs an assignment auction for all possible time interval sequences
    in a single timestep, all in parallel.
    
    The algorithm stores the assigned task for each agent in their .choice attribute.
    """
    def __init__(self, sat_proximities, curr_assignment, all_time_intervals, all_time_interval_sequences, benefit_fn=generic_handover_pen_benefit_fn,
                 eps=0.01, graph=None, lambda_=1, verbose=False, benefit_info=None):
        # benefit matrix for the next L timesteps
        self.sat_proximities = sat_proximities
        self.n = sat_proximities.shape[0]
        self.m = sat_proximities.shape[1]
        self.T = sat_proximities.shape[2]

        #If no graph is provided, assume the communication graph is complete.
        if graph is None:
            self.graph = nx.complete_graph(self.n)
        else:
            self.graph = graph

        self.curr_assignment = curr_assignment
        
        self.chosen_assignments = []

        self.lambda_ = lambda_
        self.benefit_fn = benefit_fn
        self.benefit_info = benefit_info

        self.eps = eps
        self.verbose = verbose

        #The number of steps without an update before the algorithm converges
        self.max_steps_since_last_update = nx.diameter(self.graph)

        #Build the list of agents participating in the auction, providing them only
        #their proximities to tasks, benefit info, and a list of their neighbors.
        self.agents = [HAAL_D_Parallel_Agent(self, i, all_time_intervals, all_time_interval_sequences, \
                                    self.sat_proximities[i,:,:], list(self.graph.neighbors(i))) for i in range(self.n)]

    def run_auction(self):
        """
        Run the auction to completion, which has two phases:

        1. Agents communicate with their neighbors to determine the prices and high bidders
            Alternate between calculating updated prices and bids
            for each agent, and formulating and sending packets of information that each agent recieves
            from it's neighbors containing this updated information.
        2. Agents communicate with their neighbors to determine the value of each time interval sequence
        for each agent
            Continually update the value of each time interval sequence for each agent, and send that
            info to agent neighbors.
        """
        self.n_iterations = 0
        #Run communications until agents have converged on prices and bids
        while sum([agent.prices_bids_converged for agent in self.agents]) < self.n:
            #Send the appropriate communication packets to each agent
            self.update_price_bid_comm_packets()

            #Have each agent calculate it's prices, bids, and values
            #based on the communication packet it currently has
            for agent in self.agents:
                agent.perform_auction_iteration_for_agent()

            self.n_iterations += 1

        #Run value communication until each agent has a value for each time interval sequence
        while sum([agent.tis_values_converged for agent in self.agents]) < self.n:
            self.update_tis_value_comm_packets()

            for agent in self.agents:
                agent.update_tis_values()

            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")

    def update_price_bid_comm_packets(self):
        """
        Compiles price and high bidder information for all of each agent's 
        neighbors and stores it in an array.

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

                #Index zero in each packet corresponds to the information from the agent itself
                price_packet[0,:] = agent.prices[ti]
                high_bidder_packet[0,:] = agent.high_bidders[ti]
                
                for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                    price_packet[neighbor_num+1,:] = self.agents[neighbor_idx].prices[ti]
                    high_bidder_packet[neighbor_num+1,:] = self.agents[neighbor_idx].high_bidders[ti]

                price_packets[ti] = price_packet
                high_bidder_packets[ti] = high_bidder_packet

            agent.price_comm_packets = price_packets
            agent.high_bidder_comm_packets = high_bidder_packets

    def update_tis_value_comm_packets(self):
        """
        Compiles value information for all of each agent's
        neighbors and stores it in an array.

        Then, update the agents variables accordingly so it can access that information
        during the auction.

        This allows each agent to independently determine which time interval sequence
        is optimal across the entire constellation.
        """
        for agent in self.agents:
            value_packets = {}
            for tis in agent.all_time_interval_sequences:
                value_packet = np.zeros((len(agent.neighbors)+1,self.n), dtype=np.float16)

                #Index zero in each packet corresponds to the information from the agent itself
                value_packet[0,:] = agent.tis_values_by_agent[tis]
                
                for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                    value_packet[neighbor_num+1,:] = self.agents[neighbor_idx].tis_values_by_agent[tis]

                value_packets[tis] = value_packet

            agent.value_comm_packets = value_packets


class HAAL_D_Parallel_Agent(object):
    def __init__(self, auction, id, all_time_intervals, all_time_interval_sequences, proximities, neighbors):
        self.id = id

        #Grab info from auction
        self.init_assignment = auction.curr_assignment[self.id,:] if auction.curr_assignment is not None else None
        self.lambda_ = auction.lambda_
        self.n = auction.n
        self.eps = auction.eps
        self.max_steps_since_last_update = auction.max_steps_since_last_update
        self.benefit_fn = auction.benefit_fn
        #TODO: this benefit info is gonna need to change per individual satellite
        self.benefit_info = auction.benefit_info

        self.all_time_intervals = all_time_intervals
        self.all_time_interval_sequences = all_time_interval_sequences

        #Benefits and prices are mxL matrices.
        self.proximities = proximities

        self.m = proximities.shape[0]
        self.T = proximities.shape[1]

        #~~~~~~~Attributes which the agent uses to run the auction, and which it publishes to other agents~~~~~~~~
        self.high_bidders = {}
        self.prices = {}
        self.choice_by_ti = {} #the preferred choice of the agent in each time interval

        #Generate bids and prices for every time interval you're running an auction for
        for time_interval in all_time_intervals:
            #Initialize yourself as the highest bidder on the first task
            self.high_bidders[time_interval] = -1*np.ones(self.m, dtype=np.int16)
            self.high_bidders[time_interval][0] = self.id

            self.prices[time_interval] = np.zeros(self.m, dtype=np.float16)

            self.choice_by_ti[time_interval] = 0

        #The value of each time interval sequence for each agent,
        #used for value communication phase.
        self.tis_values_by_agent = {}
        for time_interval_sequence in all_time_interval_sequences:
            self.tis_values_by_agent[time_interval_sequence] = -np.inf*np.ones(self.n, dtype=np.float16)

        #~~~~~~~~Communication packet related attributes~~~~~~~~~~
        self.neighbors = neighbors

        #price and high bidder packets are (num neighbor x m) matrices,
        #one for each different time interval
        self.price_comm_packets = None
        self.high_bidder_comm_packets = None

        #Value packet is a (num neighbors x n) matrix,
        #one for each different time interval sequence
        self.value_comm_packets = None

        #~~~~~~~~~~~~~~Convergence related attributes~~~~~~~~~~~~~~~~
        self.steps_since_last_update = 0
        self.prices_bids_converged = False
        self.tis_values_converged = False

        self.n_iters = 0

        #Final task choice selected by the algorithm
        self.choice = None

    def get_total_benefits_for_ti(self, ti, prev_assignment):
        """
        Given a time interval, gets a single 2D benefit matrix which determines the benefit
        of a given assignment for the entire time interval.
        
        This includes state dependent penalties and benefits from all time steps in the interval.

        For a single agent, this manifests as returning a m-length vector of benefits for each task.
        """
        #Grab proximities for this time interval
        time_interval_proximities = np.copy(self.proximities[:,ti[0]:ti[1]+1])
        #Add dimension for number of agents so we can use our standard state dependent functions
        time_interval_proximities = np.expand_dims(time_interval_proximities, axis=0)
        if time_interval_proximities.ndim == 2: #make sure time_interval_proximities is 3D
            time_interval_proximities = np.expand_dims(time_interval_proximities, axis=2)

        benefit_hat = self.benefit_fn(time_interval_proximities, prev_assignment, self.lambda_, self.benefit_info)
        benefit_hat = np.squeeze(time_interval_proximities.sum(axis=-1))

        return benefit_hat

    def perform_auction_iteration_for_agent(self):
        """
        After recieving an updated communication packet, runs a single iteration
        of the auctions for the agents.
        """
        self.update_prices_bids()

        #Determine if anything has been updated. If so, increment the counter
        updated = False
        for ti in self.all_time_intervals:
            #The information in the 0th index of the comm packets is the information on this agent
            #as of last cycle. Thus we compare the current prices to this data to measure change
            if not np.array_equal(self.prices[ti], self.price_comm_packets[ti][0,:]) or \
                not np.array_equal(self.high_bidders[ti], self.high_bidder_comm_packets[ti][0,:]):
                updated = True

        if not updated:
            self.steps_since_last_update += 1
        else:
            self.steps_since_last_update = 0

        self.n_iters += 1

        self.prices_bids_converged = self.steps_since_last_update >= self.max_steps_since_last_update

        #If the agent has converged, then it should calculate the value of each time interval sequence
        #for communication to its neighbors
        if self.prices_bids_converged:
            for time_interval_sequence in self.all_time_interval_sequences:
                tis_value = 0
                curr_assignment = self.init_assignment

                for time_interval in time_interval_sequence:
                    benefit_hat = self.get_total_benefits_for_ti(time_interval, curr_assignment)

                    tis_value += benefit_hat[self.choice_by_ti[time_interval]]

                    curr_assignment = np.zeros(self.m)
                    curr_assignment[self.choice_by_ti[time_interval]] = 1
                
                self.tis_values_by_agent[time_interval_sequence][self.id] = tis_value

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
                benefit_hat = self.get_total_benefits_for_ti(ti, self.init_assignment)

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

    def update_tis_values(self):
        """
        Updates the agent's values based on the value communication packet.
        """
        value_sum = 0
        for tis in self.all_time_interval_sequences:
            max_values = np.max(self.value_comm_packets[tis], axis=0)

            self.tis_values_by_agent[tis] = max_values

            value_sum += sum(max_values)

        self.n_iters += 1

        #If value is not -np.inf, that means information
        #has been recieved from all neighbors, and the agent has converged
        if value_sum > -np.inf:
            self.tis_values_converged = True

            best_tis_value = -np.inf
            best_tis = None
            for tis in self.all_time_interval_sequences:
                if sum(self.tis_values_by_agent[tis]) > best_tis_value:
                    best_tis_value = sum(self.tis_values_by_agent[tis])
                    best_tis = tis
            
            #Select the choice associated with the best time interval sequence
            self.choice = self.choice_by_ti[best_tis[0]]
        else: self.tis_values_converged = False


if __name__ == "__main__":
    np.random.seed(48)
    benefits = 2*np.random.random((50, 50, 10))
    s = time.time()
    chosen_assignments, val, _ = solve_w_haal(benefits, None, 0.5, 4, distributed=True, verbose=False)
    print(val,time.time()-s)
    # print(calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, 1))