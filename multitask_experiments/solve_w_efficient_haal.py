import numpy as np

from common.methods import *
from algorithms.solve_w_haal import choose_time_interval_sequence_centralized, HAAL_D_Parallel_Agent

def solve_w_efficient_haal(env, L, distributed=False, parallel=None, verbose=False, 
                     eps=0.01, graphs=None, track_iters=False):
    """
    Sequentially solves the problem given by the environment using the HAAL algorithm.

    When parallel_appox = True, computes the solution centrally, but by constraining each assignment to the current assignment,
        as is done to parallelize auctions in the distributed version of the algorithm.
    """
    if parallel is None: parallel = distributed #If no option is selected, parallel is off for centralized, but on for distributed
    if distributed and not parallel: print("Note: No serialized version of HAAL-D implemented yet. Solving parallelized.")
    
    n = env.sat_prox_mat.shape[0]
    m = env.sat_prox_mat.shape[1]
    T = env.sat_prox_mat.shape[2]

    if env.graphs is None and distributed:
        env.graphs = [nx.complete_graph(n) for i in range(T)]
    
    total_iterations = 0 if distributed else None
    total_value = 0
    chosen_assignments = []
    done = False
    while not done:
        if verbose: print(f"Solving w HAAL, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = env.k
        tstep_end = min(curr_tstep+L, T)
        prox_mat_window = env.sat_prox_mat[:,:,curr_tstep:tstep_end]

        len_window = prox_mat_window.shape[-1]

        all_time_intervals = generate_all_time_intervals(len_window)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, len_window)

        if distributed:
            if not nx.is_connected(env.graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
            haal_d_auction = HAAL_D_Efficient_Auction(prox_mat_window, env.curr_assignment, all_time_intervals, all_time_interval_sequences, 
                                                     eps=eps, graph=env.graphs[curr_tstep], lambda_=env.lambda_, 
                                                     benefit_fn=env.benefit_fn, benefit_info=env.benefit_info)
            haal_d_auction.run_auction()
            chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

            total_iterations += haal_d_auction.n_iterations
        else:
            chosen_assignment = choose_time_interval_sequence_centralized(all_time_interval_sequences, env.curr_assignment, prox_mat_window, 
                                                                      env.lambda_, env.benefit_fn, parallel_approx=parallel,
                                                                      benefit_info=env.benefit_info)

        chosen_assignments.append(chosen_assignment)
        
        _, value, done = env.step(chosen_assignment)
        total_value += value
    
    if not track_iters or not distributed:
        return chosen_assignments, total_value
    else:
        return chosen_assignments, total_value, total_iterations/T
    
class HAAL_D_Efficient_Auction(object):
    """
    This class runs an assignment auction for all possible time interval sequences
    in a single timestep, all in parallel.
    
    The algorithm stores the assigned task for each agent in their .choice attribute.
    """
    def __init__(self, sat_proximities, curr_assignment, all_time_intervals, all_time_interval_sequences, benefit_fn,
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

            #Have agents within each satellite coordinate on prices
            #TODO

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