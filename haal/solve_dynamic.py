from typing import Any
import numpy as np
from common.methods import *
import networkx as nx
import time

def solve_w_dynamic_haal(benefits, init_assignment, lambda_, L, graphs=None, distributed=False, central_approx=False, verbose=False, eps=0.01):
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

        if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
        haal_d_auction = HAAL_D_Auction(benefit_mat_window, curr_assignment, all_time_intervals, all_time_interval_sequences, eps=eps, graph=graphs[curr_tstep], lambda_=lambda_)
        haal_d_auction.run_auction()

        chosen_assignment = convert_agents_to_assignment_matrix(haal_d_auction.agents)

        chosen_assignments.append(chosen_assignment)
        curr_assignment = chosen_assignment
    
    total_value, nh = calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_)
    
    return chosen_assignments, total_value, nh

def build_time_interval_sequences(all_time_intervals, time_interval_sequence, len_window):
    """
    Recursively constructs all possible time interval sequences from the set of all time intervals.

    Implements the logic behind BUILD_TIME_INTERVAL_SEQUENCES from the paper.
    """
    global all_time_interval_sequences
    #Grab the most recent timestep from the end of the current sol
    if time_interval_sequence == []:
        most_recent_timestep = -1 #set it to -1 so that time intervals starting w 0 will be selected
    else:
        most_recent_timestep = time_interval_sequence[-1][-1]

    #When we have an time interval seq which ends at the last timestep, we're done
    #and can add it to the list
    if most_recent_timestep == (len_window-1):
        all_time_interval_sequences.append(tuple(time_interval_sequence))
    else:
        #Iterate through all of the time intervals, looking for ones that start where this one ended
        for time_interval in all_time_intervals:
            if most_recent_timestep == time_interval[0]-1:
                build_time_interval_sequences(all_time_intervals, time_interval_sequence + [time_interval], len_window)

def generate_all_time_intervals(L):
    """
    Generates all possible time intervals from the next few timesteps.

    Implements GENERATE_ALL_TIME_INTERVALS from the paper.
    """
    all_time_intervals = []
    for i in range(L):
        for j in range(i,L):
            all_time_intervals.append((i,j))
        
    return all_time_intervals

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

        print(f"After sum auction {self.n_iterations}")

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

    def update_task_comm_packets(self):
        for agent in self.agents:
            task_packets = {}
            for ti in agent.all_time_intervals:
                task_packet = []

                for neighbor_idx in agent.neighbors:
                    task_packet.append(self.agents[neighbor_idx].tasks)

                task_packets[ti] = task_packet

            agent.task_comm_packets = task_packets

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


class HAAL_D_Agent(object):
    def __init__(self, auction, id, all_time_intervals, all_time_interval_sequences, tasks, neighbors):
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
        self.tasks = tasks
        self.tasks = [self.calc_task_personal_benefit(task) for task in self.tasks]
        self.build_benefit_matrices_from_tasks()

        self.m = self.benefits.shape[0]
        self.T = self.benefits.shape[1]

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

        #task comm packet is a dictionary of lists of DynamicTask objects,
        self.task_comm_packets = None

        #price and high bidder packets are dictionaries of (num neighbor x m) matrices,
        #one for each different time interval
        self.price_comm_packets = None
        self.high_bidder_comm_packets = None

        #Value packet is a dictionary of (num neighbors x n) matrices,
        #one for each different time interval sequence
        self.value_comm_packets = None

        #~~~~~~~~~~~~~~Convergence related attributes~~~~~~~~~~~~~~~~
        self.steps_since_last_update = 0
        self.prices_bids_converged = False
        self.tis_values_converged = False

        self.n_iters = 0

        self.auction_params_updated = False

        #Final task choice selected by the algorithm
        self.choice = None

    # def calc_task_personal_benefit(self, task):
    #     """
    #     Given a task id (lat lon) and base benefit,
    #     determines the benefit for the specific agent.

    #     Can be overriden with a specific benefit calculation algorithm.
    #     """
    #     #normally, this would be a function of latitude and longitude
    #     task.personal_benefit = task.base_benefit + np.random.uniform(-0.1, 0.1)

    #     return task
        
    def price_highbidder_packets_from_task_packet(self, tasks_comm_packet):
        """
        Given the information on each of the tasks that each neighbor is aware of,
        constructs a list of all the active tasks across the knowledge of all neighbors,
        and generates price and high bidder matrices for each task, on each time interval.

        INPUT:
            tasks_comm_packet: list of lists of DynamicTask objects, one for each neighbor.
                Each list contains the tasks that the neighbor is aware of.
        OUTPUT:
            price_comm_packets: dictionary with keys of time intervals, and values of
                (num neighbor x m) matrices, where m is the number of tasks that are active
                across all neighbors.
            high_bidder_comm_packets: analagous dictionary, just with high bidder information
        """
        #Determine the list of all tasks that are active across all neighbors
        existing_task_locs = [task.id for task in self.tasks]

        for neigh_task_list in tasks_comm_packet:
            for neigh_task in neigh_task_list:
                if neigh_task.id not in existing_task_locs:
                    existing_task_locs.append(neigh_task.id)
                    self.auction_params_updated = True

                    #Add a version of this task to the agent's list of tasks
                    self.tasks.append(DynamicTask(id, neigh_task.price_by_ti, neigh_task.high_bidders_by_ti, 
                                                  neigh_task.benefits.shape[0]))
                    
                    #Update the benefit matrices to include this new task
                    self.build_benefit_matrices_from_tasks()
                    

        self.price_comm_packets = {}
        self.high_bidder_comm_packets = {}

        for ti in self.all_time_intervals:
            price_comm_packet_for_this_ti = np.zeros((len(self.neighbors)+1, len(existing_task_locs)), dtype=np.float16)
            high_bidder_comm_packet_for_this_ti = -1*np.ones((len(self.neighbors)+1, len(existing_task_locs)), dtype=np.int16)

            for neigh_idx, neigh_task_list in enumerate(tasks_comm_packet):
                for task in neigh_task_list:
                    task_idx = existing_task_locs.index(task.id)

                    price_comm_packet_for_this_ti[neigh_idx, task_idx] = task.price_by_ti[ti]
                    high_bidder_comm_packet_for_this_ti[neigh_idx, task_idx] = task.high_bidders[ti]

            self.price_comm_packets[ti] = price_comm_packet_for_this_ti
            self.high_bidder_comm_packets[ti] = high_bidder_comm_packet_for_this_ti


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
        of the auctions for the agents.
        """
        self.auction_params_updated = False
        #Convert a packet information about tasks to price and high bidder comm packet format
        self.price_highbidder_packets_from_task_packet()

        #Based on neighbor
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
                if self.init_assignment is None:
                    curr_assignment = None
                else: curr_assignment = np.argmax(self.init_assignment[self.id,:])
                
                for time_interval in time_interval_sequence:
                    tis_value += self.time_interval_benefits[time_interval][self.choice_by_ti[time_interval]]
                    
                    if curr_assignment != self.choice_by_ti[time_interval] and curr_assignment is not None:
                        tis_value -= self.lambda_

                    curr_assignment = self.choice_by_ti[time_interval]
                
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

            outbid = max_prices[self.choice_by_ti[ti]] >= self.prices[ti][self.choice_by_ti[ti]] and self.high_bidders[ti][self.choice_by_ti[ti]] != self.id

            if outbid or self.auction_params_updated:
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

            #Update tasks with new price and high bidder information
            for task_idx, task in enumerate(self.tasks):
                task.price_by_ti[ti] = self.prices[ti][task_idx]
                task.high_bidders_by_ti[ti] = self.high_bidders[ti][task_idx]

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

class DynamicTask(object):
    def __init__(self, id, price_by_ti, high_bidders_by_ti, time_left):
        self.id = id
        self.benefits = np.random.rand(time_left)
        
        #Dictionaries of high bidders and prices, by time interval
        self.price_by_ti = price_by_ti
        self.high_bidders_by_ti = high_bidders_by_ti

        #for removing tasks
        self.time = None
        self.deleted = False

if __name__ == "__main__":
    np.random.seed(48)
    benefits = 2*np.random.random((50, 50, 10))
    s = time.time()
    chosen_assignments, val, _ = solve_w_haal(benefits, None, 0.5, 4, distributed=True, verbose=False, graphs=[rand_connected_graph(50) for _ in range(10)])
    print(val,time.time()-s)
    # print(calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, 1))