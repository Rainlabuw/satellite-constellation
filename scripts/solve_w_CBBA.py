import numpy as np
import networkx as nx
from methods import *
import cProfile

class CBBAAuction(object):
    def __init__(self, benefits, lambda_, graph=None, verbose=False):
        self.verbose = verbose

        self.lambda_ = lambda_

        # n x m x T array holding benefits for each agent and each task at each timestep
        self.benefits = benefits
        self.n = benefits.shape[0]
        self.m = benefits.shape[1]
        self.T = benefits.shape[2]

        if graph is None:
            self.graph = nx.complete_graph(self.n)
        else:
            self.graph = graph

        steps_wout_release_until_convergence = nx.diameter(self.graph)

        self.prices = np.zeros_like(self.benefits)
        self.high_bidders = -1*np.ones_like(self.benefits, dtype=int)

        self.agents = [CBBAAgent(i, self.benefits[i,:,:], self.prices[i,:,:], self.high_bidders[i,:,:], np.zeros(self.n), 
                                 list(self.graph.neighbors(i)), self.lambda_, steps_wout_release_until_convergence) for i in range(self.n)]

    def run_auction(self):
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
        Compiles price, high bidder, and timestamp
        information for all of each agent's neighbors and stores it in an array.

        Then, update the agents variables accordingly so it can access that information
        during the auction.

        By precomputing these packets, we can parallelize the auction.
        """
        for agent in self.agents:
            agent.price_comm_packet = np.zeros((len(agent.neighbors)+1, self.m, self.T))
            agent.high_bidder_comm_packet = np.zeros((len(agent.neighbors)+1, self.m, self.T), dtype=int)
            agent.timestamp_comm_packet = np.zeros((len(agent.neighbors)+1,self.n), dtype=int)

            agent.price_comm_packet[0,:,:] = agent.prices
            agent.high_bidder_comm_packet[0,:,:] = agent.high_bidders
            agent.timestamp_comm_packet[0,:] = agent.timestamps
            for neighbor_num, neighbor_idx in enumerate(agent.neighbors):
                agent.price_comm_packet[neighbor_num+1,:,:] = self.agents[neighbor_idx].prices
                agent.high_bidder_comm_packet[neighbor_num+1,:,:] = self.agents[neighbor_idx].high_bidders
                agent.timestamp_comm_packet[neighbor_num+1,:] = self.agents[neighbor_idx].timestamps

class CBBAAgent(object):
    def __init__(self, id, benefits, prices, high_bidders, timestamps, neighbors, lambda_, steps_to_converge):
        self.id = id

        # m x T array holding the benefits/prices of each task at each timestep
        self.benefits = benefits

        # Attributes that are used for consensus
        self.prices = prices #m x T
        self.high_bidders = high_bidders # m x T
        self.timestamps = timestamps # n x 1

        #Communication packets (num_neigh + 1 x m x T) arrays
        self.price_comm_packet = None
        self.high_bidder_comm_packet = None
        self.timestamp_comm_packet = None #(num_neigh+1 x n)

        self.n = timestamps.shape[0]
        self.m = benefits.shape[0]
        self.T = benefits.shape[1]

        #Current assigned bundle of tasks
        #At the end of iteration, the bundle should contain the task index being completed at each timestep
        #in a list
        self.bundle_task_path = [None]*self.T
        #List of tuples of (task_idx, timestep) that are in the bundle, in order of addition to the bundle.
        self.bundle_tasks = []

        self.lambda_ = lambda_

        self.neighbors = neighbors

        #Convergence variables
        self.n_iters = 0
        self.n_iters_wout_releasing = 0
        self.n_iters_wout_release_to_converge = steps_to_converge
        self.converged = False

        self.choice = 0

    def build_bundle(self):
        #Iterate until you've assigned a task in each spot in the bundle
        while len(self.bundle_tasks) < self.T:
            most_marginal_benefit = -np.inf
            most_marginal_benefit_task_idx = None
            most_marginal_benefit_timestep = None
            for timestep, selected_task_idx in enumerate(self.bundle_task_path):
                #If selected_task_idx is None, then we can potentially add a task to the bundle
                if selected_task_idx is None:
                    for task_idx in range(self.m):
                        raw_benefit = self.benefits[task_idx, timestep]
                        task_price = self.prices[task_idx, timestep]
                        marginal_benefit = self.score_task_based_on_bundle(task_idx, timestep, raw_benefit, task_price, self.lambda_)
                        if marginal_benefit > most_marginal_benefit:
                            most_marginal_benefit = marginal_benefit
                            most_marginal_benefit_task_idx = task_idx
                            most_marginal_benefit_timestep = timestep

            self.bundle_task_path[most_marginal_benefit_timestep] = most_marginal_benefit_task_idx
            self.bundle_tasks.append((most_marginal_benefit_task_idx, most_marginal_benefit_timestep))

            self.prices[most_marginal_benefit_task_idx, most_marginal_benefit_timestep] = most_marginal_benefit + self.prices[most_marginal_benefit_task_idx, most_marginal_benefit_timestep]
            # if most_marginal_benefit_task_idx == 2 and most_marginal_benefit_timestep == 0: print("mmb", self.prices[most_marginal_benefit_task_idx, most_marginal_benefit_timestep])
            self.high_bidders[most_marginal_benefit_task_idx, most_marginal_benefit_timestep] = self.id

        #Select final choice to be the first task in the bundle path
        self.choice = self.bundle_task_path[0]

    def score_task_based_on_bundle(self, task_idx, task_timestep, benefit, price, lambda_):
        marginal_benefit = benefit - price
        #Calculate the score of the task based on the bundle
        if task_timestep != 0:
            if self.bundle_task_path[task_timestep-1] is None or self.bundle_task_path[task_timestep-1] == task_idx:
                pass #there is no penalty for switching tasks
            else:
                marginal_benefit -= lambda_

        if task_timestep != len(self.bundle_task_path) - 1:
            if self.bundle_task_path[task_timestep+1] is None or self.bundle_task_path[task_timestep+1] == task_idx:
                pass
            else:
                marginal_benefit -= lambda_
        return marginal_benefit

    def perform_auction_iteration_for_agent(self):
        #Apply the consensus rules described in CBBA paper
        self.apply_consensus_rules()

        #Release tasks that have had bids change during consensus process
        released_a_task = self.release_tasks()
        if not released_a_task:
            self.n_iters_wout_releasing += 1
        else:
            self.n_iters_wout_releasing = 0

        #After having removed tasks from the bundle, rebuild the bundle
        self.build_bundle()

        self.n_iters += 1

        if self.n_iters_wout_releasing > self.n_iters_wout_release_to_converge:
            self.converged = True

    def release_tasks(self):
        """
        If any tasks have had their prices change during the consensus process,
        then release them and all tasks added to the bundle after them.
        """
        released_a_task = False
        for j in range(self.m):
            for k in range(self.T):
                if (j,k) in self.bundle_tasks:
                    #information at index 0 in communication packet is information from
                    #this agent at the previous timestep.
                    #If the price for this task at this timestep has changed, then remove it and
                    #all tasks added to the bundle after it
                    if self.prices[j,k] != self.price_comm_packet[0,j,k]:
                        released_a_task = True
                        removal_index = self.bundle_tasks.index((j,k))
                        removed_tasks = self.bundle_tasks[removal_index:]

                        self.bundle_tasks = self.bundle_tasks[:removal_index]
                        for removed_task_num, (removed_task_idx, removed_task_timestep) in enumerate(removed_tasks):
                            self.bundle_task_path[removed_task_timestep] = None
                            if removed_task_num > 0:
                                self.prices[removed_task_idx, removed_task_timestep] = 0
                                self.high_bidders[removed_task_idx, removed_task_timestep] = -1
                                
        return released_a_task

    def apply_consensus_rules(self):
        """
        Apply consensus rules from paper - implementation largely
        adopted from https://github.com/zehuilu/CBBA-Python
        """
        eps = 1e-6
        self.timestamps[self.id] = self.n_iters + 1

        #Apply the consensus rules described in CBBA paper
        for neighbor_idx in range(len(self.neighbors)):
            neigh_comm_idx = neighbor_idx + 1
            neigh_id = self.neighbors[neighbor_idx]

            # Implement table for each task at each timestep
            for j in range(self.m):
                for k in range(self.T):
                    # Entries 1 to 4: Sender thinks he has the task
                    if self.high_bidder_comm_packet[neigh_comm_idx, j, k] == neigh_id:
                        
                        # Entry 1: Update or Leave
                        if self.high_bidders[j,k] == self.id:
                            if (self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) > eps:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                            elif abs(self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) <= eps:  # Equal scores
                                if self.high_bidders[j,k] > self.high_bidder_comm_packet[neigh_comm_idx,j,k]:  # Tie-break based on smaller index
                                    self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                    self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 2: Update
                        elif self.high_bidders[j,k] == neigh_id:
                            self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                            self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                
                        # Entry 3: Update or Leave
                        elif self.high_bidders[j,k] > -1:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidders[j,k]] > self.timestamps[self.high_bidders[j,k]]:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                            elif (self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) > eps:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                            elif abs(self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) <= eps:  # Equal scores
                                if self.high_bidders[j,k] > self.high_bidder_comm_packet[neigh_comm_idx,j,k]:  # Tie-break based on smaller index
                                    self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                    self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 4: Update
                        elif self.high_bidders[j,k] == -1:
                            self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                            self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        else:
                            print(self.high_bidders[j,k])
                            raise Exception("Unknown winner value: please revise!")

                    # Entries 5 to 8: Sender thinks receiver has the task
                    elif self.high_bidder_comm_packet[neigh_comm_idx,j,k] == self.id:

                        # Entry 5: Leave
                        if self.high_bidders[j,k] == self.id:
                            # Do nothing
                            pass
                            
                        # Entry 6: Reset
                        elif self.high_bidders[j,k] == neigh_id:
                            self.high_bidders[j,k] = -1
                            self.prices[j,k] = -1

                        # Entry 7: Reset or Leave
                        elif self.high_bidders[j,k] > -1:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidders[j,k]] > self.timestamps[self.high_bidders[j,k]]:  # Reset
                                self.high_bidders[j,k] = -1
                                self.prices[j,k] = -1
                            
                        # Entry 8: Leave
                        elif self.high_bidders[j,k] == -1:
                            # Do nothing
                            pass

                        else:
                            print(self.high_bidders[j,k])
                            raise Exception("Unknown winner value: please revise!")

                    # Entries 9 to 13: Sender thinks someone else has the task
                    elif self.high_bidder_comm_packet[neigh_comm_idx,j,k] > -1:
                        
                        # Entry 9: Update or Leave
                        if self.high_bidders[j,k] == self.id:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] > self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:
                                if (self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) > eps:
                                    self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]  # Update
                                    self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                                elif abs(self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) <= eps:  # Equal scores
                                    if self.high_bidders[j,k] > self.high_bidder_comm_packet[neigh_comm_idx,j,k]:  # Tie-break based on smaller index
                                        self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                        self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 10: Update or Reset
                        elif self.high_bidders[j,k] == neigh_id:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] > self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                            else:  # Reset
                                self.high_bidders[j,k] = -1
                                self.prices[j,k] = -1

                        # Entry 11: Update or Leave
                        elif self.high_bidders[j,k] == self.high_bidder_comm_packet[neigh_comm_idx,j,k]:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] > self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 12: Update, Reset or Leave
                        elif self.high_bidders[j,k] > -1:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidders[j,k]] > self.timestamps[self.high_bidders[j,k]]:
                                if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] >= self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:  # Update
                                    self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                    self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                                elif self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] < self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:  # Reset
                                    self.high_bidders[j,k] = -1
                                    self.prices[j,k] = -1
                                else:
                                    raise Exception("Unknown condition for Entry 12: please revise!")
                            else:
                                if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] > self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:
                                    if (self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) > eps:  # Update
                                        self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                        self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]
                                    elif abs(self.price_comm_packet[neigh_comm_idx,j,k] - self.prices[j,k]) <= eps:  # Equal scores
                                        if self.high_bidders[j,k] > self.high_bidder_comm_packet[neigh_comm_idx,j,k]:  # Tie-break based on smaller index
                                            self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                            self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 13: Update or Leave
                        elif self.high_bidders[j,k] == -1:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidder_comm_packet[neigh_comm_idx,j,k]] > self.timestamps[self.high_bidder_comm_packet[neigh_comm_idx,j,k]]:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        else:
                            raise Exception("Unknown winner value: please revise!")

                    # Entries 14 to 17: Sender thinks no one has the task
                    elif self.high_bidder_comm_packet[neigh_comm_idx,j,k] == -1:

                        # Entry 14: Leave
                        if self.high_bidders[j,k] == self.id:
                            # Do nothing
                            pass

                        # Entry 15: Update
                        elif self.high_bidders[j,k] == neigh_id:
                            self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                            self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 16: Update or Leave
                        elif self.high_bidders[j,k] > -1:
                            if self.timestamp_comm_packet[neigh_comm_idx,self.high_bidders[j,k]] > self.timestamps[self.high_bidders[j,k]]:  # Update
                                self.high_bidders[j,k] = self.high_bidder_comm_packet[neigh_comm_idx,j,k]
                                self.prices[j,k] = self.price_comm_packet[neigh_comm_idx,j,k]

                        # Entry 17: Leave
                        elif self.high_bidders[j,k] == -1:
                            # Do nothing
                            pass
                        else:
                            raise Exception("Unknown winner value: please revise!")

                        # End of table
                    else:
                        raise Exception("Unknown winner value: please revise!")

            # Update timestamps for all agents based on latest comm
            for n in range(self.n):
                if (n != self.id) and (self.timestamps[n] < self.timestamp_comm_packet[neigh_comm_idx,n]):
                    self.timestamps[n] = self.timestamp_comm_packet[neigh_comm_idx,n]
            self.timestamps[neigh_id] = self.n_iters

def solve_w_CBBA(unscaled_benefits, init_assignment, lambda_, L, graphs=None, verbose=False):
    n = unscaled_benefits.shape[0]
    m = unscaled_benefits.shape[1]
    T = unscaled_benefits.shape[2]

    min_benefit = np.min(unscaled_benefits)
    benefit_to_add = max(2*lambda_-min_benefit, 0)
    benefits = unscaled_benefits + benefit_to_add

    if graphs is None:
        graphs = [nx.complete_graph(n) for i in range(T)]

    curr_assignment = init_assignment
    
    chosen_assignments = []

    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w distributed CBBA, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        len_window = benefit_mat_window.shape[-1]

        if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
        cbba_auction = CBBAAuction(benefit_mat_window, lambda_, graph=graphs[curr_tstep], verbose=verbose)
        cbba_auction.run_auction()


        chosen_assignment = convert_agents_to_assignment_matrix(cbba_auction.agents)

        chosen_assignments.append(chosen_assignment)
    
    total_value, nh = calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_)
    real_value = total_value - benefit_to_add*n*T

    return chosen_assignments, real_value, nh

def solve_w_CBBA_track_iters(unscaled_benefits, init_assignment, lambda_, L, graphs=None, verbose=False):
    n = unscaled_benefits.shape[0]
    m = unscaled_benefits.shape[1]
    T = unscaled_benefits.shape[2]

    min_benefit = np.min(unscaled_benefits)
    benefit_to_add = max(2*lambda_-min_benefit, 0)
    benefits = unscaled_benefits + benefit_to_add

    if graphs is None:
        graphs = [nx.complete_graph(n) for i in range(T)]

    curr_assignment = init_assignment
    
    total_iterations = 0

    chosen_assignments = []

    while len(chosen_assignments) < T:
        if verbose: print(f"Solving w distributed CBBA, {len(chosen_assignments)}/{T}", end='\r')
        curr_tstep = len(chosen_assignments)
        tstep_end = min(curr_tstep+L, T)
        benefit_mat_window = benefits[:,:,curr_tstep:tstep_end]

        if not nx.is_connected(graphs[curr_tstep]): print("WARNING: GRAPH NOT CONNECTED")
        cbba_auction = CBBAAuction(benefit_mat_window, lambda_, graph=graphs[curr_tstep])
        cbba_auction.run_auction()


        chosen_assignment = convert_agents_to_assignment_matrix(cbba_auction.agents)

        chosen_assignments.append(chosen_assignment)
        total_iterations += cbba_auction.n_iterations
    
    total_value, nh = calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_)
    real_value = total_value - benefit_to_add*n*T

    return chosen_assignments, real_value, nh, total_iterations/T

if __name__ == "__main__":
    benefits = np.random.rand(100,100,10)
    init_assignment = None
    lambda_ = 0.5
    L = benefits.shape[-1]
    graphs = [rand_connected_graph(100) for i in range(benefits.shape[-1])]

    cProfile.run('solve_w_CBBA(benefits, init_assignment, lambda_, 1,graphs=graphs,verbose=True)')