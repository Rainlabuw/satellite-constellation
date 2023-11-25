import numpy as np

class CBBAAuction(object):
    def __init__(self, benefits, lambda_):
        self.lambda_ = lambda_

        # n x m x T array holding benefits for each agent and each task at each timestep
        self.benefits = benefits
        self.n = benefits.shape[0]
        self.m = benefits.shape[1]
        self.T = benefits.shape[2]

        self.prices = np.zeros_like(self.benefits)
        self.winning_agents = np.zeros_like(self.benefits)

        self.agents = [CBBAAgent(i, self.benefits[i,:,:], self.prices[i,:,:], self.winning_agents[i,:,:], self.lambda_) for i in range(self.n)]

    def run_auction(self):
        self.n_iterations = 0
        while sum([agent.converged for agent in self.agents]) < self.n:
            #Send the appropriate communication packets to each agent
            self.update_communication_packets()

            #Have each agent calculate it's prices, bids, and values
            #based on the communication packet it currently has
            for agent in self.agents:
                agent.update_agent_prices_bids()

            for agent in self.agents:
                agent.publish_agent_prices_bids()

            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")

class CBBAAgent(object):
    def __init__(self, id, benefits, prices, winning_agents, neighbors, lambda_):
        self.id = id

        # m x T array holding the benefits/prices of each task at each timestep
        self.benefits = benefits

        self._prices = prices
        self._winning_agents = winning_agents
        self.public_prices = prices
        self.public_winning_agents = winning_agents

        self.m = benefits.shape[0]
        self.T = benefits.shape[1]

        self.bundle_task_path = [None]*self.T
        self.bundle_tasks = []

        self.lambda_ = lambda_

        self.neighbors = neighbors

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
                        task_price = self._prices[task_idx, timestep]
                        marginal_benefit = self.score_task_based_on_bundle(task_idx, timestep, raw_benefit, task_price, self.lambda_)
                        if marginal_benefit > most_marginal_benefit:
                            most_marginal_benefit = marginal_benefit
                            most_marginal_benefit_task_idx = task_idx
                            most_marginal_benefit_timestep = timestep

            self.bundle_task_path[most_marginal_benefit_timestep] = most_marginal_benefit_task_idx
            self.bundle_tasks.append((most_marginal_benefit_task_idx, most_marginal_benefit_timestep))

            self._prices[most_marginal_benefit_task_idx, most_marginal_benefit_timestep] = most_marginal_benefit
            self._winning_agents[most_marginal_benefit_task_idx, most_marginal_benefit_timestep] = self.id

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
    
    def publish_agent_prices_bids(self):
        #Determine if prices and bids have changed since the last iteration.
        if np.array_equal(self._prices, self.public_prices) and np.array_equal(self._high_bidders, self.public_high_bidders):
            self.agent_prices_stable = True
        else:
            self.agent_prices_stable = False

        self.public_prices = np.copy(self._prices)
        self.public_high_bidders = np.copy(self._high_bidders)

if __name__ == "__main__":
    b = np.array([[10, 1],[1, 1.5]])
    p = np.zeros_like(b)
    w = np.zeros_like(b)
    a = CBBAAgent(1, b, p, w, None, 1)
    a.build_bundle()

    print(a.bundle_task_path)
    print(a.bundle_tasks)