import numpy as np
from methods import rand_connected_graph, plot_graph, check_almost_equilibrium
import networkx as nx
import scipy.optimize

class Auction(object):
    def __init__(self, n_agents, n_tasks, eps=0.01, benefits=None, prices=None, graph=None, verbose=False):
        if benefits is not None:
            self.benefits = benefits
        else:
            self.benefits = np.random.rand(n_agents, n_tasks)

        if prices is not None:
            if prices.shape == (n_agents, n_tasks):
                self.prices = prices
            elif prices.shape == (n_tasks,):
                self.prices = np.tile(prices, (n_agents,1))
            else:
                print("Invalid prices shape")
                raise ValueError
            self.prices = self.prices.astype(float)
        else:
            self.prices = np.zeros((n_agents, n_tasks))

        if graph != None:
            self.graph = graph
        else:
            self.graph = rand_connected_graph(n_agents)

        self.verbose = verbose

        self.n_agents = n_agents
        self.n_tasks = n_tasks

        self.eps = eps

        #Initialize agents with ID, benefits and neighbors
        self.agents = [AuctionAgent(self, i, self.eps, self.benefits[i,:], self.prices[i,:], list(self.graph.neighbors(i))) for i in range(n_agents)]

        self.n_iterations = 0
        self.total_benefit_hist = []

    def show_graph(self):
        plot_graph(self.graph)

    def solve_centralized(self):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(self.benefits, maximize=True)
        if self.verbose:
            print("Centralized solution")
            for row, col in zip(row_ind, col_ind):
                print(f"Agent {row} chose task {col_ind[row]} with benefit {self.benefits[row, col]}")
            print(f"Total benefit: {self.benefits[row_ind, col_ind].sum()}")

        return row_ind, col_ind

    def calc_total_benefit(self):
        total_benefit = 0
        for agent in self.agents:
            total_benefit += agent.benefits[agent.choice]
        return total_benefit

    def run_auction(self):
        self.n_iterations = 0
        while sum([agent.agent_prices_stable for agent in self.agents]) < self.n_agents:
            for agent in self.agents:
                agent.update_agent_prices_bids()

            self.total_benefit_hist.append(self.calc_total_benefit())

            for agent in self.agents:
                agent.publish_agent_prices_bids()
            
            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            for agent in self.agents:
                print(f"Agent {agent.id} chose task {agent.choice} with benefit {agent.benefits[agent.choice]} and price {agent.public_prices[agent.choice]}")
            print(f"Total benefit: {sum([agent.benefits[agent.choice] for agent in self.agents])}")

    def run_reverse_auction_for_asymmetric(self):
        reverse_iterations = 0

        assigned_tasks = [ag.choice for ag in self.agents]
        unassigned_tasks = [j for j in range(self.n_tasks) if j not in assigned_tasks]

        #generate profits
        prices = self.agents[0].public_prices
        profits = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            j = self.agents[i].choice
            profits[i] = self.benefits[i,j] - prices[j]

        #Find minimum price for any assigned task
        lambda_ = np.inf
        for assigned_task in assigned_tasks:
            if prices[assigned_task] < lambda_:
                lambda_ = prices[assigned_task]

        #Remove unassigned tasks which cannot be undervalued
        #(i.e. their price is below the minimum price for any assigned task)
        potentially_undervalued_tasks = [uat for uat in unassigned_tasks if prices[uat] > lambda_]
        
        print(f"Initial potentially undervalued tasks: {potentially_undervalued_tasks}")
        print(f"Initial prices: {prices}")
        print(f"Initial profits: {profits}")

        while len(potentially_undervalued_tasks) > 0:
            reverse_iterations += 1
            puv_task = potentially_undervalued_tasks[0]
      
            #Find the agent which provides the most profit for the task
            best_agent = np.argmax(self.benefits[:,puv_task] - profits)
            
            #Find the profit provided by the two best agents for the task.

            #The amount the best agent would be willing to pay for the task,
            #given how much the agent is already making in profit from the other task.
            best_profit = np.max(self.benefits[:,puv_task] - profits) 
            second_best_profit = np.partition(self.benefits[:,puv_task] - profits, -2)[-2]

            #If the best agent turns out to not want to pay more than the minimum price,
            #then this task had a high value but was simply bad
            if lambda_ >= best_profit - self.eps:
                prices[puv_task] = lambda_
                potentially_undervalued_tasks.remove(puv_task)
                continue

            # Determine how much you're willing to lower the price.
            # The first term indicates that you don't want to lower the price below
            # lambda, or else we might think this task is no longer undervalued.
            # The second term indicates that you're only willing to lower the price enough
            # so that the second best agent would be willing to pay for the task.
            delta = min(best_profit - lambda_, best_profit-second_best_profit+self.eps)

            prices[puv_task] = best_profit - delta
            profits[best_agent] = profits[best_agent] + delta

            #If the unassigned task has a price above lambda, then it's still potentially undervalued.
            if prices[self.agents[best_agent].choice] > lambda_:
                potentially_undervalued_tasks.append(self.agents[best_agent].choice)

            # Update the agent's task
            self.agents[best_agent].choice = puv_task

            #Remove the task from the list of potentially undervalued tasks
            potentially_undervalued_tasks.remove(puv_task)

        if self.verbose:
            print(f"Reverse auction results ({reverse_iterations} iterations):")
            for agent in self.agents:
                print(f"Agent {agent.id} chose task {agent.choice} with benefit {agent.benefits[agent.choice]} and price {agent.public_prices[agent.choice]}")
            print(f"Total benefit: {sum([agent.benefits[agent.choice] for agent in self.agents])}")

class AuctionAgent(object):
    def __init__(self, auction, id, eps, benefits, prices, neighbors):
        self.auction = auction
        self.id = id
        self.benefits = benefits

        self.neighbors = neighbors
        self.choice = np.argmax(benefits-prices)
        print(f"Agent {self.id} chose task {self.choice} with benefit {self.benefits[self.choice]} and price {prices[self.choice]}")

        self.eps = eps

        self._prices = prices #private prices
        self.public_prices = prices

        self._high_bidders = -1*np.ones_like(benefits) #private bidders
        self.public_high_bidders = -1*np.ones_like(benefits)
        self._high_bidders[self.choice] = self.id
        self.public_high_bidders[self.choice] = self.id

        self.agent_prices_stable = False
    
    def __repr__(self):
        ret_str = f"Agent {self.id}, neighbors {self.neighbors}\n"
        ret_str += f"\tCurrent prices: {self.public_prices}"
        ret_str += f"\tCurrent high bidders: {self.public_high_bidders}"
        ret_str += f"\tCurrent choice: {self.choice}"
        return ret_str
    
    def update_agent_prices_bids(self):
        neighbor_prices = np.array(self.public_prices)
        highest_bidders = np.array(self.public_high_bidders)
        for n in self.neighbors:
            neighbor_prices = np.vstack((neighbor_prices, self.auction.agents[n].public_prices))
            highest_bidders = np.vstack((highest_bidders, self.auction.agents[n].public_high_bidders))

        max_prices = np.max(neighbor_prices, axis=0)
        
        # Filter the high bidders by the ones that have the max price, and set the rest to -1.
        # Grab the highest index max bidder to break ties.
        max_price_bidders = np.where(neighbor_prices == max_prices, highest_bidders, -1)
        self._high_bidders = np.max(max_price_bidders, axis=0)

        if max_prices[self.choice] >= self.public_prices[self.choice] and self._high_bidders[self.choice] != self.id:
            best_net_value = np.max(self.benefits - max_prices)
            second_best_net_value = np.partition(self.benefits - max_prices, -2)[-2] #https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix

            self.choice = np.argmax(self.benefits-max_prices) #choose the task with the highest benefit to the agent

            self._high_bidders[self.choice] = self.id
            
            inc = best_net_value - second_best_net_value + self.eps

            self._prices = max_prices
            self._prices[self.choice] = max_prices[self.choice] + inc
        else:
            #Otherwise, don't change anything and just update prices
            #based on the new info from other agents.
            self._prices = max_prices

    def publish_agent_prices_bids(self):
        #Determine if prices and bids have changed since the last iteration.
        if np.array_equal(self._prices, self.public_prices) and np.array_equal(self._high_bidders, self.public_high_bidders):
            self.agent_prices_stable = True
        else:
            self.agent_prices_stable = False

        self.public_prices = np.copy(self._prices)
        self.public_high_bidders = np.copy(self._high_bidders)
        

if __name__ == "__main__":
    #Benefit array which can show proof of suboptimality by epsilon
    # b = np.array([[0.58171674, 0.16394833, 0.9471958,  0.67933512, 0.75647097, 0.96396759],
    #                 [0.19934895, 0.89260454, 0.18748017, 0.96584496, 0.37879552, 0.20749475],
    #                 [0.07692341, 0.15046442, 0.34058061, 0.93558144, 0.785595,   0.30242082],
    #                 [0.53182682, 0.92819657, 0.79620561, 0.71194428, 0.8427648,  0.11332127]])
    # b = None
    b = np.array([[0.15, 0.05, 101, 100],
                  [0.2, 0.15, 100, 101]])
    
    p = np.array([0,0,1000,1000])

    a = Auction(2,4, benefits=b, prices=p, verbose=True)
    for ag in a.agents:
        ag.eps = 0.01
    eps = check_almost_equilibrium(a)
    print(f"eps eq before: {eps}")

    a.run_auction()
    eps = check_almost_equilibrium(a)
    print(f"eps eq after: {eps}")
    print("Final Prices:")
    print(a.agents[0].public_prices)

    a.run_reverse_auction_for_asymmetric()
    a.solve_centralized()

    # b = np.array([[101.0, 10, 1],
    #               [100, 10, 1]])
    
    # p = np.array([1000.0, 0, 0])

    # a = Auction(2,3, benefits=b, prices=p, verbose=True)

    # eps = check_almost_equilibrium(a)
    # print(f"eps eq before: {eps}")
    # a.run_auction()
    # eps = check_almost_equilibrium(a)
    # print(f"eps eq after: {eps}")

    # a.run_reverse_auction_for_asymmetric()
    # a.solve_centralized()
