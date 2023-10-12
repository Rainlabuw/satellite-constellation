import numpy as np
from methods import rand_connected_graph, plot_graph
import networkx as nx
import scipy.optimize

class Auction(object):
    def __init__(self, n_agents, n_tasks, benefits=None, prices=None, graph=None, verbose=False):
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

        #Initialize agents with ID, benefits and neighbors
        self.agents = [AuctionAgent(self, i, self.benefits[i,:], self.prices[i,:], list(self.graph.neighbors(i))) for i in range(n_agents)]

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

class AuctionAgent(object):
    def __init__(self, auction, id, benefits, prices, neighbors):
        self.auction = auction
        self.id = id
        self.benefits = benefits

        self.neighbors = neighbors
        self.choice = np.argmax(benefits-prices)
        print(f"Agent {self.id} chose task {self.choice} with benefit {self.benefits[self.choice]} and price {prices[self.choice]}")

        self.eps = 0.01

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

        print(f"Agent {self.id}")
        print(self._high_bidders)
        print(self._prices)

    def publish_agent_prices_bids(self):
        #Determine if prices and bids have changed since the last iteration.
        if np.array_equal(self._prices, self.public_prices) and np.array_equal(self._high_bidders, self.public_high_bidders):
            self.agent_prices_stable = True
        else:
            self.agent_prices_stable = False

        self.public_prices = np.copy(self._prices)
        self.public_high_bidders = np.copy(self._high_bidders)  

def checkAlmostEquilibrium(auction):
    max_eps = -np.inf
    for agent in auction.agents:
        max_net_value = -np.inf

        curr_ben = agent.benefits[agent.choice] - agent.public_prices[agent.choice]
        for j in range(auction.n_tasks):
            net_ben = agent.benefits[j] - agent.public_prices[j]

            if net_ben > max_net_value:
                max_net_value = net_ben
        
        eps = max_net_value - curr_ben
        if eps > max_eps:
            max_eps = eps

    return eps

if __name__ == "__main__":
    #Benefit array which can show proof of suboptimality by epsilon
    # b = np.array([[0.58171674, 0.16394833, 0.9471958,  0.67933512, 0.75647097, 0.96396759],
    #                 [0.19934895, 0.89260454, 0.18748017, 0.96584496, 0.37879552, 0.20749475],
    #                 [0.07692341, 0.15046442, 0.34058061, 0.93558144, 0.785595,   0.30242082],
    #                 [0.53182682, 0.92819657, 0.79620561, 0.71194428, 0.8427648,  0.11332127]])
    # b = None
    # b = np.array([[0.15, 0.05, 101, 100],
    #               [0.2, 0.15, 100, 101]])
    
    # p = np.array([0,0,1000,1000])

    # a = Auction(2,4, benefits=b, prices=p, verbose=True)
    # for ag in a.agents:
    #     ag.eps = 0.005
    # eps = checkAlmostEquilibrium(a)
    # print(f"eps eq before: {eps}")
    # # a.agents[0].choice = 2
    # # a.agents[1].choice = 3

    # # a.agents[0].public_high_bidders = np.array([-1,-1,0,-1])
    # # a.agents[1].public_high_bidders = np.array([-1,-1,-1,1])

    # a.run_auction()
    # eps = checkAlmostEquilibrium(a)
    # print(f"eps eq after: {eps}")
    # print(a.agents[0].public_prices)
    # print(a.agents[1].public_prices)
    # a.solve_centralized()

    b = np.array([[100.0, 10, 1],
                  [100, 10, 1]])
    
    p = np.array([1000.0, 0, 0])

    a = Auction(2,3, benefits=b, prices=p, verbose=True)
    # for ag in a.agents:
    #     ag.eps = 0.005
    eps = checkAlmostEquilibrium(a)
    print(f"eps eq before: {eps}")
    a.run_auction()
    eps = checkAlmostEquilibrium(a)
    print(f"eps eq after: {eps}")
