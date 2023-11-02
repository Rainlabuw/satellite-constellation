import numpy as np
from methods import *
import networkx as nx
import scipy.optimize
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

class Auction(object):
    def __init__(self, n_agents, n_tasks, eps=0.01, benefits=None, prices=None, graph=None, verbose=False):
        if benefits is not None:
            self.benefits = benefits
        else:
            zeros_ones = np.random.randint(2, size=(n_agents, n_tasks))
            bens = np.random.normal(1, 0.1, (n_agents, n_tasks))

            self.benefits = zeros_ones * bens

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
        total_benefit = self.benefits[row_ind, col_ind].sum()
        if self.verbose:
            print("Centralized solution:")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")
            print(f"\tTotal benefit: {total_benefit}")

        return row_ind, col_ind, total_benefit

    def calc_total_benefit(self):
        """
        Calculates the benefits of the current assignments.
        If multiple agents have chosen a single assignment, only take into account
        the higher benefit.
        """
        total_benefit = 0
        benefits_from_choices = {}
        choices_made = set([ag.choice for ag in self.agents])
        for choice_made in choices_made:
            benefits_from_choices[choice_made] = -np.inf
        
        for agent in self.agents:
            if agent.benefits[agent.choice] > benefits_from_choices[agent.choice]:
                benefits_from_choices[agent.choice] = agent.benefits[agent.choice]

        total_benefit = sum(benefits_from_choices.values())

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
            print(f"\tAssignments: {[a.choice for a in self.agents]}")
            print(f"\tTotal benefit: {sum([agent.benefits[agent.choice] for agent in self.agents])}")

        return self.calc_total_benefit()

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
            # The first term ensures that if we're lowering the prices a lot, we just lower
            # them to exactly lambda (for convenience?)
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

        total_benefit = sum([agent.benefits[agent.choice] for agent in self.agents])
        if self.verbose:
            print(f"Reverse auction results ({reverse_iterations} iterations):")
            print(f"\tAssignments: {[a.choice for a in self.agents]}")
            print(f"\tTotal benefit: {total_benefit}")
    
        return total_benefit

class AuctionAgent(object):
    def __init__(self, auction, id, eps, benefits, prices, neighbors):
        self.auction = auction
        self.id = id
        self.benefits = benefits

        self.neighbors = neighbors
        self.choice = np.argmax(benefits-prices)
        # print(f"Agent {self.id} chose task {self.choice} with benefit {self.benefits[self.choice]} and price {prices[self.choice]}")

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
    n = 10
    m = 10
    pert_scale = 0.05
    eps = 0.01

    graph = rand_connected_graph(n)
    auction = Auction(n, m, graph=graph, eps=0.01)
    auction.run_auction()
    prices_init = auction.agents[0].public_prices - np.min(auction.agents[0].public_prices)

    #Perturb the benefits
    perturb = np.random.normal(scale=pert_scale, size=(n, m))
    # perturb = np.random.choice([-0.05, 0, 0.05], size=(n, m))
    perturbed_benefits = auction.benefits + perturb
    max_perturb = np.max(np.abs(perturb))
    max_ben = np.max(perturbed_benefits)
    min_ben = np.min(perturbed_benefits)
    print(max_perturb)
    b_diff = max_ben - min_ben
    speedup = b_diff/(2*max_perturb+eps)
    print(speedup)
    # perturbed_benefits = np.clip(perturbed_benefits, 0, 1)

    #Run seeded auction
    seeded_auction = Auction(n, m, graph=graph, benefits=perturbed_benefits, prices=prices_init, eps=eps)
    for s_agent, agent in zip(seeded_auction.agents, auction.agents):
        s_agent.choice = agent.choice
    s_ae = check_almost_equilibrium(seeded_auction)
    seeded_benefit = seeded_auction.run_auction()

    unseeded_auction = Auction(n, m, graph=graph, benefits=perturbed_benefits, eps=eps)
    us_ae = check_almost_equilibrium(unseeded_auction)
    print(us_ae, s_ae)
    unseeded_benefit = unseeded_auction.run_auction()

    dist = calc_distance_btwn_solutions(seeded_auction.agents, auction.agents)

    print(seeded_auction.n_iterations, unseeded_auction.n_iterations, dist)

    plt.plot(seeded_auction.total_benefit_hist, label='Seeded')
    plt.plot(unseeded_auction.total_benefit_hist, label='Unseeded')
    plt.legend()
    plt.show()