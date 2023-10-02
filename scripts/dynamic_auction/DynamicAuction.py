import numpy as np
from methods import *
import scipy.optimize
from AuctionAgent import AuctionAgent
from AuctionTask import AuctionTask

class DynamicAuction(object):
    def __init__(self, n_agents, n_tasks, benefits=None, graph=None, verbose=False, seed=None):
        if seed != None:
            np.random.seed(seed)
        else:
            np.random.seed(np.random.randint(0,100000))

        if benefits is not None:
            self.benefits = benefits
        else:
            self.benefits = np.random.rand(n_agents, n_tasks)

        if graph != None:
            self.graph = graph
        else:
            self.graph = rand_connected_graph(n_agents)

        self.verbose = verbose

        self.n_agents = n_agents
        self.n_tasks = n_tasks

        #Initialize agents with ID, benefits and neighbors
        self._initialize_agents()

        self.n_iterations = 0
        self.total_benefit_hist = []

    def _initialize_agents(self):
        self.agents = []
        for i in range(self.n_agents):
            agent_tasks = []
            for j in range(self.n_tasks):
                agent_tasks.append(AuctionTask(self.benefits[i,j]))
            
            self.agents.append(AuctionAgent(self, i, agent_tasks, list(self.graph.neighbors(i))))

    def seed_agents_with_centralized_solution(self):
        agent_inds, choices = self.solve_centralized()
        for agent_ind, choice in zip(agent_inds, choices):
            self.agents[agent_ind].choice = choice

    def show_graph(self):
        plot_graph(self.graph)

    def solve_true_problem_central(self):
        """
        Solve the task allocation problem with the true benefits (i.e. including agent cost multipliers)
        """
        adjusted_benefits = np.zeros_like(self.benefits)
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                adjusted_benefits[i,j] = self.benefits[i,j]*self.agents[i].tasks[j].agent_cost_mult


        row_ind, col_ind = scipy.optimize.linear_sum_assignment(adjusted_benefits, maximize=True)
        if self.verbose:
            print("Centralized solution")
            for row, col in zip(row_ind, col_ind):
                print(f"Agent {row} chose task {col} with benefit {self.benefits[row, col]}*{self.agents[row].tasks[col].agent_cost_mult}={adjusted_benefits[row][col]}")
            print(f"Total benefit: {adjusted_benefits[row_ind, col_ind].sum()}")

        return row_ind, col_ind
    
    def solve_centralized(self):
        """
        Solve the task allocation problem with only centralized knowledge (i.e. no agent cost multipliers)
        """
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(self.benefits, maximize=True)
        if self.verbose:
            print("Centralized solution")
            for row, col in zip(row_ind, col_ind):
                print(f"Agent {row} chose task {col} with benefit {self.benefits[row, col]}")
            print(f"Total benefit: {self.benefits[row_ind, col_ind].sum()}")

        return row_ind, col_ind

    def calc_total_benefit(self):
        total_benefit = 0
        for agent in self.agents:
            total_benefit += agent.net_benefits[agent.choice]
        return total_benefit

    def run_auction(self):
        self.n_iterations = 0
        while sum([agent.agent_tasks_stable for agent in self.agents]) < self.n_agents:
            for agent in self.agents:
                agent.update_agent_tasks_bids()

            self.total_benefit_hist.append(self.calc_total_benefit())

            for agent in self.agents:
                agent.publish_agent_info()
            
            self.n_iterations += 1

        if self.verbose:
            print(f"Auction results ({self.n_iterations} iterations):")
            for agent in self.agents:
                print(f"Agent {agent.id} chose task {agent.choice} with benefit {agent.net_benefits[agent.choice]} and price {agent.public_prices[agent.choice]}")
            print(f"Total benefit: {sum([agent.net_benefits[agent.choice] for agent in self.agents])}")

if __name__ == "__main__":
    #Benefit array which can show proof of suboptimality by epsilon
    # b = np.array([[0.58171674, 0.16394833, 0.9471958,  0.67933512, 0.75647097, 0.96396759],
    #                 [0.19934895, 0.89260454, 0.18748017, 0.96584496, 0.37879552, 0.20749475],
    #                 [0.07692341, 0.15046442, 0.34058061, 0.93558144, 0.785595,   0.30242082],
    #                 [0.53182682, 0.92819657, 0.79620561, 0.71194428, 0.8427648,  0.11332127]])
    # b = None
    b = np.random.rand(8,8)
    a = DynamicAuction(8,8, benefits=b, verbose=True, seed=42)
    a.run_auction()
    print("~~~~~~~~~~~~~~")
    a = DynamicAuction(8,8, benefits=b, verbose=True, seed=42)
    a.seed_agents_with_centralized_solution()
    a.run_auction()

    # a.run_auction()


    # a.solve_centralized()