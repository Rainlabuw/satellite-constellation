import numpy as np
from copy import deepcopy

class AuctionAgent(object):
    def __init__(self, auction, id, tasks, neighbors):
        self.auction = auction
        self.id = id

        self._tasks = tasks
        self.public_tasks = deepcopy(tasks)

        #Calculate true benefits based on agent cost multipliers
        # (i.e. having lower fuel than expected would imply a cost mult <1)
        for task in self._tasks:
            task.agent_cost_mult = self._calc_cost_mult_for_task(task)

        self.net_benefits = np.array([task.benefit*task.agent_cost_mult for task in tasks])

        self.neighbors = neighbors

        #make initial choice in a greedy way
        self.choice = np.argmax(self.net_benefits)

        self.eps = 0.01

        self.agent_tasks_stable = False

    def __repr__(self):
        ret_str = f"Agent {self.id}, neighbors {self.neighbors}\n"
        ret_str += f"\tCurrent prices: {self.public_prices}"
        ret_str += f"\tCurrent high bidders: {self.public_high_bidders}"
        ret_str += f"\tCurrent choice: {self.choice}"
        return ret_str
    
    def _calc_cost_mult_for_task(self, task):
        """
        Calculate the cost multiplier for a given task.

        Currently, this is just a random choice between 0.0 and 1.0,
        but could overload this method with an actual way of computing
        a cost benefit, perhaps from a fuel model, etc.
        """
        if np.random.rand() < 0.1:
            return 0.0
        else:
            return 1.0

    def change_agent_tasks(self):
        #TODO: insert logic for changing tasks (i.e. creating or removing tasks)
        self._tasks = self._tasks

        self.net_benefits = np.array([task.benefit*task.agent_cost_mult for task in self._tasks])

    def update_tasks_from_neighbors(self):
        """
        Given a list of neighbors, update the agent's tasks based on
        neighbor's tasks which have more recent updates, or which have
        entirely new tasks.
        """
        for n in self.neighbors:
            for neighbor_task in self.auction.agents[n].public_tasks:
                #If the neighbor task list has an entirely new task, add it to the agent's task list.
                #Here, we want to include the price and high bidder information, because it's the most
                #updated we have.
                #However, we don't want to include the agent cost multiplier, because we want to calculate
                #it fresh for this agent.
                if neighbor_task.loc not in [task.loc for task in self._tasks]:
                    new_task = deepcopy(neighbor_task)
                    new_task.agent_cost_mult = self._calc_cost_mult_for_task(new_task)
                    self._tasks.append(new_task)
                #Otherwise, if the task already exists in the task list, update it if the neighbor's task is newer
                else:
                    for task in self._tasks:
                        if task.loc == neighbor_task.loc:
                            if task.update_time < neighbor_task.update_time:
                                task.benefit = neighbor_task.benefit
                                task.update_time = neighbor_task.update_time
                                break
                    

    def update_agent_prices_bids(self):
        #grab this agent's prices and high bidders from the previous iteration
        agent_public_prices = np.array([t.price for t in self.public_tasks])
        agent_public_high_bidders = np.array([t.price for t in self.public_tasks])

        #Assemble array of prices and high bidders from neighbors and self into one array
        all_neighbor_prices_array = np.array(agent_public_prices)
        all_neighbor_highest_bidders_array = np.array(agent_public_high_bidders)
        for n in self.neighbors:
            neighbor_prices = [t.price for t in self.auction.agents[n].public_tasks]
            neighbor_prices = np.vstack((neighbor_prices, self.auction.agents[n].public_prices))
            highest_bidders = np.vstack((highest_bidders, self.auction.agents[n].public_high_bidders))

        max_prices = np.max(neighbor_prices, axis=0)
        
        # Filter the high bidders by the ones that have the max price, and set the rest to -1.
        # Grab the highest index max bidder to break ties.
        max_price_bidders = np.where(neighbor_prices == max_prices, highest_bidders, -1)
        self._high_bidders = np.max(max_price_bidders, axis=0)

        if max_prices[self.choice] >= self.public_prices[self.choice] and self._high_bidders[self.choice] != self.id:
            best_net_value = np.max(self.net_benefits - max_prices)
            second_best_net_value = np.partition(self.net_benefits - max_prices, -2)[-2] #https://stackoverflow.com/questions/33181350/quickest-way-to-find-the-nth-largest-value-in-a-numpy-matrix

            self.choice = np.argmax(self.net_benefits-max_prices) #choose the task with the highest benefit to the agent
            
            self._high_bidders[self.choice] = self.id
            
            inc = best_net_value - second_best_net_value + self.eps

            self._prices = max_prices
            self._prices[self.choice] = max_prices[self.choice] + inc
        else:
            #Otherwise, don't change anything and just update prices
            #based on the new info from other agents.
            self._prices = max_prices

    def publish_agent_info(self):
        """
        Copies information about the agents tasks to the public variable.
        """
        self.agent_tasks_stable = True
        if len(self._tasks) == len(self.public_tasks):
            for task, prev_task in zip(self._tasks, self.public_tasks):
                if task.price != prev_task.price or task.high_bidder != prev_task.high_bidder:
                    self.agent_tasks_stable = False
                    break
        else:
            self.agent_tasks_stable = True

        self.public_tasks = deepcopy(self._tasks)