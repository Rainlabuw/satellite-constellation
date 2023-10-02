import numpy as np

class AuctionAgent(object):
    def __init__(self, auction, id, tasks, neighbors):
        self.auction = auction
        self.id = id
        self.tasks = tasks

        #Calculate true benefits based on agent cost multipliers
        # (i.e. having lower fuel than expected would imply a cost mult <1)
        self._init_agent_cost_mults()
        self.net_benefits = np.array([task.benefit*task.agent_cost_mult for task in tasks])

        self.neighbors = neighbors

        #make initial choice in a greedy way
        self.choice = np.argmax(self.net_benefits)

        self.eps = 0.01

        self._prices = np.zeros_like(self.net_benefits) #private prices
        self.public_prices = np.zeros_like(self.net_benefits)

        self._high_bidders = np.zeros_like(self.net_benefits) #private bidders
        self.public_high_bidders = np.zeros_like(self.net_benefits)

        self.agent_prices_stable = False

    def __repr__(self):
        ret_str = f"Agent {self.id}, neighbors {self.neighbors}\n"
        ret_str += f"\tCurrent prices: {self.public_prices}"
        ret_str += f"\tCurrent high bidders: {self.public_high_bidders}"
        ret_str += f"\tCurrent choice: {self.choice}"
        return ret_str
    
    def _init_agent_cost_mults(self):
        for task in self.tasks:
            if np.random.rand() < 0.1:
                task.agent_cost_mult = 0.0
            else:
                task.agent_cost_mult = 1.0

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

    def publish_agent_prices_bids(self):
        #Determine if prices and bids have changed since the last iteration.
        if np.array_equal(self._prices, self.public_prices) and np.array_equal(self._high_bidders, self.public_high_bidders):
            self.agent_prices_stable = True
        else:
            self.agent_prices_stable = False

        self.public_prices = np.copy(self._prices)
        self.public_high_bidders = np.copy(self._high_bidders)