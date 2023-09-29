import numpy as np 
import matplotlib.pyplot as plt
from methods import rand_connected_graph, plot_graph
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm. I assume every 
agent can assume every task, and that the number of agents equals the number of 
tasks."""

num_agents = 5
num_tasks = num_agents

# the assignment
S = [i for i in range(num_agents)] # initially assign agent i to task i
benefits = np.random.randint(0,10,(num_agents,num_tasks))
epsilon = 1e-6 # min bid increment
prices = np.zeros(num_agents) # task prices

util_hist = []
for t in range(100):
    total_util = 0
    for i in range(num_agents):
        total_util += benefits[i, S[i]]
    util_hist.append(total_util)

    bids = dict() # bids is kept in a dict for ease 
    for i in range(num_agents):
        # bidding process
        values_i = benefits[i,:] - prices
        j_i = np.argmax(values_i)
        w_i = np.sort(values_i)[-2]
        bids[i, j_i] = benefits[i,j_i] - w_i + epsilon

    for j in range(num_tasks):
        # assignments process
        P_j = []
        for pair in bids.keys():
            if pair[1] == j:
                P_j.append(pair[0])
        if len(P_j) > 0:
            prices[j] = max({bids[i,j] for i in P_j})
            S[P_j[0]] = j

plt.plot(util_hist)
plt.grid()
plt.show()

# Spencer: I think there is some issue in that we aren't converging to a single 
# assignment, but maybe that is part of the algorithm. WIP. 