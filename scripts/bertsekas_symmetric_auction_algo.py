import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm for symmetric 
instances. That is, the number of agents equals the number of tasks. I assume 
all agent-task pairs are assignable."""

n = 10
num_agents = n
num_tasks = 2*n

# the assignment
assignment = [-1]*num_agents
benefits = np.random.randn(num_agents,num_tasks)
epsilon = 1e-3 # min bid increment
prices = np.zeros(num_tasks) # task prices

while sum(j == -1 for j in assignment) > 0:
    bids = dict() # bids is kept in a dict for ease 
    for i in range(num_agents):
        if assignment[i] == -1:
            # bidding process
            values_i = benefits[i,:] - prices
            j_i = np.argmax(values_i)
            w_i = np.sort(values_i)[-2]
            bids[i, j_i] = benefits[i,j_i] - w_i + epsilon

    for j in range(num_tasks):
        # assignment process
        P_j = [pair[0] for pair in bids.keys() if pair[1] == j]
        if len(P_j) > 0:
            prices[j] = max({bids[i,j] for i in P_j})
            i_j = P_j[0]
            remove = {i for i,k in enumerate(assignment) if k == j}
            for i in remove:
                assignment[i] = -1
            assignment[i_j] = j

# centeralized comptuation 
opt_assignment = methods.solve_centralized(benefits).tolist()
print("cent. computed assignment: \n", assignment, np.round(methods.cost(benefits, assignment), 3))
print("dist. computed assignment: \n", opt_assignment, np.round(methods.cost(benefits, opt_assignment), 3))
