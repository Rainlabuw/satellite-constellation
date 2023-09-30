import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm for symmetric 
instances. That is, the number of agents equals the number of tasks. I assume 
all agent-task pairs are assignable."""

n = 50
num_agents = n
num_tasks = n

# the assignment
assignment = dict()
unassigned_agents = {i for i in range(num_agents)}
benefits = np.random.randn(num_agents,num_tasks)
epsilon = 1e-3 # min bid increment
prices = np.zeros(num_tasks) # task prices

while len(unassigned_agents) > 0:
    bids = dict() # bids is kept in a dict for ease 
    for i in unassigned_agents:
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
            remove = [k for k,v in assignment.items() if v == j]
            for i in remove:
                unassigned_agents.add(i)
                del assignment[i]
            assignment[i_j] = j
            unassigned_agents.remove(i_j)


cost = 0
for i, j in assignment.items():
    print(f"agent {i} is assigned to task {j}")
    cost += benefits[i,j]
print(cost)

# centeralized comptuation 
print(methods.optimal_assignment(benefits))
