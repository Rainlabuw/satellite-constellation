import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm. I assume every 
agent can assume every task, and that the number of agents equals the number of 
tasks."""

n = 5
num_agents = n
num_tasks = n

# the assignment
assignment = dict()
unassigned_agents = {i for i in range(num_agents)}
benefits = np.random.randint(0, 10, (num_agents,num_tasks))
epsilon = 1e-3 # min bid increment
prices = np.zeros(num_tasks) # task prices

util_hist = []
while True:
    print(assignment)
    if len(unassigned_agents) == 0:
        break
    bids = dict() # bids is kept in a dict for ease 
    for i in unassigned_agents:
        # bidding process
        values_i = benefits[i,:] - prices
        j_i = np.argmax(values_i)
        w_i = np.sort(values_i)[-2]
        bids[i, j_i] = benefits[i,j_i] - w_i + epsilon

    for j in range(num_tasks):
        # assignment process
        P_j = []
        for pair in bids.keys():
            if pair[1] == j:
                P_j.append(pair[0])
        if len(P_j) > 0:
            prices[j] = max({bids[i,j] for i in P_j})
            i_j = P_j[0]
            for i in [i for i,v in assignment.items() if v == j]:
                unassigned_agents.add(i)
                del assignment[i]
            assignment[i_j] = j
            unassigned_agents.remove(i_j)


def solve_centralized(self):
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(self.benefits, maximize=True)
        if self.verbose:
            print("Centralized solution")
            for row, col in zip(row_ind, col_ind):
                print(f"Agent {row} chose task {col_ind[row]} with benefit {self.benefits[row, col]}")
            print(f"Total benefit: {self.benefits[row_ind, col_ind].sum()}")

        return row_ind, col_ind

cost = 0
for i, j in assignment.items():
    print(f"agent {i} is assigned to task {j}")
    cost += benefits[i,j]
print(cost)

# centeralized comptuation 
print(methods.optimal_assignment(benefits))
