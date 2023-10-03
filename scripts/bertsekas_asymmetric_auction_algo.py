import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm for asymmetric 
instances. I assume all agent-task pairs are assignable. """


num_agents = 50
num_tasks = 100

# the assignment
assignment = dict()
unassigned_agents = {i for i in range(num_agents)}
unassigned_tasks = {j for j in range(num_tasks)}
benefits = np.random.randn(num_agents, num_tasks)
epsilon = 1e-3 # min bid increment
prices = np.zeros(num_tasks)
profits = np.max(benefits, axis=1)
lamb = 0
while len(unassigned_agents) > 0:
    # forward iteration
    bids = dict() # bids is kept in a dict for ease 
    for i in range(num_agents):
        if i in unassigned_agents:
            values_i = benefits[i,:] - prices
            j_i = np.argmax(values_i)
            w_i = np.sort(values_i)[-2]
            prices[j_i] = max(lamb, benefits[i,j_i] - w_i + epsilon)
            profits[i] = w_i - epsilon
            if lamb <= benefits[i, j_i] - w_i + epsilon:
                remove = {k for (k,v) in assignment.items() if v == j_i}
                for k in remove:
                    del assignment[k]
                    unassigned_agents.add(k)
                assignment[i] = j_i
                unassigned_agents.remove(i)
                unassigned_tasks.discard(j_i)

    # backward iteration
    for j in range(num_tasks): 
        if j in unassigned_tasks and prices[j] > lamb:
            values_j = benefits[:,j] - profits
            i_j = np.argmax(values_j)
            values_j = np.sort(values_j)
            beta_j = values_j[-1]
            gamma_j = values_j[-2]
            if beta_j >= lamb + epsilon:
                prices[j] = max(lamb, gamma_j - epsilon)
                profits[i_j] = benefits[i_j, j] - max(lamb, gamma_j - epsilon)
                remove = {k for (k,v) in assignment.items() if k == i_j}
                if assignment[i_j] != j:
                    j_prime = assignment[i_j]
                    del assignment[i_j]
                    unassigned_tasks.add(j_prime)
                assignment[i_j] = j
                unassigned_agents.discard(i_j)
                unassigned_tasks.remove(j)
            else:
                prices[j] = beta_j - epsilon
                aux = [i for i,p in enumerate(prices) if p < lamb]
                if len(aux) > num_tasks - num_agents:
                    lamb = np.max(prices[aux])



cost = 0
for i, j in assignment.items():
    print(f"agent {i} is assigned to task {j}")
    cost += benefits[i,j]
print(cost)

# centeralized comptuation 
print(methods.cost(benefits, assignment))


