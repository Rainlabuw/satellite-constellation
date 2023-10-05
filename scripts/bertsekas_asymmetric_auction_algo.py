import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random
from itertools import product

"""In this script, I implement Bertsekas' auction algorithm for asymmetric 
instances. I assume all agent-task pairs are assignable. """


num_agents = 3000
num_tasks = 3000

# the assignment
assignment = [-1]*num_agents
unassigned_tasks = {j for j in range(num_tasks)}
benefits = np.random.randn(num_agents, num_tasks)
epsilon = 1e-3 # min bid increment
prices = np.zeros(num_tasks)
profits = np.max(benefits, axis=1)
lamb = 0
while sum(j == -1 for j in assignment) > 0:
    print(sum(j == -1 for j in assignment))
    # forward iteration
    bids = dict() # bids is kept in a dict for ease 
    for i in range(num_agents):
        if i in range(num_agents):
            if assignment[i] == -1:
                values_i = benefits[i,:] - prices
                j_i = np.argmax(values_i)
                w_i = np.sort(values_i)[-2]
                prices[j_i] = max(lamb, benefits[i,j_i] - w_i + epsilon)
                profits[i] = w_i - epsilon
                if lamb <= benefits[i, j_i] - w_i + epsilon:
                    remove = {k for (k,j) in enumerate(assignment) if j == j_i}
                    for k in remove:
                        assignment[k] = -1
                    assignment[i] = j_i
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
                if assignment[i_j] != j:
                    j_prime = assignment[i_j]
                    assignment[i_j] = -1
                    unassigned_tasks.add(j_prime)
                assignment[i_j] = j
                unassigned_tasks.remove(j)
            else:
                prices[j] = beta_j - epsilon
                aux = [i for i,p in enumerate(prices) if p < lamb]
                if len(aux) > num_tasks - num_agents:
                    lamb = np.max(prices[aux])



cost = 0
for i, j in enumerate(assignment):
    print(f"agent {i} is assigned to task {j}")
    cost += benefits[i,j]
print(cost)

# centeralized comptuation 
opt_assignment = methods.solve_centralized(benefits)
print(methods.cost(benefits, opt_assignment))

