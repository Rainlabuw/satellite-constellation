import numpy as np 
import matplotlib.pyplot as plt
import methods
import scipy.optimize
import random

num_agents = 5
num_tasks = 8

beta = np.random.rand(num_agents, num_tasks)
p = np.zeros((num_agents, num_tasks))
alpha = np.arange(0, num_agents, 1).astype('int')
b = np.zeros((num_agents, num_tasks)).astype('int')
epsilon = .01
next_p = np.zeros((num_agents, num_tasks))
next_b = np.zeros((num_agents, num_tasks)).astype('int')
next_alpha = np.zeros(num_agents)

tol = 1e-5
cost = []
for t in range(100):
    
    total_assignment_benefit = 0
    for i in range(num_agents):
        total_assignment_benefit += beta[i, alpha[i]]
    cost.append(total_assignment_benefit)
    print(t,alpha,total_assignment_benefit)  
    for i in range(num_agents):
        N_i = list(G.neighbors(i))
        N_i.append(i)
        for j in range(num_tasks):
            # update prices
            next_p[i,j] = np.max(p[N_i, j])

            # update highest bidders
            max_value = next_p[i,j]
            max_indices = np.argwhere(abs(max_value - p[N_i,j]) < tol)[:,0]
            next_b[i,j] = np.max(b[max_indices, j])
            next_b = next_b.astype('int')

        if p[i, alpha[i]] <= next_p[i, alpha[i]] and next_b[i, alpha[i]] != i:
            max_value = np.max(beta[i,:] - next_p[i,:])
            L = np.argwhere(abs(max_value - (beta[i,:] - next_p[i,:])) < tol)
            random.shuffle(L)
            next_alpha[i] = L[0]
            next_alpha = next_alpha.astype('int')
            next_b[i, next_alpha[i]] = i
            L = list(beta[i,:] - p[i,:])
            L.sort()
            v_i = L[-1]
            w_i = L[-2]
            gamma_i = v_i - w_i + epsilon
            next_p[i, next_alpha[i]] = p[i, next_alpha[i]] + gamma_i
        else:
            next_alpha[i] = int(alpha[i])
    p = next_p
    b = next_b.astype('int')
    alpha = next_alpha.astype('int')
out = solve_centralized(beta)
print(out)
plt.plot(cost)
plt.grid()
plt.show()
