import numpy as np 
import matplotlib.pyplot as plt
import methods
import networkx as nx

num_agents = 50
num_tasks = 100
G = methods.rand_connected_graph(num_agents)
Delta = nx.diameter(G)
benefits = np.random.rand(num_agents, num_tasks)
prices = np.zeros((num_agents, num_tasks))
assignment = [i for i in range(num_agents)]

bidders = [[-1]*num_tasks]*num_agents
tol = 1e-5
epsilon = 1e-5/num_agents**2
next_prices = prices
next_bidders = bidders
next_assignment = [0]*num_agents

unchanged_prices_count = 0
rounds_count = 0
while unchanged_prices_count < Delta:
    flag = False
    for i in range(num_agents):
        N_i = list(G.neighbors(i))
        N_i.append(i)
        N_i = sorted(N_i)
        for j in range(num_tasks):
            max_price = np.max(prices[N_i, j])
            next_prices[i,j] = max_price
            highest_bidders = [
                k for k in N_i if max_price - prices[k,j] < tol
            ]
            max_bidders = [bidders[k][j] for k in highest_bidders]
            next_bidders[i][j] = max(max_bidders)
        if prices[i, assignment[i]] <= next_prices[i, assignment[i]] \
            and bidders[i][assignment[i]] != i:
            flag = True
            values = benefits[i,:] - next_prices[i,:]
            max_value = max(values)
            most_valuable_tasks = [
                j for j in range(num_tasks) if max_value - values[j] < tol
            ]
            next_assignment[i] = most_valuable_tasks[0]
            next_bidders[i][next_assignment[i]] = i
            values = sorted(values)
            v = max_value
            w = sorted(values)[-2]
            gamma = v - w + epsilon
            next_prices[i, next_assignment[i]] \
                = prices[i, next_assignment[i]] + gamma
        else:
            next_assignment[i] = assignment[i]
    assignment = next_assignment
    if flag:
        unchanged_prices_count = 0
    else:
        unchanged_prices_count += 1
    prices = next_prices
    bidders = next_bidders
    rounds_count += 1
    print(rounds_count)

# centeralized comptuation 
opt_assignment = methods.solve_centralized(benefits).tolist()
print("cent. computed assignment: \n", assignment, np.round(methods.cost(benefits, opt_assignment), 3))
print("dist. computed assignment: \n", opt_assignment, np.round(methods.cost(benefits, assignment), 3))
