import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize

def rand_connected_graph(num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    num_additional_edges = int(num_nodes**(1.3))
    for _ in range(num_additional_edges):
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)
        if node1 != node2:
            G.add_edge(node1, node2)

    return G

def plot_graph(G: nx.classes.graph.Graph) -> None:
    nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.show()

def cost(benefits: np.ndarray, assignment: list) -> float:
    """Returns the sum of each agent's benefit under the given assignment.

    assignment: (dict) a map from agent index to task index. We use dict instead
    of list since in computation, we can easily add and delete entries. 

    benefits: (np.ndarray) a (num_agents, num_tasks)-array of the benefits for 
    agent i executing task j 

    Returns:
    out: (float) the cost
    """
    out = 0
    for i, j in enumerate(assignment):
        out += benefits[i,j]
    return out

def solve_centralized(benefits):
        _, col_ind = scipy.optimize.linear_sum_assignment(benefits, maximize=True)
        return col_ind

def check_almost_equilibrium(auction):
    max_eps = -np.inf
    for agent in auction.agents:
        max_net_value = -np.inf

        curr_ben = agent.benefits[agent.choice] - agent.public_prices[agent.choice]
        for j in range(auction.n_tasks):
            net_ben = agent.benefits[j] - agent.public_prices[j]

            if net_ben > max_net_value:
                max_net_value = net_ben
        
        eps = max_net_value - curr_ben
        if eps > max_eps:
            max_eps = eps

    return max_eps

def calc_distance_btwn_solutions(agents1, agents2):
    dist = 0
    for agent1, agent2 in zip(agents1, agents2):
        if agent1.choice != agent2.choice:
            dist += 1

    return dist

def convert_central_sol_to_assignment_mat(n, m, assignments):
    assignment_mat = np.zeros((n, m))
    for i, assignment in enumerate(assignments):
        assignment_mat[i, assignment] = 1

    return assignment_mat

def calc_handover_penalty(init_assignment, assignments, lambda_):
    """
    Given an initial assignment and a list of assignment matrices,
    calculates the handover penalty associated with them,
    according to the Frobenius norm definition.

    If init_assignment is None, the the handover penalty from the first
    step is zero.
    """
    handover_pen = 0

    if init_assignment is not None:
        assign_diff = np.sqrt(lambda_/2)*(assignments[0] - init_assignment)
        handover_pen += np.sum(assign_diff**2)

    for i in range(len(assignments)-1):
        new_assign = assignments[i+1]
        old_assign = assignments[i]

        assign_diff = np.sqrt(lambda_/2)*(new_assign - old_assign)
        handover_pen += np.sum(assign_diff**2)

    return handover_pen