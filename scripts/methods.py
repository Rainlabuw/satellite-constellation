import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize

#~~~~~~~~~~~~~~~~~~~GRAPH STUFF~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~SOLUTION METHODS~~~~~~~~~~~~~~
def solve_centralized(benefits):
    _, col_ind = scipy.optimize.linear_sum_assignment(benefits, maximize=True)
    return col_ind

#~~~~~~~~~~~~~~~~AUCTION UTILS~~~~~~~~~~~~~~~~~~~~
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

def check_almost_equilibrium(auction):
    """
    Checks the epsilon with which the results of an auction
    are within almost equilibrium.
    """
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

def convert_central_sol_to_assignment_mat(n, m, assignments):
    """
    Converts a list of column indices to an assignment matrix.
    (column indices are the output from scipy.optimize.linear_sum_assignment)

    i.e. for n=m=3, [1,2,0] -> [[0,1,0],[0,0,1],[1,0,0]]
    """
    assignment_mat = np.zeros((n, m))
    for i, assignment in enumerate(assignments):
        assignment_mat[i, assignment] = 1

    return assignment_mat

def convert_agents_to_assignment_matrix(agents):
    """
    Convert list of agents as returned by auction to assignment matrix
    """
    assignment_matrix = np.zeros((len(agents), len(agents[0].benefits)))
    for i, agent in enumerate(agents):
        assignment_matrix[i, agent.choice] = 1
    return assignment_matrix

#~~~~~~~~~~~~~~~~~~~~HANDOVER PENALTY STUFF~~~~~~~~~~~~~~~
def calc_handover_penalty(init_assignment, assignments, lambda_):
    """
    Given an initial assignment and a list of assignment matrices,
    calculates the handover penalty associated with them,
    according to the Frobenius norm definition.

    If init_assignment is None, the the handover penalty from the first
    step is zero.

    NOTE: this provides a positive value, so to get the penalty you should subtract
    this from the total benefits.
    """
    handover_pen = 0

    if init_assignment is not None:
        handover_pen += np.linalg.norm(np.sqrt(lambda_/2)*(assignments[0] - init_assignment))**2

    for i in range(len(assignments)-1):
        new_assign = assignments[i+1]
        old_assign = assignments[i]

        handover_pen += np.linalg.norm(np.sqrt(lambda_/2)*(new_assign - old_assign))**2

    return handover_pen

def calc_distance_btwn_solutions(agents1, agents2):
    """
    Calc how many switches were made between two assignments,
    given lists of agents as would be returned by an auction.
    """
    dist = 0
    for agent1, agent2 in zip(agents1, agents2):
        if agent1.choice != agent2.choice:
            dist += 1

    return dist

#~~~~~~~~~~~~~~~~~~~~BENEFIT MATRIX UTILITIES~~~~~~~~~~~~~~
def generate_benefits_over_time(n, m, T, t_final, scale_min=0.5, scale_max=2):
    """
    lightweight way of generating "constellation-like" benefit matrices.
    """
    benefits = np.zeros((n,m,T))
    for i in range(n):
        for j in range(m):
            #where is the benefit curve maximized
            time_center = np.random.uniform(0, t_final)

            #how wide is the benefit curve
            time_spread = np.random.uniform(0, t_final/2)

            #how high is the benefit curve
            benefit_scale = np.random.uniform(scale_min, scale_max)

            #iterate from time zero to t_final with 100 steps in between
            for t_index, t in enumerate(np.linspace(0, t_final, T)):
                #calculate the benefit at time t
                benefits[i,j,t_index] = benefit_scale*np.exp(-(t-time_center)**2/time_spread**2)
    return benefits

def add_handover_pen_to_benefit_matrix(benefits, prev_assign, lambda_):
    adjusted_benefits = np.where(prev_assign == 1, benefits, benefits - lambda_*2)
    return adjusted_benefits