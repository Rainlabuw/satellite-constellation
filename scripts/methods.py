import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
from astropy import units as u

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

def check_assign_matrix_validity(assignment_matrix):
    for i in range(assignment_matrix.shape[0]):
        if assignment_matrix[i,:].sum() != 1:
            return False
    for j in range(assignment_matrix.shape[1]):
        if assignment_matrix[:,j].sum() != 1:
            return False
    return True

#~~~~~~~~~~~~~~~~~~~~HANDOVER PENALTY STUFF~~~~~~~~~~~~~~~
def calc_assign_seq_handover_penalty(init_assignment, assignments, lambda_):
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

def calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_):
    """
    Given a sequence of assignments, an initial assignment, and a benefit matrix,
    returns the total value and the number of handovers.
    """
    total_benefit = 0
    for i, chosen_ass in enumerate(chosen_assignments):
        curr_benefit = benefits[:,:,i]
        total_benefit += (curr_benefit * chosen_ass).sum()
    handover_pen = calc_assign_seq_handover_penalty(init_assignment, chosen_assignments, lambda_)
    total_benefit -= handover_pen

    return total_benefit, handover_pen/lambda_

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
    adjusted_benefits = np.where(prev_assign == 1, benefits + lambda_, benefits)
    return adjusted_benefits

#~~~~~~~~~~~~~~~~~~~~ ORBITAL MECHANICS STUFF ~~~~~~~~~~~~~~
def calc_distance_based_benefits(sat, task):
    """
    Given a satellite and a task, computes the benefit of the satellite.

    Benefit here is zero if the task is not visible from the satellite,
    and is a gaussian centered at the minimum distance away from the task,
    and dropping to 5% of the max value at the furthest distance away from the task.
    """
    if task.loc.is_visible(*sat.orbit.r):
        body_rad = np.linalg.norm(sat.orbit._state.attractor.R.to_value(u.km))
        max_distance = np.sqrt(np.linalg.norm(sat.orbit.r.to_value(u.km))**2 - body_rad**2)

        gaussian_height = task.benefit
        height_at_max_dist = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-max_distance**2/(2*np.log(height_at_max_dist/gaussian_height)))

        sat_height = np.linalg.norm(sat.orbit.r.to_value(u.km)) - body_rad
        task_dist = np.linalg.norm(task.loc.cartesian_cords.to_value(u.km) - sat.orbit.r.to_value(u.km)) - sat_height
        task_benefit = gaussian_height*np.exp(-task_dist**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0

    return task_benefit

def calc_fov_benefits(sat, task):
    sat_r = sat.orbit.r.to_value(u.km)
    sat_to_task = task.loc.cartesian_cords.to_value(u.km) - sat_r

    angle_btwn = np.arccos(np.dot(-sat_r, sat_to_task)/(np.linalg.norm(sat_r)*np.linalg.norm(sat_to_task)))
    angle_btwn *= 180/np.pi #convert to degrees

    if angle_btwn < sat.fov and task.loc.is_visible(*sat.orbit.r):
        gaussian_height = task.benefit
        height_at_max_fov = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-sat.fov**2/(2*np.log(height_at_max_fov/gaussian_height)))

        task_benefit = gaussian_height*np.exp(-angle_btwn**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0
    
    return task_benefit

def generate_optimal_L(timestep, sat):
    """
    Generates the optimal L given a satellite and a timestep size.

    Calculates this by determining the angle of a satellites ground visibility,
    and then calculating the time taken to cover that angular distance based on period.
    """
    earth = sat.orbit.attractor

    earth_r = earth.R.to_value(u.km)
    sat_r = np.linalg.norm(sat.orbit.r.to_value(u.km))

    third_angle = (180 - np.arcsin(sat_r/earth_r*np.sin(sat.fov*np.pi/180))*180/np.pi)
    delta_angle = (2*(180 - sat.fov - third_angle))

    min_to_travel_delta_angle = sat.orbit.period.to(u.min) * delta_angle/360

    L = np.ceil(min_to_travel_delta_angle/timestep)

    return L