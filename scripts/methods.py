import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import scipy.optimize
from astropy import units as u

# Set the printing options to display more entries
np.set_printoptions(threshold=np.inf)

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
    """
    Solve assignment problem for a single n x m benefit matrix
    using the scipy implementation. Much faster than our python
    distributed implementation.

    Returns a list, col_ind, which contains the task id j assigned
    to agent i in index i of the list.
    """
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
    Convert list of agents as returned by auction to assignment matrix.
    (grab the choice in each Agent's .choice attribute and put it into a matrix.)
    """
    assignment_matrix = np.zeros((len(agents), len(agents[0].benefits)))
    for i, agent in enumerate(agents):
        assignment_matrix[i, agent.choice] = 1
    return assignment_matrix

def is_assignment_mat_sequence_valid(assignment_mat_seq):
    """
    Checks if an assignment matrix is valid.
    """
    for assignment_mat in assignment_mat_seq:
        for i in range(assignment_mat.shape[0]):
            if assignment_mat[i,:].sum() != 1:
                print(f"agent {i} not doing exactly 1 task")
                return False
        for j in range(assignment_mat.shape[1]):
            if assignment_mat[:,j].sum() > 1:
                print(f"too many doing task {j}")
                return False
    return True

#~~~~~~~~~~~~~~~~~~~~GENERIC HANDOVER PENALTY STUFF~~~~~~~~~~~~~~
def generic_handover_state_dep_fn(benefits, prev_assign, lambda_, task_trans_state_dep_scaling_mat=None):
    """
    Adjusts a 2D benefit matrix to account for generic handover penalty (i.e. constant penalty for switching tasks).

    task_trans_state_dep_scaling_mat is a matrix which determines which transitions are active.
        It is m x m, where entry ij is the state dependence multiplier that should be applied when switching from task i to task j.
        (If it is None, then all transitions are scaled by 1.)
    Then, prev_assign @ task_trans_state_dep_scaling_mat is the matrix which entries of the benefit matrix should be adjusted.
    """
    if prev_assign is None: return benefits

    m = benefits.shape[1]
    if task_trans_state_dep_scaling_mat is None:
        task_trans_state_dep_scaling_mat = np.ones((m,m))
    state_dep_scaling = prev_assign @ task_trans_state_dep_scaling_mat

    return np.where(prev_assign == 0, benefits-lambda_*state_dep_scaling, benefits)

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

#~~~~~~~~~~~~~~~~~~~~STATE DEPENDENT VALUE STUFF~~~~~~~~~~~~~~
def calc_assign_seq_state_dependent_value(init_assignment, assignments, benefits, lambda_,
                                          state_dep_fn=generic_handover_state_dep_fn, task_trans_state_dep_scaling_mat=None):
    state_dependent_value = 0

    benefit_hat = np.copy(benefits[:,:,0])
    if init_assignment is not None: #adjust based on init_assignment if it exists
        benefit_hat = state_dep_fn(benefits[:,:,0], init_assignment, lambda_, task_trans_state_dep_scaling_mat)
    state_dependent_value += (benefit_hat * assignments[0]).sum()

    for k in range(len(assignments)-1):
        benefit_hat = state_dep_fn(benefits[:,:,k+1], assignments[k], lambda_, task_trans_state_dep_scaling_mat)
        state_dependent_value += (benefit_hat * assignments[k+1]).sum()

    return state_dependent_value

def calc_assign_seq_handover_penalty(init_assignment, assignments, lambda_, benefits=None):
    """
    Given an initial assignment and a list of assignment matrices,
    calculates the handover penalty associated with them,
    according to the Frobenius norm definition.

    If init_assignment is None, the the handover penalty from the first
    step is zero.

    If a benefit matrix is provided, then if the assignment switches between 
    two tasks with zero value, then no handover penalty is applied because 
    the satellite is doing nothing anyway.

    NOTE: this provides a positive value, so to get the penalty you should subtract
    this from the total benefits.
    """
    handover_pen = 0

    if init_assignment is not None:
        handover_pen += np.linalg.norm(np.sqrt(lambda_/2)*(assignments[0] - init_assignment))**2

    for k in range(len(assignments)-1):
        new_assign = assignments[k+1]
        old_assign = assignments[k]

        #iterate through agents
        for i in range(new_assign.shape[0]):
            old_task_assigned = np.argmax(old_assign[i,:])
            new_task_assigned = np.argmax(new_assign[i,:])

            if benefits is not None and benefits[i,new_task_assigned,k+1] == 0:
                pass #add no penalty
            elif old_task_assigned != new_task_assigned:
                handover_pen += lambda_

    return handover_pen

def calc_value_and_num_handovers(chosen_assignments, benefits, init_assignment, lambda_, non_assign_pen=True):
    """
    Given a sequence of assignments, an initial assignment, and a benefit matrix,
    returns the total value and the number of handovers.
    """
    total_benefit = 0
    for i, chosen_ass in enumerate(chosen_assignments):
        curr_benefit = benefits[:,:,i]
        total_benefit += (curr_benefit * chosen_ass).sum()

    if non_assign_pen: #If a penalty should be applied 
        handover_pen = calc_assign_seq_handover_penalty(init_assignment, chosen_assignments, lambda_, benefits=None)
    else:
        handover_pen = calc_assign_seq_handover_penalty(init_assignment, chosen_assignments, lambda_, benefits=benefits)
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

def add_handover_pen_to_benefit_matrix(benefits, prev_assign, lambda_, non_assign_pen=True):
    """
    Adjusts the benefits matrix to account for handover penalties.

    Expects to take in a 3D array, with potentially multiple timesteps in the last axis, but only
    applies the penalty to the first benefit matrix (in index 0 of the time axis). 
    """
    #If there is no penalty for switching to a non assignment, the don't add a penalty
    #if the new benefits are zero (the task is not valid)
    if not non_assign_pen:
        adjusted_first_benefits = np.where((prev_assign == 0) & (benefits[:,:,0] != 0), benefits[:,:,0]-lambda_, benefits[:,:,0])
    else:
        adjusted_first_benefits = np.where(prev_assign == 0, benefits[:,:,0]-lambda_, benefits[:,:,0])
    
    adjusted_benefits = np.copy(benefits)
    adjusted_benefits[:,:,0] = adjusted_first_benefits
    
    return adjusted_benefits

#~~~~~~~~~~~~~~~~~~~~HAAL UTILITIES~~~~~~~~~~~~~~
def build_time_interval_sequences(all_time_intervals, len_window):
    """
    Recursively constructs all possible time interval sequences from the set of all time intervals.

    Implements the logic behind BUILD_TIME_INTERVAL_SEQUENCES from the paper.
    """
    all_time_interval_sequences = []

    def build_time_interval_sequences_rec(all_time_intervals, time_interval_sequence, len_window):
        #Grab the most recent timestep from the end of the current sol
        if time_interval_sequence == []:
            most_recent_timestep = -1 #set it to -1 so that time intervals starting w 0 will be selected
        else:
            most_recent_timestep = time_interval_sequence[-1][-1]

        #When we have an time interval seq which ends at the last timestep, we're done
        #and can add it to the list
        if most_recent_timestep == (len_window-1):
            all_time_interval_sequences.append(tuple(time_interval_sequence))
        else:
            #Iterate through all of the time intervals, looking for ones that start where this one ended
            for time_interval in all_time_intervals:
                if most_recent_timestep == time_interval[0]-1:
                    build_time_interval_sequences_rec(all_time_intervals, time_interval_sequence + [time_interval], len_window)

    build_time_interval_sequences_rec(all_time_intervals, [], len_window)

    return all_time_interval_sequences

def generate_all_time_intervals(L):
    """
    Generates all possible time intervals from the next few timesteps.

    Implements GENERATE_ALL_TIME_INTERVALS from the paper.
    """
    all_time_intervals = []
    for i in range(L):
        for j in range(i,L):
            all_time_intervals.append((i,j))
        
    return all_time_intervals

#~~~~~~~~~~~~~~~~~~~~ ORBITAL MECHANICS STUFF ~~~~~~~~~~~~~~
def calc_distance_based_benefits(sat, task, k):
    """
    Given a satellite, a task, and a timestep computes the benefit of the satellite.

    Benefit here is zero if the task is not visible from the satellite,
    and is a gaussian centered at the minimum distance away from the task,
    and dropping to 5% of the max value at the furthest distance away from the task.
    """
    if task.loc.is_visible(*sat.orbit.r):
        body_rad = np.linalg.norm(sat.orbit._state.attractor.R.to_value(u.km))
        max_distance = np.sqrt(np.linalg.norm(sat.orbit.r.to_value(u.km))**2 - body_rad**2)

        gaussian_height = task.benefit[k]
        height_at_max_dist = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-max_distance**2/(2*np.log(height_at_max_dist/gaussian_height)))

        sat_height = np.linalg.norm(sat.orbit.r.to_value(u.km)) - body_rad
        task_dist = np.linalg.norm(task.loc.cartesian_cords.to_value(u.km) - sat.orbit.r.to_value(u.km)) - sat_height
        task_benefit = gaussian_height*np.exp(-task_dist**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0

    return task_benefit

def calc_fov_benefits(sat, task, k):
    """
    Given a satellite, a task, and a timestep, computes the benefit of the satellite.

    We calculate the angle between the satellite and the task, and then
    use a gaussian to determine the benefit, starting at 5% of the benefit
    when the angle between the satellite and the task is the maximum FOV,
    and rising to the maximum when the satellite is directly overhead.
    """
    sat_r = sat.orbit.r.to_value(u.km)
    sat_to_task = task.loc.cartesian_cords.to_value(u.km) - sat_r

    angle_btwn = np.arccos(np.dot(-sat_r, sat_to_task)/(np.linalg.norm(sat_r)*np.linalg.norm(sat_to_task)))
    angle_btwn *= 180/np.pi #convert to degrees

    if angle_btwn < sat.fov and task.loc.is_visible(*sat.orbit.r):
        gaussian_height = task.benefit[k]
        height_at_max_fov = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-sat.fov**2/(2*np.log(height_at_max_fov/gaussian_height)))

        task_benefit = gaussian_height*np.exp(-angle_btwn**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0
    
    return task_benefit

def generate_safe_L(timestep, sat):
    """
    Generates the optimal L given a satellite and a timestep size, as described in the paper.

    Calculates this by determining the angle of a satellites ground visibility,
    and then calculating the time taken to cover that angular distance based on period.
    """
    timestep = timestep.to(u.min)

    earth = sat.orbit.attractor

    earth_r = earth.R.to_value(u.km)
    sat_r = np.linalg.norm(sat.orbit.r.to_value(u.km))

    #Max FOV is when the angle is a tangent to the surface of the earth
    max_fov = np.arcsin(earth_r/sat_r)*180/np.pi
    if max_fov < sat.fov: print(f"Lowering FOV to {max_fov}")
    sat.fov = min(sat.fov, max_fov)

    #Subtract from 180 to ensure the angle is in the correct quadrant
    third_angle = (180 - np.arcsin(sat_r/earth_r*np.sin(sat.fov*np.pi/180))*180/np.pi)
    delta_angle = (2*(180 - sat.fov - third_angle))

    min_to_travel_delta_angle = sat.orbit.period.to(u.min) * delta_angle/360

    L = int(np.ceil(min_to_travel_delta_angle/timestep))

    return L