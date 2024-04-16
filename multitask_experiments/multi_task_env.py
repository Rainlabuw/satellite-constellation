from common.methods import *

"""
An assignment environment is an environment that tracks all elements of a sequential assignment problem,
and allows algorithms to create assignments over time that maximize some benefit function.

All assignment environments should recieve the following inputs, at least:
    - sat_prox_mat: a 3D matrix of the proximity of each satellite to each task at each timestep.
    - init_assignment: an initial assignment of tasks to satellites.
    - lambda_: a handover penalty parameter.

The environment should have the following methods:
    - step: after recieving an assignment, advance the environment by one timestep.
        This includes updating the benefit info.
    - reset: reset the environment to its initial state.
    - get_state: get a state representation of the environment for use in RL.
    - update_benefit_info: update the benefit info for the environment.

Each environment should have a benefit functiona 3D slice of the proximity matrix,
the previous assignments, lambda_, and any other information it might need in benefit_info.
It will then output a 3D matrix of benefits, where the benefit of each task at each timestep is calculated.
"""

class MultiTaskAssignEnv(object):
    """
    Assignment environment corresponding to a generic assignment problem with agents who can
    complete multiple tasks each.

    We handle the fact that each agent can accomplish multiple tasks at once by creating synthetic agents.
    i.e. there is an agent for the first task that agent 1 can do, another agent for the second task agent 1 can do, etc.

    These are ordered in matrices and graphs as follows:
    - The first n synthetic agents are the agents that can complete the first task.
    - The next n synthetic agents are second tasks of each of the first n agents (i.e. synth agent n corresponds to agent 0's second task).
    - etc.

    INPUTS:
    tasks_per_agent is a scalar which determines how many tasks each agent can complete. (i.e. how many beams are on each satellite.)
    pct_benefit_for_each_further_task_per_agent is a scalar which determines how much benefit is gained for each 
        additional task completed by an agent. (i.e. using more beams has decreasing benefits)
    We expect the sat_prox_mat and graphs to be provided with information only on REAL agents, not synthetic ones.
        (i.e. size n, not size n*tasks_per_agent.)
    """
    def __init__(self, sat_prox_mat, init_assignment, lambda_, tasks_per_agent, 
                 pct_benefit_for_each_further_task_per_agent, graphs=None, task_benefits=None):
        #Note that we are not padding the sat_prox_mat, because when extra
        #beams are added, the padding will be different.
        self.sat_prox_mat = sat_prox_mat

        self.n = sat_prox_mat.shape[0]
        self.total_n = self.n * tasks_per_agent
        self.m = sat_prox_mat.shape[1]
        self.T = sat_prox_mat.shape[2]
        
        self.init_assignment = init_assignment
        self.curr_assignment = init_assignment
        self.lambda_ = lambda_

        #Adjust the graph to account for synthetic agents
        if graphs is None:
            graphs = [nx.complete_graph(self.total_n) for _ in range(self.T)]
        else:
            if graphs[0].number_of_nodes() == self.total_n:
                self.graphs = graphs
            #add connections to graph to account for synthetic agents if not already done
            else:
                for _ in range(1, tasks_per_agent): #only add for non-original tasks
                    for k in range(self.T):
                        grph = graphs[k]
                        for real_sat_num in range(self.n):
                            synthetic_sat_num = grph.number_of_nodes()
                            grph.add_node(synthetic_sat_num)
                            grph.add_edge(real_sat_num, synthetic_sat_num)
                            for neigh in grph.neighbors(real_sat_num):
                                grph.add_edge(neigh, synthetic_sat_num)

        self.k = 0

        #Build benefit_info
        self.benefit_info = BenefitInfo()
        self.benefit_info.task_benefits = task_benefits
        self.benefit_info.tasks_per_agent = tasks_per_agent
        self.benefit_info.pct_benefit_for_each_task = pct_benefit_for_each_further_task_per_agent

        self.benefit_fn = simple_handover_pen_multitask_benefit_fn

        self.k = 0

    def step(self, assignment):
        """
        Returns state, value, and whether the environment is done.
        """
        benefit_hat = self.benefit_fn(self.sat_prox_mat[:,:,self.k], self.curr_assignment, self.lambda_)
        value = np.sum(benefit_hat * assignment)

        self.k += 1
        self.curr_assignment = assignment

        return self.get_state(), value, self.k>=self.T

    def reset(self):
        self.curr_assignment = self.init_assignment
        self.k = 0

    def get_state(self):
        """
        Get a state representation of the environment
        for use in RL
        """
        return None
    
def simple_handover_pen_multitask_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Adjusts a 3D benefit matrix to account for generic handover penalty (i.e. constant penalty for switching tasks).

    benefit_info is an object which can contain extra information about the benefits - it should store:
     - tasks_per_agent is an integer describing how many tasks each agent can complete.
     - task_benefits is a m-length array of the baseline benefits associated with each task.
     - T_trans is a matrix which determines which transitions between TASKS are penalized.
        It is m x m, where entry ij is the state dependence multiplier that should be applied when switching from task i to task j.
        (If it is None, then all transitions between different tasks are scaled by 1.)
    Then, prev_assign @ T_trans is the matrix which entries of the benefit matrix should be adjusted.
    """
    if lambda_ is None: lambda_ = 0 #if lambda_ is not provided, then add no penalty so lambda_=0
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    L = sat_prox_mat.shape[2]

    #Create a matrix which has <tasks_per_agent> entries in the benefit matrix for each agent.
    padded_m = min(n*benefit_info.tasks_per_agent, m)
    benefits_hat = np.zeros((n*benefit_info.tasks_per_agent,padded_m,L))

    #Generate a matrix which determines the benefits of each task at each timestep.
    if benefit_info is not None and benefit_info.task_benefits is not None:
        task_benefits = benefit_info.task_benefits
    else:
        task_benefits = np.ones(m)
    task_benefits = np.tile(task_benefits, (n,1))
    task_benefits = np.repeat(task_benefits[:,:,np.newaxis], L, axis=2)

    for sat_task in range(benefit_info.tasks_per_agent):
        benefits_hat[sat_task*n:(sat_task+1)*n,:m,:] = sat_prox_mat * task_benefits * benefit_info.pct_benefit_for_each_further_task_per_agent**sat_task

    if prev_assign is None:
        pass
    else:
        try:
            T_trans = benefit_info.T_trans
        except AttributeError:
            T_trans = None

        if T_trans is None:
            T_trans = np.ones((m,m)) - np.eye(m) #default to all transitions (except nontransitions) being penalized

        state_dep_scaling = prev_assign @ T_trans
        benefits_hat[:,:m,0] -= lambda_*state_dep_scaling

    if init_dim == 2: 
        benefits_hat = np.squeeze(benefits_hat, axis=2)
    return benefits_hat