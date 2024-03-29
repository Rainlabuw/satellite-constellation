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
    - reset: reset the environment to its initial state.
    - get_state: get a state representation of the environment for use in RL.
    - update_benefit_info: update the benefit info for the environment.

Each environment should have a benefit functiona 3D slice of the proximity matrix,
the previous assignments, lambda_, and any other information it might need in benefit_info.
It will then output a 3D matrix of benefits, where the benefit of each task at each timestep is calculated.
"""

class SimpleAssignEnv(object):
    """
    Assignment environment corresponding to a generic assignment problem with a constant
    lambda handover penalty, a benefit function based on the variance of the task.
    """
    def __init__(self, sat_prox_mat, init_assignment, lambda_,
                 T_trans=None, task_benefits=None):
        self.sat_prox_mat = sat_prox_mat
        self.init_assignment = init_assignment
        self.curr_assignment = init_assignment
        self.lambda_ = lambda_

        self.k = 0

        #Build benefit_info
        self.benefit_info = BenefitInfo()
        self.benefit_info.T_trans = T_trans
        self.benefit_info.task_benefits = task_benefits

        self.benefit_fn = simple_handover_pen_benefit_fn

        self.k = 0
        

    def update_benefit_info(self):
        """
        Update the benefit info for the environment. In this case,
        the benefit info stays constant
        """
        pass 

    def step(self, assignment):
        """
        Returns state, value, and whether the environment is done.
        """
        benefit_hat = self.benefit_fn(self.sat_prox_mat[:,:,self.k], self.curr_assignment, self.lambda_)
        value = np.sum(benefit_hat * assignment)

        self.k += 1
        self.curr_assignment = assignment

        return self.get_state(), value, self.k>=self.sat_prox_mat.shape[2]

    def reset(self):
        self.curr_assignment = self.init_assignment
        self.k = 0

    def get_state(self):
        """
        Get a state representation of the environment
        for use in RL
        """
        return None
    
def simple_handover_pen_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Adjusts a 3D benefit matrix to account for generic handover penalty (i.e. constant penalty for switching tasks).

    benefit_info is an object which can contain extra information about the benefits - it should store:
     - task_benefits is a m-length array of the baseline benefits associated with each task.
     - T_trans is a matrix which determines which transitions between TASKS are penalized.
        It is m x m, where entry ij is the state dependence multiplier that should be applied when switching from task i to task j.
        (If it is None, then all transitions between different tasks are scaled by 1.)
    Then, prev_assign @ T_trans is the matrix which entries of the benefit matrix should be adjusted.
    """
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    L = sat_prox_mat.shape[2]

    #Generate a matrix which determines the benefits of each task at each timestep.
    if benefit_info is not None and benefit_info.task_benefits is not None:
        task_benefits = benefit_info.task_benefits
    else:
        task_benefits = np.ones(m)
    task_benefits = np.tile(task_benefits, (n,1))
    task_benefits = np.repeat(task_benefits[:,:,np.newaxis], L, axis=2)

    if prev_assign is None:
        benefits_hat = sat_prox_mat * task_benefits
    else:
        try:
            T_trans = benefit_info.T_trans
        except AttributeError:
            T_trans = None

        if T_trans is None:
            T_trans = np.ones((m,m)) - np.eye(m) #default to all transitions (except nontransitions) being penalized

        state_dep_scaling = prev_assign @ T_trans

        benefits_hat = sat_prox_mat * task_benefits
        benefits_hat[:,:,0] = benefits_hat[:,:,0]-lambda_*state_dep_scaling

    if init_dim == 2: 
        benefits_hat = np.squeeze(benefits_hat, axis=2)
    return benefits_hat