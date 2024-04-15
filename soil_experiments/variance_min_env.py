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

Each environment should have a benefit functiona 3D slice of the proximity matrix,
the previous assignments, lambda_, and any other information it might need in benefit_info.
It will then output a 3D matrix of benefits, where the benefit of each task at each timestep is calculated.
"""

class VarianceMinEnv(object):
    """
    Assignment environment corresponding to a generic assignment problem with a constant
    lambda handover penalty, a benefit function based on the variance of the task.
    """
    def __init__(self, sat_prox_mat, init_assignment, lambda_,
                 init_task_vars=None, var_add=None, base_sensor_var=0.1):
        #Pad benefit matrix to ensure that the number of tasks is at least as large as the number of agents
        self.unpadded_m = sat_prox_mat.shape[1]
        if sat_prox_mat.shape[1] < sat_prox_mat.shape[0]:
            padded_sat_prox_mat = np.zeros((sat_prox_mat.shape[0], sat_prox_mat.shape[0], sat_prox_mat.shape[2]))
            padded_sat_prox_mat[:sat_prox_mat.shape[0],:sat_prox_mat.shape[1],:] = sat_prox_mat
            sat_prox_mat = padded_sat_prox_mat

        self.sat_prox_mat = sat_prox_mat
        self.n = sat_prox_mat.shape[0]
        self.m = sat_prox_mat.shape[1]
        self.T = sat_prox_mat.shape[2]

        self.init_assignment = init_assignment
        self.curr_assignment = init_assignment
        self.lambda_ = lambda_

        self.k = 0

        #Build benefit_info
        self.benefit_info = BenefitInfo()

        if init_task_vars is None: 
            self.init_task_vars = np.zeros(self.m, dtype=np.float64)
            self.init_task_vars[:self.unpadded_m] = np.ones(self.unpadded_m)
        else:
            self.init_task_vars = np.zeros(self.m, dtype=np.float64)
            self.init_task_vars[:self.unpadded_m] = init_task_vars
        if var_add is None:
            self.benefit_info.var_add = np.zeros(self.m)
            self.benefit_infovar_add[:self.unpadded_m] = 0.01*np.ones(self.unpadded_m)
        else:
            self.benefit_info.var_add = np.zeros(self.m)
            self.benefit_info.var_add[:self.unpadded_m] = var_add
        self.benefit_info.task_vars = np.copy(self.init_task_vars)
        self.benefit_info.base_sensor_var = base_sensor_var

        self.benefit_fn = variance_based_benefit_fn

        self.k = 0

        self.task_var_hist = np.zeros((self.m, self.T))

    def step(self, assignment):
        """
        Returns state, value, and whether the environment is done.
        """
        benefit_hat = self.benefit_fn(self.sat_prox_mat[:,:,self.k], self.curr_assignment, self.lambda_, self.benefit_info)
        value = np.sum(benefit_hat * assignment)

        #Update the variance of the task based on the new measurement
        for j in range(self.unpadded_m):
            if np.max(assignment[:,j]) == 1:
                i = np.argmax(assignment[:,j])
                if self.sat_prox_mat[i,j,self.k] == 0: sensor_var = 1000000 #If the task is not reachable, set the variance to a high value
                else: sensor_var = self.benefit_info.base_sensor_var / self.sat_prox_mat[i,j,self.k]

                #If agent was not previously assigned to this task, multiply the sensor variance by lambda_ (>1)
                if self.curr_assignment is not None:
                    prev_i = np.argmax(self.curr_assignment[:,j])
                    if prev_i != i: sensor_var *= self.lambda_
                
                self.benefit_info.task_vars[j] = 1/(1/self.benefit_info.task_vars[j] + 1/sensor_var)

        #Save the task vars to the history
        self.task_var_hist[:,self.k] = self.benefit_info.task_vars

        self.benefit_info.task_vars += self.benefit_info.var_add

        self.k += 1
        self.curr_assignment = assignment

        return self.get_state(), value, self.k>=self.sat_prox_mat.shape[2]

    def reset(self):
        self.curr_assignment = self.init_assignment
        self.benefit_info.task_vars = np.copy(self.init_task_vars)
        self.task_var_hist = np.zeros((self.m, self.T))
        self.k = 0

    def get_state(self):
        """
        Get a state representation of the environment
        for use in RL
        """
        return None
    
def variance_based_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Create a benefit mat based on the previous assignment and the variances of each area.
    As a handover penalty, the sensor variance of the first measurement of a given task is much higher.

    INPUTS:
     - a 3D (n x m x T) mat of the satellite coverage mat, as well as previous assignments.
     - lambda_ is in this case the amount the sensor covariance is multiplied for the first measurement.
     - benefit_info should be a structure containing the following info:
        - task_vars is the m variances of the tasks at the current state.
        - base_sensor_var is the baseline sensor variance
        - var_add is how much variance is added at each time step.
    """
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    T = sat_prox_mat.shape[2]

    benefit_mat = np.zeros_like(sat_prox_mat)

    #Transform curr_var (m,) to (n,m) by repeating the value across the agent axis (axis 0).
    #This array will track the variance of each task j, assuming that it was done by satellite i.
    agent_task_vars = np.tile(benefit_info.task_vars, (n, 1))
    for k in range(T):
        for i in range(n):
            for j in range(m):
                if agent_task_vars[i,j] == 0 or sat_prox_mat[i,j,k] == 0: #if variance is zero or the sat cant see the task, that means there's no benefit from further measuring it
                    benefit_mat[i,j,k] = 0
                else:
                    #Calculate the sensor variance for sat i measuring task j at time k
                    sensor_var = benefit_info.base_sensor_var / sat_prox_mat[i,j,k]

                    #if the task was not previously assigned, multiply the sensor variance by lambda_ (>1)
                    if prev_assign is not None and prev_assign[i,j] != 1 and k == 0:
                        sensor_var *= lambda_

                    #Calculate the new variance for the task after taking the observation.
                    #The reduction in variance is the benefit for the agent-task pair.
                    new_agent_task_var = 1/(1/agent_task_vars[i,j] + 1/sensor_var)
                    benefit_mat[i,j,k] = agent_task_vars[i,j] - new_agent_task_var

                    #Update the variance of the task based on the new measurement
                    agent_task_vars[i,j] = new_agent_task_var

        #Add variance to all tasks
        agent_task_vars += benefit_info.var_add

    if init_dim == 2: benefit_mat = np.squeeze(benefit_mat, axis=2)
    return benefit_mat