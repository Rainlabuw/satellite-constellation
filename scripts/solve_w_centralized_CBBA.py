import numpy as np

class SequentialCBBAAgent(object):
   def __init__(self, num_timesteps):
      self.bundle = []
      self.bundle_task_path = [None]*num_timesteps

def score_task(i, j, k, benefits, agents, lambda_):
    marginal_benefit = benefits[i,j,k]
    agent = agents[i]

    tasks_done_at_timestep = [ag.bundle_task_path[k] for ag in agents if ag.bundle_task_path[k] is not None]

    #If a task has already been selected for this agent and timestep, don't select again
    if agent.bundle_task_path[k] is not None:
        marginal_benefit = -np.inf
    #If a task has already been selected for this timestep, don't select again
    elif j in tasks_done_at_timestep:
        marginal_benefit = -np.inf
    #Calculate the score of the task based on the bundle
    else:
        if k != 0:
            if agent.bundle_task_path[k-1] is None or agent.bundle_task_path[k-1] == j:
                pass #there is no penalty for switching tasks
            else:
                marginal_benefit -= lambda_

        if k != (len(agent.bundle_task_path) - 1):
            if agent.bundle_task_path[k+1] is None or agent.bundle_task_path[k+1] == j:
                pass
            else:
                marginal_benefit -= lambda_
    return marginal_benefit

def solve_w_centralized_CBBA(benefits, lambda_):
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    agents = [SequentialCBBAAgent(T) for i in range(n)]

    for iter in range(n*T):
        print(f"{iter}/{n*T}", end='\r')
        best_marginal_benefit = -np.inf
        best_i = None
        best_j = None
        best_k = None
        for i in range(n):
            for j in range(m):
                for k in range(T):
                    marginal_benefit = score_task(i, j, k, benefits, agents, lambda_)
                    if marginal_benefit > best_marginal_benefit:
                        best_marginal_benefit = marginal_benefit
                        best_i = i
                        best_j = j
                        best_k = k

        agents[best_i].bundle_task_path[best_k] = best_j
        agents[best_i].bundle.append((best_j, best_k))

    #Convert to assignment matrix form
    assignment_mats = []
    for k in range(T):
        assignment_mat = np.zeros((n,m))
        for i, agent in enumerate(agents):
            j = agent.bundle_task_path[k]
            assignment_mat[i,j] = 1
        assignment_mats.append(assignment_mat)

    return assignment_mats

if __name__ == "__main__":
    benefits = np.random.rand(10, 10, 10)
    b = np.zeros((2,2,2))
    b[:,:,0] = np.array([[10, 1],[1, 10]])
    b[:,:,1] = np.array([[1, 1.5],[1.5, 1]])
    lambda_ = 0
    ams = sequential_greedy(b, lambda_)
    print(ams)