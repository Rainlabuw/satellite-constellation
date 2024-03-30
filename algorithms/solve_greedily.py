from common.methods import *
import numpy as np

def gen_greedy_assigns(benefits, prev_benefits, curr_assignment_mat):
    """
    Given benefits and a greedy assignment, generate the next greedy assignment.
    """
    if prev_benefits is None: prev_benefits = np.zeros_like(benefits)

    n = benefits.shape[0]
    m = benefits.shape[1]

    next_assignment_mat = curr_assignment_mat.copy()
    # Determine which tasks are now available for reassignment
    # (only previously assigned tasks with nonzero benefit are NOT available.)
    # (agents currently achieving zero benefit ARE available)
    avail_agents = []
    avail_tasks = [j for j in range(m)]
    for i in range(n):
        agent_prev_choice = np.argmax(curr_assignment_mat[i,:])

        #Unless the task has been providing benefit for last timestep and this timestep,
        #it is available for greedy reassignment
        if prev_benefits[i,agent_prev_choice] > 0 and benefits[i,agent_prev_choice] > 0:
            avail_tasks.remove(agent_prev_choice)
        else:
            avail_agents.append(i)

    agents_w_no_avail_task = []
    #Now, reassign agents to the best available task if one exists
    for i in avail_agents:
        agent_prev_choice = np.argmax(curr_assignment_mat[i,:])

        best_task_idx = np.argmax(benefits[i,avail_tasks])
        best_task_val = np.max(benefits[i,avail_tasks])

        #If the best task for the agent has nonzero value, then assign it
        if best_task_val > 0:
            #Reassign the current agent
            next_assignment_mat[i,agent_prev_choice] = 0
            next_assignment_mat[i,avail_tasks.pop(best_task_idx)] = 1
        else:
            agents_w_no_avail_task.append(i)

    agents_to_assign_randomly = []
    #Now, assign agents to the same zero-benefit task if it is still available
    for i in agents_w_no_avail_task:
        agent_prev_choice = np.argmax(curr_assignment_mat[i,:])

        best_task_val = np.max(benefits[i,avail_tasks])
        best_task_choice = None
        #If the agent has no tasks with nonzero benefit, and its old task
        #is still available, don't make any changes
        if best_task_val == 0 and agent_prev_choice in avail_tasks:
            best_task_choice = agent_prev_choice
            avail_tasks.remove(agent_prev_choice)
        elif best_task_val > 0:
            best_task_idx = np.argmax(benefits[i,avail_tasks])
            best_task_choice = avail_tasks.pop(best_task_idx)
        else:
            agents_to_assign_randomly.append(i)
            best_task_choice = agent_prev_choice #make no changes for now

        #Reassign the current agent
        next_assignment_mat[i,agent_prev_choice] = 0
        next_assignment_mat[i,best_task_choice] = 1

    # Finally, reassign agents with no available tasks,
    # and whose previous task is unavailable, to a random task
    for i in agents_to_assign_randomly:
        agent_prev_choice = np.argmax(curr_assignment_mat[i,:])

        best_task_idx = np.argmax(benefits[i,avail_tasks])
        best_task_choice = avail_tasks.pop(best_task_idx)
        
        #Reassign the current agent
        next_assignment_mat[i,agent_prev_choice] = 0
        next_assignment_mat[i,best_task_choice] = 1

    return next_assignment_mat

def solve_greedily(env):
    """
    Solve with greedy handover strategy - start with optimal assignments,
    and when a handover is needed (a satellite moved too far away from task)
    just switch to the best assignment for that satellite, in increasing order
    of satellite task index.

    Some extra infrastructure is included to ensure that satellites don't switch
    between zero-benefit tasks without a good reason - this increases the performance
    of the algorithm in a minimum-handover environment.
    """
    n = env.sat_prox_mat.shape[0]
    m = env.sat_prox_mat.shape[1]
    T = env.sat_prox_mat.shape[2]

    total_value = 0

    benefits = env.benefit_fn(env.sat_prox_mat[:,:,0], env.init_assignment, env.lambda_, env.benefit_info)

    csol = solve_centralized(benefits)
    assignment_mat = convert_central_sol_to_assignment_mat(n,m,csol)

    _, val, done = env.step(assignment_mat)
    total_value += val

    assignment_mats = [assignment_mat]

    prev_assignment = env.init_assignment
    while not done:
        #Calculate the previous and current benefits
        prev_benefits = env.benefit_fn(env.sat_prox_mat[:,:,env.k-1], prev_assignment, env.lambda_, env.benefit_info)
        benefits = env.benefit_fn(env.sat_prox_mat[:,:,env.k], env.curr_assignment, env.lambda_, env.benefit_info)

        curr_assignment_mat = gen_greedy_assigns(benefits, prev_benefits, env.curr_assignment)
        prev_assignment = env.curr_assignment
        
        _, val, done = env.step(curr_assignment_mat)
        total_value += val

        assignment_mats.append(curr_assignment_mat)
    
    return assignment_mats, total_value