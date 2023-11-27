from methods import *
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def solve_greedily(benefits, init_assignment, lambda_):
    """
    Solve with greedy handover strategy - start with optimal assignments,
    and when a handover is needed (a satellite moved too far away from task)
    just switch to the best assignment.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    csol = solve_centralized(benefits[:,:,0])
    assignment_mat = convert_central_sol_to_assignment_mat(n,m,csol)

    assignment_mats = [assignment_mat]
    for k in range(1,T):
        #check if any satellites are too far away from their task
        prev_assignment_mat = assignment_mats[k-1]
        curr_assignment_mat = prev_assignment_mat.copy()

        #Determine which tasks are now available for reassignment
        #(only assigned tasks with nonzero benefit are NOT available.)
        #(agents with zero benefit ARE available)
        avail_agents = []
        avail_tasks = [j for j in range(m)]
        for i in range(n):
            agent_prev_choice = np.argmax(prev_assignment_mat[i,:])

            if benefits[i,agent_prev_choice,k-1] == 0:
                avail_agents.append(i)
            if benefits[i,agent_prev_choice,k] > 0:
                avail_tasks.remove(agent_prev_choice)

        agents_w_no_avail_task = []
        #Now, reassign agents to the best available task if one exists
        for i in avail_agents:
            agent_prev_choice = np.argmax(prev_assignment_mat[i,:])

            best_task_idx = np.argmax(benefits[i,avail_tasks,k])
            best_task_val = np.max(benefits[i,avail_tasks,k])

            if best_task_val > 0:
                #Reassign the current agent
                curr_assignment_mat[i,agent_prev_choice] = 0
                curr_assignment_mat[i,avail_tasks.pop(best_task_idx)] = 1
            else:
                agents_w_no_avail_task.append(i)

        #Now, reassign agents with no available tasks to the a random task
        for i in agents_w_no_avail_task:
            agent_prev_choice = np.argmax(prev_assignment_mat[i,:])

            best_task_idx = np.argmax(benefits[i,avail_tasks,k])

            #Reassign the current agent
            curr_assignment_mat[i,agent_prev_choice] = 0
            curr_assignment_mat[i,avail_tasks.pop(best_task_idx)] = 1

        assignment_mats.append(curr_assignment_mat)

    total_value, num_handovers = calc_value_and_num_handovers(assignment_mats, benefits, init_assignment, lambda_)
    return assignment_mats, total_value, num_handovers