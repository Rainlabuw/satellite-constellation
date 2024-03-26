import numpy as np
import torch as th
from common.methods import *

def get_local_and_neighboring_benefits(benefits, i, M):
    """
    Given a benefit matrix n x m x L, gets the local benefits for agent i.
        Filters the benefit matrices to only include the top M tasks for agent i.

    Also returns the neighboring benefits for agent i.
        A (n-1) x M x L matrix with the benefits for every other agent for the top M tasks.

    Finally, returns the global benefits for agent i.
        A (n-1) x (m-M) x L matrix with the benefits for all the other (m-M) tasks for all other agents.
    """
    # ~~~ Get the local benefits for agent i ~~~
    local_benefits = benefits[i, :, :]

    total_local_benefits_by_task = np.sum(local_benefits, axis=-1)
    #find M max indices in total_local_benefits_by_task
    top_local_tasks = np.argsort(-total_local_benefits_by_task)[:M]

    local_benefits = local_benefits[top_local_tasks, :]

    # ~~~ Get the neighboring benefits for agent i ~~~
    neighboring_benefits = np.copy(benefits[:, top_local_tasks, :])
    neighboring_benefits = np.delete(neighboring_benefits, i, axis=0)

    # ~~~ Get the global benefits for agent i ~~~
    global_benefits = np.copy(benefits)
    global_benefits = np.delete(global_benefits, i, axis=0) #remove agent i
    global_benefits = np.delete(global_benefits, top_local_tasks, axis=1) #remove top M tasks

    return top_local_tasks, th.from_numpy(local_benefits).float(), th.from_numpy(neighboring_benefits).float(), th.from_numpy(global_benefits).float()

def solve_randomly(benefits, init_assign, lambda_, state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None):
    """
    Solves the assignment problem with a random assignment.
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    chosen_assignments = []
    for k in range(T):
        random_assignment = np.zeros((n, m))
        random_assignment_vector = np.random.choice(range(m), size=n, replace=False)

        for i, j in enumerate(random_assignment_vector):
            random_assignment[i, j] = 1
        
        chosen_assignments.append(random_assignment)

    total_value = calc_assign_seq_state_dependent_value(init_assign, chosen_assignments, benefits, lambda_, 
                                                        state_dep_fn=state_dep_fn, extra_handover_info=extra_handover_info)

    return chosen_assignments, total_value