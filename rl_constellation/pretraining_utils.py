import numpy as np

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

    return top_local_tasks, local_benefits, neighboring_benefits, global_benefits

def generate_training_data_pairs(benefits_list, assignments_list, M, L, gamma=0.9, lambda_=0.6,
                                 state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None):
    """
    Given a list of benefits and assignments, generates a list of training data pairs.

    L ~ np.ceil(np.log(0.05)/np.log(gamma))
    """
    value_func_training_pairs = []
    policy_func_training_pairs = []
    for benefits, assigns, in zip(benefits_list, assignments_list):
        n = benefits.shape[0]
        m = benefits.shape[1]
        T = benefits.shape[2]

        for k in range(T-L):
            if k == 0:
                prev_assign = np.eye(n,m)
                agent_prev_assign = np.expand_dims(prev_assign[i,:],0)
            else:
                prev_assign = assigns[k-1]
                agent_prev_assign = np.expand_dims(prev_assign[i,:],0)

            #add handover penalty to the benefits
            handover_adjusted_benefits = state_dep_fn(np.copy(benefits[:,:,k:k+L]), prev_assign, lambda_, extra_handover_info)
            for i in range(n):
                #~~~~~~~~~~ CALC TRAINING INPUT DATA~~~~~~~~~~~
                top_local_tasks, local_benefits, neighboring_benefits, global_benefits = get_local_and_neighboring_benefits(handover_adjusted_benefits, i, M)
                inputs = (local_benefits, neighboring_benefits, global_benefits)

                #Compute agent benefit and assignments, maintaining the same shapes
                agent_benefits = np.expand_dims(np.copy(benefits[i,:,k:k+L]), axis=0)
                agent_assigns = [np.expand_dims(assigns[t][i,:],0) for t in range(k,k+L)]

                #~~~~~~~~~~ CALC VALUE FUNC TRAINING OUTPUT DATA~~~~~~~~~~~
                discounted_value = calc_assign_seq_state_dependent_value(agent_prev_assign, agent_assigns, agent_benefits, lambda_,
                                            state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None, gamma=gamma)

                value_func_training_pairs.append((inputs, discounted_value))

                #~~~~~~~~~~ CALC POLICY TRAINING OUTPUT DATA~~~~~~~~~~~
                local_benefits = handover_adjusted_benefits[i, top_local_tasks, 0]
                
                policy_func_training_pairs.append((inputs, local_benefits))

    return value_func_training_pairs, policy_func_training_pairs

if __name__ == "__main__":
    benefits = np.random.rand(10, 10, 10)
    #Change the type of benefits to float16
    benefits = benefits.astype(np.float16)
    local, neigh, globl = get_local_and_neighboring_benefits(benefits, 0, 4)