import numpy as np
from tqdm import tqdm
import pickle

from common.methods import *

from haal.solve_w_haal import solve_w_haal

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

def generate_training_data_pairs(benefits_list, assignments_list, M, L, gamma=0.9, lambda_=0.5,
                                 state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None):
    """
    Given a list of benefits and assignments, generates a list of training data pairs.

    L ~ np.ceil(np.log(0.05)/np.log(gamma))
    """
    value_func_training_pairs = []
    policy_func_training_pairs = []
    for iter, (benefits, assigns) in enumerate(zip(benefits_list, assignments_list)):
        n = benefits.shape[0]
        m = benefits.shape[1]
        T = benefits.shape[2]
        print(f"Adding training data for benefit matrix {iter} of {len(benefits_list)}...")
        for k in tqdm(range(T-L)):
            if k == 0:
                prev_assign = np.eye(n,m)
            else:
                prev_assign = assigns[k-1]

            #add handover penalty to the benefits
            handover_adjusted_benefits = np.copy(benefits[:,:,k:k+L]) #if k+L>T, then it will just take the last T-k tasks
            handover_adjusted_benefits[:,:,0] = state_dep_fn(handover_adjusted_benefits[:,:,0], prev_assign, lambda_, extra_handover_info)
            for i in range(n):
                agent_prev_assign = np.expand_dims(prev_assign[i,:],0)

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
        break

    print(f"Saved {len(value_func_training_pairs)} pairs.")

    with open("rl_constellation/data/value_func_training_pairs.pkl", "wb") as f:
        pickle.dump(value_func_training_pairs, f)
        print("Saved value function training pairs.")
    with open("rl_constellation/data/policy_func_training_pairs.pkl", "wb") as f:
        pickle.dump(policy_func_training_pairs, f)
        print("Saved policy function training pairs.")

    return value_func_training_pairs, policy_func_training_pairs

if __name__ == "__main__":
    with open("rl_constellation/data/benefits_list.pkl", "rb") as f:
        benefits_list = pickle.load(f)
        print("Loaded benefits list.")
    with open("rl_constellation/data/assigns_list.pkl", "rb") as f:
        assignments_list = pickle.load(f)
        print("Loaded assignments list.")
    
    L_max = np.ceil(np.log(0.05)/np.log(0.9))
    vftp, pftp = generate_training_data_pairs(benefits_list, assignments_list, 10, L_max)