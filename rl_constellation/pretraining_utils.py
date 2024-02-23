import numpy as np
from tqdm import tqdm
import pickle
import torch
import time

from common.methods import *

from rl_constellation.networks import ValueNetwork, PolicyNetwork

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

def generate_training_data_pair(benefits, assigns, prev_assign, i, M, L, lambda_=0.5, gamma=0.9,
                                state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None):
    """
    Given a benefit tensor and list of assignment matrices (assuming it is already n x m x T)

    Note that L is the lookahead range to give to the neural network, while T is L_max,
    or the window at which to stop calculating discounted value (because it is too small to matter).
    """
    normalizing_value = sum([1*gamma**t for t in range(L)])

    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    #add handover penalty to the benefits
    handover_adjusted_benefits = np.copy(benefits[:,:,:L])
    handover_adjusted_benefits[:,:,0] = state_dep_fn(handover_adjusted_benefits[:,:,0], prev_assign, lambda_, extra_handover_info)

    agent_prev_assign = np.expand_dims(prev_assign[i,:],0)

    #~~~~~~~~~~ CALC TRAINING INPUT DATA~~~~~~~~~~~
    top_local_tasks, local_benefits, neighboring_benefits, global_benefits = get_local_and_neighboring_benefits(handover_adjusted_benefits, i, M)

    #Compute agent benefit and assignments, maintaining the same shapes
    agent_benefits = np.expand_dims(np.copy(benefits[i,:,:]), axis=0)
    agent_assigns = [np.expand_dims(assigns[t][i,:],0) for t in range(T)]

    #~~~~~~~~~~ CALC VALUE FUNC TRAINING OUTPUT DATA~~~~~~~~~~~
    discounted_value = calc_assign_seq_state_dependent_value(agent_prev_assign, agent_assigns, agent_benefits, lambda_,
                                state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None, gamma=gamma)
    target_value_output = discounted_value/normalizing_value

    #~~~~~~~~~~ CALC POLICY TRAINING OUTPUT DATA~~~~~~~~~~~
    # TODO: update policy training data to use the counterfactual HAAL benefits
    unnormalized_policy_outputs = handover_adjusted_benefits[i, top_local_tasks, 0]
    target_policy_outputs = unnormalized_policy_outputs/normalizing_value

    return local_benefits, neighboring_benefits, global_benefits, target_value_output, target_policy_outputs

def build_batch_of_training_data(k_range, batch_size, benefits_list, assignments_list, M, L, L_max):
    n = benefits_list[0].shape[0]
    m = benefits_list[0].shape[1]
    T = benefits_list[0].shape[2]
    
    #Build batch of inputs to train on
    local_benefits_batch = np.zeros((batch_size, 1, M, L))
    neighboring_benefits_batch = np.zeros((batch_size, n-1, M, L))
    global_benefits_batch = np.zeros((batch_size, n-1, m-M, L))

    target_value_batch = np.zeros((batch_size,1))
    for batch_ind in range(batch_size):
        selected_bens_assigns_ind = np.random.choice(len(benefits_list))

        #pick a random benefit matrix from the list of sims
        benefits = benefits_list[selected_bens_assigns_ind]
        assigns = assignments_list[selected_bens_assigns_ind]

        #pick a random time step
        k = np.random.randint(k_range[0], k_range[1])

        #pick a random agent
        i = np.random.randint(n)

        truncated_benefits = benefits[:,:,k:k+L_max]
        truncated_assigns = assigns[k:k+L_max]
        if k == 0: prev_assign = np.eye(n,m)
        else: prev_assign = assigns[k-1]

        local_benefits, neighboring_benefits, global_benefits, target_value_output, target_policy_outputs = \
            generate_training_data_pair(truncated_benefits, truncated_assigns, prev_assign, i, M, L)
        
        local_benefits_batch[batch_ind, 0, :, :] = local_benefits
        neighboring_benefits_batch[batch_ind, :, :, :] = neighboring_benefits
        global_benefits_batch[batch_ind, :, :, :] = global_benefits

        target_value_batch[batch_ind,0] = target_value_output

    return local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, target_value_batch

def pretrain_value_network():
    with open("rl_constellation/data/benefits_list.pkl", "rb") as f:
        benefits_list = pickle.load(f)
        print("Loaded benefits list.")
    with open("rl_constellation/data/assigns_list.pkl", "rb") as f:
        assignments_list = pickle.load(f)
        print("Loaded assignments list.")

    L_max = int(np.ceil(np.log(0.05)/np.log(0.9)))
    L = 10
    M = 10
    batch_size = 64

    training_data_timestep_start = 90

    n = benefits_list[0].shape[0]
    m = benefits_list[0].shape[1]
    T = benefits_list[0].shape[2]
    num_filters = 10
    hidden_units = 64
    value_network = ValueNetwork(L, n, m, M, num_filters, hidden_units)

    optimizer = torch.optim.SGD(value_network.parameters(), lr=0.005)
    device = torch.device("mps")
    value_network.to(device)

    losses = []
    test_losses = []
    for _ in tqdm(range(2000)):
        #Build batch of inputs to train on
        training_timestep_range = (training_data_timestep_start, T-L_max)
        local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, target_value_batch = \
            build_batch_of_training_data(training_timestep_range, batch_size, benefits_list, assignments_list, M, L, L_max)

        #Forward pass
        local_benefits_batch = torch.tensor(local_benefits_batch, dtype=torch.float32).to(device)
        neighboring_benefits_batch = torch.tensor(neighboring_benefits_batch, dtype=torch.float32).to(device)
        global_benefits_batch = torch.tensor(global_benefits_batch, dtype=torch.float32).to(device)

        output_value_batch = value_network(local_benefits_batch, neighboring_benefits_batch, global_benefits_batch)
        
        #Compute loss compared to target value
        target_value_batch = torch.tensor(target_value_batch, dtype=torch.float32).to(device)
        criterion = torch.nn.MSELoss()
        loss = criterion(output_value_batch, target_value_batch)

        # print(f"Loss: {loss} in {time.time()-start} seconds")

        #Backward pass
        value_network.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        #test on test set
        if _ % 100 == 0:
            test_timestep_range = (0, training_data_timestep_start)
            local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, target_value_batch = \
                build_batch_of_training_data(test_timestep_range, batch_size, benefits_list, assignments_list, M, L, L_max)

            #Forward pass
            local_benefits_batch = torch.tensor(local_benefits_batch, dtype=torch.float32).to(device)
            neighboring_benefits_batch = torch.tensor(neighboring_benefits_batch, dtype=torch.float32).to(device)
            global_benefits_batch = torch.tensor(global_benefits_batch, dtype=torch.float32).to(device)
            
            #Compute loss compared to test value
            target_value_batch = torch.tensor(target_value_batch, dtype=torch.float32).to(device)
            criterion = torch.nn.MSELoss()
            test_loss = criterion(output_value_batch, target_value_batch)
            test_losses.append(test_loss.item())

    torch.save(value_network.state_dict(), "rl_constellation/networks/value_network_pretrained.pt")

    fig, axes = plt.subplots(1,2)
    axes[0].plot(losses)
    axes[1].plot(test_losses)
    plt.show()

if __name__ == "__main__":
    pretrain_value_network()