import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import time
# from ray import tune

from common.methods import *

from rl_experiments.rl_utils import get_local_and_neighboring_benefits
from rl_experiments.networks import ValueNetwork, PolicyNetwork

from algorithms.solve_w_haal import solve_w_haal

class DynamicDataset(Dataset):
    """
    Takes a list of benefits and assignments and generates a dataset of training data for the value and policy networks.
    """
    def __init__(self, benefits_list, assignments_list, M, L, L_max, lambda_=0.5, gamma=0.9, 
                benefit_info=None, calc_policy_outputs=True, calc_value_outputs=True):
        self.benefits_list = benefits_list
        self.assignments_list = assignments_list
        self.M = M
        self.L = L
        self.L_max = L_max
        
        self.lambda_ = lambda_
        self.gamma = gamma

        self.benefit_info = benefit_info

        self.calc_policy_outputs = calc_policy_outputs
        self.calc_value_outputs = calc_value_outputs

        self.length = len(benefits_list) * (benefits_list[0].shape[2]-L_max) * benefits_list[0].shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Selects a random entry in the provided benefit and assignment databases,
        and generates the associated data point on the fly.
        """
        selected_bens_assigns_ind = np.random.choice(len(self.benefits_list))

        #pick a random benefit matrix from the list of sims
        benefits = self.benefits_list[selected_bens_assigns_ind]
        assigns = self.assignments_list[selected_bens_assigns_ind]
        n = benefits.shape[0]
        m = benefits.shape[1]

        #pick a random time step
        k = np.random.randint(benefits.shape[2]-self.L_max)

        #pick a random agent
        i = np.random.randint(n)

        truncated_benefits = benefits[:,:,k:k+self.L_max]
        truncated_assigns = assigns[k:k+self.L_max]
        if k == 0: prev_assign = np.eye(n,m)
        else: prev_assign = assigns[k-1]

        return self.generate_training_data_pair(truncated_benefits, truncated_assigns, prev_assign, i)
    
    def calc_counterfactual_haal_value(self, benefits, assigns, prev_assign, agent, chosen_action):
        """
        Calculates the value that would be returned by HAAL over the next L_max timesteps
        if the chosen action was chosen by the agent.

        NOTE: assumes L=3 for the purposes of HAAL, because that's what was used to generate the data
        """
        adjusted_benefits = np.copy(benefits)
        #make the value for the chosen action very high so it is always chosen
        adjusted_benefits[agent, chosen_action, 0] = 1000

        T = adjusted_benefits.shape[2]

        #Solve for the optimal assignments for the first HAAL_steps timesteps
        HAAL_steps = 6
        haal_ass, _ = solve_w_haal(adjusted_benefits[:,:,:HAAL_steps], prev_assign, self.lambda_, 3)
        #The rest of the assignment can be from the previously calculated assignments
        ass = haal_ass + assigns[HAAL_steps:]

        #Calculate benefits using the unadjusted benefits (so the 1000 doesnt inflate our value)
        agent_benefits = np.expand_dims(np.copy(benefits[agent,:,:]), axis=0)
        agent_assigns = [np.expand_dims(ass[t][agent,:],0) for t in range(T)]
        agent_prev_assign = np.expand_dims(prev_assign[agent,:],0)

        discounted_value = calc_assign_seq_state_dependent_value(agent_prev_assign, agent_assigns, agent_benefits, self.lambda_,
                                    benefit_fn=self.benefit_fn, benefit_info=self.benefit_info, gamma=self.gamma)
        
        #Add back handover penalty if one occured between the HAAL and the old assignments
        if np.argmax(ass[HAAL_steps-1][agent,:]) != np.argmax(ass[HAAL_steps][agent,:]) and np.argmax(assigns[HAAL_steps-1][agent,:]) == np.argmax(assigns[HAAL_steps][agent,:]):
            discounted_value += self.lambda_*self.gamma**(HAAL_steps-1)

        return discounted_value

    def generate_training_data_pair(self, benefits, assigns, prev_assign, i):
        """
        Given a location in the benefit and assignment matrices, 
        computes the training data for the value and policy networks on the fly.

        Expects a benefit tensor and list of assignment matrices (assuming it is already n x m x T)

        Note that while self.L is the lookahead range to give to the neural network, T is L_max,
        or the window at which to stop calculating discounted value (because it is too small to matter).
        """
        normalizing_value = sum([0.75*self.gamma**t for t in range(self.L)])

        n = benefits.shape[0]
        m = benefits.shape[1]
        T = benefits.shape[2]

        #add handover penalty to the benefits
        handover_adjusted_benefits = np.copy(benefits[:,:,:self.L])
        handover_adjusted_benefits[:,:,0] = self.benefit_fn(handover_adjusted_benefits[:,:,0], prev_assign, self.lambda_, self.benefit_info)

        agent_prev_assign = np.expand_dims(prev_assign[i,:],0)

        #~~~~~~~~~~ CALC TRAINING INPUT DATA~~~~~~~~~~~
        top_local_tasks, local_benefits, neighboring_benefits, global_benefits = get_local_and_neighboring_benefits(handover_adjusted_benefits, i, self.M)

        #Compute agent benefit and assignments, maintaining the same shapes
        agent_benefits = np.expand_dims(np.copy(benefits[i,:,:]), axis=0)
        agent_assigns = [np.expand_dims(assigns[t][i,:],0) for t in range(T)]

        #~~~~~~~~~~ CALC VALUE FUNC TRAINING OUTPUT DATA~~~~~~~~~~~
        if self.calc_value_outputs:
            discounted_value = calc_assign_seq_state_dependent_value(agent_prev_assign, agent_assigns, agent_benefits, self.lambda_,
                                        benefit_fn=self.benefit_fn, benefit_info=self.benefit_info, gamma=self.gamma)
            target_value_output = discounted_value/normalizing_value
        else:
            target_value_output = 0

        #~~~~~~~~~~ CALC POLICY TRAINING OUTPUT DATA~~~~~~~~~~~
        target_policy_outputs = np.zeros(self.M)
        if self.calc_policy_outputs:
            for idx, chosen_task in enumerate(top_local_tasks):
                target_policy_outputs[idx] = self.calc_counterfactual_haal_value(benefits, assigns, prev_assign, i, chosen_task)
            target_policy_outputs = target_policy_outputs/normalizing_value

        return local_benefits, neighboring_benefits, global_benefits, target_value_output, target_policy_outputs

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
    batch_size = 256

    training_data_timestep_start = 90
    train_benefits_list = [b[:,:,training_data_timestep_start:] for b in benefits_list]
    train_assigns_list = [a[training_data_timestep_start:] for a in assignments_list]
    train_dataset = DynamicDataset(train_benefits_list, train_assigns_list, M, L, L_max, calc_policy_outputs=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_benefits_list = [b[:,:,:training_data_timestep_start+L_max] for b in benefits_list]
    test_assigns_list = [a[:training_data_timestep_start+L_max] for a in assignments_list]
    test_dataset = DynamicDataset(test_benefits_list, test_assigns_list, M, L, L_max, calc_policy_outputs=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

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
    num_epochs = 5
    max_batches_per_epoch = 100
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader, 0):
            print(f"Epoch {epoch}/{num_epochs}, {i}/{max_batches_per_epoch}")
            #Build batch of inputs to train on
            (local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, target_value_batch, _) = data

            #Forward pass
            local_benefits_batch = local_benefits_batch[:, None, :, :] #Add size 1 dimension for the num agents
            local_benefits_batch = local_benefits_batch.float().to(device)
            neighboring_benefits_batch = neighboring_benefits_batch.float().to(device)
            global_benefits_batch = global_benefits_batch.float().to(device)

            output_value_batch = torch.squeeze(value_network(local_benefits_batch, neighboring_benefits_batch, global_benefits_batch))
            
            #Compute loss compared to target value
            target_value_batch = target_value_batch.float().to(device)
            criterion = torch.nn.MSELoss()
            loss = criterion(output_value_batch, target_value_batch)

            #Backward pass
            value_network.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i > max_batches_per_epoch: break

        test_steps = 0
        test_loss_tot = 0
        for i, data in enumerate(test_dataloader, 0):
            with torch.no_grad():
                (local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, target_value_batch, _) = data
                local_benefits_batch = local_benefits_batch[:, None, :, :] #Add size 1 dimension for the num agents
                local_benefits_batch = local_benefits_batch.float().to(device)
                neighboring_benefits_batch = neighboring_benefits_batch.float().to(device)
                global_benefits_batch = global_benefits_batch.float().to(device)
                
                output_value_batch = torch.squeeze(value_network(local_benefits_batch, neighboring_benefits_batch, global_benefits_batch))

                #Compute loss compared to test value
                target_value_batch = target_value_batch.float().to(device)
                criterion = torch.nn.MSELoss()
                test_loss = criterion(output_value_batch, target_value_batch)
                test_losses.append(test_loss.item())

                test_loss_tot += test_loss.item()
                test_steps += 1
            
            if i > max_batches_per_epoch//4: break
        print(f"TEST LOSS: {test_loss_tot/test_steps}")

    torch.save(value_network.state_dict(), "rl_constellation/networks/value_network_pretrained_new.pt")

    fig, axes = plt.subplots(1,2)
    axes[0].plot(losses)
    axes[1].plot(test_losses)
    plt.show()

def pretrain_policy_network():
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
    train_benefits_list = [b[:,:,training_data_timestep_start:] for b in benefits_list]
    train_assigns_list = [a[training_data_timestep_start:] for a in assignments_list]
    train_dataset = DynamicDataset(train_benefits_list, train_assigns_list, M, L, L_max, calc_value_outputs=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_benefits_list = [b[:,:,:training_data_timestep_start+L_max] for b in benefits_list]
    test_assigns_list = [a[:training_data_timestep_start+L_max] for a in assignments_list]
    test_dataset = DynamicDataset(test_benefits_list, test_assigns_list, M, L, L_max, calc_value_outputs=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    n = benefits_list[0].shape[0]
    m = benefits_list[0].shape[1]
    T = benefits_list[0].shape[2]
    num_filters = 10
    hidden_units = 64
    policy_network = PolicyNetwork(L, n, m, M, num_filters, hidden_units)

    state_dict = torch.load('rl_constellation/networks/policy_network_pretrained.pt')
    policy_network.load_state_dict(state_dict)

    optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.005)
    device = torch.device("mps")
    policy_network.to(device)

    losses = []
    test_losses = []
    max_batches_per_epoch = 64
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}...")
        minibatch_start = time.time()
        for i, data in enumerate(train_dataloader, 0):
            if i >= max_batches_per_epoch: break
            print(f"Epoch {epoch}/{num_epochs}, {i}/{max_batches_per_epoch} in {time.time()-minibatch_start} seconds.")
            #Build batch of inputs to train on
            (local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, _, target_policy_batch) = data

            #Forward pass
            local_benefits_batch = local_benefits_batch[:, None, :, :] #Add size 1 dimension for the num agents
            local_benefits_batch = local_benefits_batch.float().to(device)
            neighboring_benefits_batch = neighboring_benefits_batch.float().to(device)
            global_benefits_batch = global_benefits_batch.float().to(device)

            output_policy_batch = torch.squeeze(policy_network(local_benefits_batch, neighboring_benefits_batch, global_benefits_batch))
            
            #Compute loss compared to target value
            target_policy_batch = target_policy_batch.float().to(device)
            criterion = torch.nn.MSELoss()
            loss = criterion(output_policy_batch, target_policy_batch)

            #Backward pass
            policy_network.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            minibatch_start = time.time()

        test_steps = 0
        test_loss_tot = 0
        for i, data in enumerate(test_dataloader, 0):
            if i >= max_batches_per_epoch/4: break
            with torch.no_grad():
                (local_benefits_batch, neighboring_benefits_batch, global_benefits_batch, _, target_policy_batch) = data
                local_benefits_batch = local_benefits_batch[:, None, :, :] #Add size 1 dimension for the num agents
                local_benefits_batch = local_benefits_batch.float().to(device)
                neighboring_benefits_batch = neighboring_benefits_batch.float().to(device)
                global_benefits_batch = global_benefits_batch.float().to(device)
                
                output_policy_batch = torch.squeeze(policy_network(local_benefits_batch, neighboring_benefits_batch, global_benefits_batch))

                #Compute loss compared to test value
                target_policy_batch = target_policy_batch.float().to(device)
                criterion = torch.nn.MSELoss()
                test_loss = criterion(output_policy_batch, target_policy_batch)

                test_loss_tot += test_loss.item()
                test_steps += 1

        test_losses.append(test_loss_tot/test_steps)
        print(f"TEST LOSS: {test_loss_tot/test_steps}")

    torch.save(policy_network.state_dict(), "rl_constellation/networks/policy_network_pretrained.pt")

    fig, axes = plt.subplots(1,2)
    axes[0].plot(losses)
    axes[1].plot(test_losses)
    plt.show()

if __name__ == "__main__":
    pretrain_policy_network()