import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#TODO: share the convolution layers between the value and policy networks
class ValueNetwork(nn.Module):
    def __init__(self, L, n, m, M, num_filters, hidden_units):
        super().__init__()

        # Agent-wise convolution: convolve each agent with all of its tasks
        self.local_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, M))
        self.neigh_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, M))
        self.global_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, m-M))
        
        # Task-wise convolution: convolve each task with all of its agents
        self.neigh_task_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(n-1, 1))
        self.global_task_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(n-1, 1))
        
        num_features_combined = num_filters + num_filters*(n-1)*2 + \
                                num_filters*M + num_filters*(m-M)

        # Define the MLP layers
        self.fc1 = nn.Linear(num_features_combined, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)  # Output layer

    def forward(self, local_benefits, neighboring_benefits, global_benefits):
        # all benefits is expected to have shapes of [batch_size, n, M, L]
        local_benefits = local_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]
        neighboring_benefits = neighboring_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]
        global_benefits = global_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]

        # Agent-wise convolution
        local_agent_features = F.relu(self.local_agent_conv(local_benefits))
        local_agent_features = local_agent_features.reshape(local_agent_features.size(0), -1)  # Flatten the features
        neighboring_agent_features = F.relu(self.neigh_agent_conv(neighboring_benefits))
        neighboring_agent_features = neighboring_agent_features.reshape(neighboring_agent_features.size(0), -1)  # Flatten the features
        global_agent_features = F.relu(self.global_agent_conv(global_benefits))
        global_agent_features = global_agent_features.reshape(global_agent_features.size(0), -1)

        # Task-wise convolution
        neighboring_task_features = F.relu(self.neigh_task_conv(neighboring_benefits))
        neighboring_task_features = neighboring_task_features.reshape(neighboring_task_features.size(0), -1)
        global_task_features = F.relu(self.global_task_conv(global_benefits))
        global_task_features = global_task_features.reshape(global_task_features.size(0), -1)
        
        # Combine and flatten features from row and column convolutions
        combined_features = torch.cat((local_agent_features, neighboring_agent_features, global_agent_features,
                                       neighboring_task_features, global_task_features), dim=1)
        
        # MLP processing
        x = F.relu(self.fc1(combined_features))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output between 0 and 1
        
        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, L, n, m, M, num_filters, hidden_units):
        super(PolicyNetwork, self).__init__()

        # Agent-wise convolution: convolve each agent with all of its tasks
        self.local_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, M))
        self.neigh_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, M))
        self.global_agent_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(1, m-M))
        
        # Task-wise convolution: convolve each task with all of its agents
        self.neigh_task_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(n-1, 1))
        self.global_task_conv = nn.Conv2d(in_channels=L, out_channels=num_filters, kernel_size=(n-1, 1))
        
        num_features_combined = num_filters + num_filters*(n-1)*2 + \
                                num_filters*M + num_filters*(m-M)

        # Define the MLP layers
        self.fc1 = nn.Linear(num_features_combined, hidden_units)
        self.fc2 = nn.Linear(hidden_units, M)  # Output layer

    def forward(self, local_benefits, neighboring_benefits, global_benefits):
        # all benefits is expected to have shapes of [batch_size, n, M, L]
        local_benefits = local_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]
        neighboring_benefits = neighboring_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]
        global_benefits = global_benefits.permute(0, 3, 1, 2)  # Permute to [batch_size, L, n, M]

        # Agent-wise convolution
        local_agent_features = F.relu(self.local_agent_conv(local_benefits))
        local_agent_features = local_agent_features.reshape(local_agent_features.size(0), -1)  # Flatten the features
        neighboring_agent_features = F.relu(self.neigh_agent_conv(neighboring_benefits))
        neighboring_agent_features = neighboring_agent_features.reshape(neighboring_agent_features.size(0), -1)  # Flatten the features
        global_agent_features = F.relu(self.global_agent_conv(global_benefits))
        global_agent_features = global_agent_features.reshape(global_agent_features.size(0), -1)

        # Task-wise convolution
        neighboring_task_features = F.relu(self.neigh_task_conv(neighboring_benefits))
        neighboring_task_features = neighboring_task_features.reshape(neighboring_task_features.size(0), -1)
        global_task_features = F.relu(self.global_task_conv(global_benefits))
        global_task_features = global_task_features.reshape(global_task_features.size(0), -1)
        
        # Combine and flatten features from row and column convolutions
        combined_features = torch.cat((local_agent_features, neighboring_agent_features, global_agent_features,
                                       neighboring_task_features, global_task_features), dim=1)
        
        # MLP processing
        x = F.relu(self.fc1(combined_features))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output between 0 and 1
        
        return x


class MultiAgentFCNetwork_SharedParameters(nn.Module):
    def __init__(self,L, n, m, M, num_filters, hidden_units):
        self.network = PolicyNetwork(L, n, m, M, num_filters, hidden_units)

    def forward(self, inputs):
        # A forward pass of the same network in parallel
        futures = [torch.jit.fork(self.network, inp) for inp in inputs]

        results = [torch.jit.wait(fut) for fut in futures]
        return results

if __name__ == "__main__":
    # Test the value network
    L = 10
    n = 100
    m = 350
    M = 10
    num_filters = 10
    hidden_units = 156
    batch_size = 10

    # Create some dummy input data
    local_benefits = torch.rand(batch_size, 1, M, L)
    neighboring_benefits = torch.rand(batch_size, n-1, M, L)
    global_benefits = torch.rand(batch_size, n-1, m-M, L)

    # Create the value network
    value_network = ValueNetwork(L, n, m, M, num_filters, hidden_units)
    # Forward pass
    output = value_network(local_benefits, neighboring_benefits, global_benefits)
    print(output)

    model_parameters = filter(lambda p: p.requires_grad, value_network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    # Create the policy network
    policy_network = PolicyNetwork(L, n, m, M, num_filters, hidden_units)
    # Forward pass
    output = policy_network(local_benefits, neighboring_benefits, global_benefits)
    print(output)

    model_parameters = filter(lambda p: p.requires_grad, policy_network.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)