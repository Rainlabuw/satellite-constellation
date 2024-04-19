from common.methods import *

from rl_experiments.rl_utils import get_local_and_neighboring_benefits

from algorithms.solve_w_haal import choose_time_interval_sequence_centralized

def solve_w_rl(env, policy_network, M, L, verbose=False):
    n = env.sat_prox_mat.shape[0]
    m = env.sat_prox_mat.shape[1]
    T = env.sat_prox_mat.shape[2]

    done = False
    total_value = 0
    chosen_assignments = []
    while not done:
        if verbose: print(f"Solving with RL, {env.k}/{T}",end='\r')
        tstep_end = min(env.k+L, T)
        prox_mat_window = np.zeros((n,m,L)) #make sure the prox mat is always length L
        prox_mat_window[:,:,:min(L, T-env.k)] = env.sat_prox_mat[:,:,env.k:tstep_end]

        #adjust benefit matrix based on handover
        adj_benefit_mat_window = np.copy(prox_mat_window)
        if env.k > 0:
            adj_benefit_mat_window = env.benefit_fn(prox_mat_window, chosen_assignments[-1], env.lambda_, env.benefit_info)
        else:
            adj_benefit_mat_window = env.benefit_fn(adj_benefit_mat_window, env.init_assignment, env.lambda_, env.benefit_info)

        auction_benefits = build_auction_benefits_from_rl_policy(policy_network, adj_benefit_mat_window, M, L)

        chosen_assignment = solve_centralized(auction_benefits)
        chosen_assignment = convert_central_sol_to_assignment_mat(n, m, chosen_assignment)

        chosen_assignments.append(chosen_assignment)

        _, value, done = env.step(chosen_assignment)
        total_value += value
    
    return chosen_assignments, total_value


def build_auction_benefits_from_rl_policy(policy_network, benefits, M, L):
    """
    Builds a benefit matrix from the RL policy for a given time step.
    """
    n = benefits.shape[0]
    m  = benefits.shape[1]

    auction_benefits = np.zeros((n, m))

    for i in range(n):
        top_local_tasks, local_benefits, neighboring_benefits, global_benefits = get_local_and_neighboring_benefits(benefits, i, M)
    
        #Add batch dimensions to the benefits
        local_benefits = local_benefits.unsqueeze(0).unsqueeze(0) #also add n dimension
        neighboring_benefits = neighboring_benefits.unsqueeze(0)
        global_benefits = global_benefits.unsqueeze(0)

        top_task_benefits = policy_network(local_benefits, neighboring_benefits, global_benefits)
        top_task_benefits = top_task_benefits.squeeze(0).detach().numpy() #convert to numpy array so we can use it in auction_benefits

        auction_benefits[i,top_local_tasks] = top_task_benefits

    return auction_benefits