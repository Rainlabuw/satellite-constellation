from common.methods import *

from rl_experiments.rl_utils import get_local_and_neighboring_benefits

from algorithms.solve_w_haal import choose_time_interval_sequence_centralized

def solve_w_rl(benefits, init_assign, lambda_, policy_network, M, L, verbose=False, benefit_info=None):
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    chosen_assignments = []
    for k in range(T):
        if verbose: print(f"Solving with RL, {k}/{T}",end='\r')

        #Build benefit matrix window
        tsteps_remaining = min(L, T-k)
        benefit_mat_window = np.zeros((n, m, L))
        benefit_mat_window[:,:,:tsteps_remaining] = benefits[:,:,k:k+tsteps_remaining]

        #adjust benefit matrix based on handover
        adj_benefit_mat_window = np.copy(benefit_mat_window)
        if k > 0:
            adj_benefit_mat_window[:,:,0] = benefit_fn(adj_benefit_mat_window[:,:,0], chosen_assignments[-1], lambda_, benefit_info)
        elif init_assign is not None:
            adj_benefit_mat_window[:,:,0] = benefit_fn(adj_benefit_mat_window[:,:,0], init_assign, lambda_, benefit_info)
        else: #if it's the first time step and no initial assignment is given, just leave it alone
            pass

        auction_benefits = build_auction_benefits_from_rl_policy(policy_network, adj_benefit_mat_window, M, L)

        chosen_assignment = solve_centralized(auction_benefits)
        chosen_assignment = convert_central_sol_to_assignment_mat(n, m, chosen_assignment)

        chosen_assignments.append(chosen_assignment)

    total_value = calc_assign_seq_state_dependent_value(init_assign, chosen_assignments, benefits, lambda_, 
                                                        benefit_fn=benefit_fn, benefit_info=benefit_info)
    
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