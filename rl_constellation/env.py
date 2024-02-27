from common.methods import *

from rl_constellation.solve_w_rl import build_auction_benefits_from_rl_policy

class ArtificalConstellationEnv(object):
    def __init__(self, n, m, T, lambda_, state_dep_fn=generic_handover_state_dep_fn, extra_handover_info=None) -> None:
        self.n = n
        self.m = m
        self.T = T
        self.lambda_ = lambda_
        self.state_dep_fn = state_dep_fn
        self.extra_handover_info = extra_handover_info
        
        self.benefits = generate_benefits_over_time(n, m, T, 3, 6, 0.25, 2)

        self.k = 0
        self.curr_assignment = None

    def step(self, auction_benefits):
        ass = solve_centralized(auction_benefits)
        ass = convert_central_sol_to_assignment_mat(self.n, self.m, ass)

        #Calculate rewards for each agent
        rewards = []
        for i in range(self.n):
            rw = calc_assign_seq_state_dependent_value(self.curr_assignment[i,:], ass[i,:], self.benefits[i,:,self.k],self.lambda_,
                                                        state_dep_fn=self.state_dep_fn, extra_handover_info=self.extra_handover_info)
            
            rewards.append(rw)

        self.k += 1
        self.curr_assignment = ass

        if self.k == self.T:
            self.reset()
            
        return rewards

    def reset(self):
        self.benefits = generate_benefits_over_time(self.n, self.m, self.T, 3, 6, 0.25, 2)
        
        self.k = 0
        self.curr_assignment = None