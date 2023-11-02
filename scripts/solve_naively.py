import numpy as np
from classic_auction import Auction
from methods import *

def solve_naively(benefit_mats_over_time, lambda_):
    n = benefit_mats_over_time.shape[0]
    m = benefit_mats_over_time.shape[1]
    T = benefit_mats_over_time.shape[2]
    #solve each timestep independently
    assignment_mats = []
    benefits_received = []
    for k in range(T):
        print(k, end='\r')
        a = Auction(n, m, benefits=benefit_mats_over_time[:,:,k], graph=nx.complete_graph(n))
        benefit_received = a.run_auction()

        assignment_mat = convert_agents_to_assignment_matrix(a.agents)
        assignment_mats.append(assignment_mat)

        benefits_received.append(benefit_received)

    return sum(benefits_received)-calc_handover_penalty(None, assignment_mats, lambda_), calc_handover_penalty(None, assignment_mats, lambda_)/lambda_