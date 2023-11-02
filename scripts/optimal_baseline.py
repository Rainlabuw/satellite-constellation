import numpy as np
from methods import *
import itertools
import time

from sequential_auction import MultiAuction, solve_naively

def gen_perms_of_perms(curr_perm_list, n, T):
    global total_perm_list

    if len(curr_perm_list) == T:
        total_perm_list.append(curr_perm_list)
        return
    else:
        for perm in itertools.permutations(range(n)):
            gen_perms_of_perms(curr_perm_list + [perm], n, T)

def find_true_optimal_ass(benefit, init_ass, lambda_):
    n = benefit.shape[0]
    m = benefit.shape[1]
    T = benefit.shape[2]

    global total_perm_list
    total_perm_list = []
    gen_perms_of_perms([], n, T)
    
    best_benefit = -np.inf
    best_assignment = None

    for perm_list in total_perm_list:
        assignment_list = []
        for perm in perm_list:
            ass = np.zeros((n,m))
            for i, j in enumerate(perm):
                ass[i,j] = 1
            assignment_list.append(ass)

        total_benefit = 0
        for j, ass in enumerate(assignment_list):
            total_benefit += (benefit[:,:,j]*ass).sum()

        total_benefit -= calc_handover_penalty(init_ass, assignment_list, lambda_)

        if total_benefit > best_benefit:
            best_benefit = total_benefit
            best_assignment = assignment_list

    return best_benefit, best_assignment

if __name__ == "__main__":
    #Aims to compute a true optimal solution via exhaustive search.

    n = 5
    m = 5
    T = 3

    init_ass = None
    
    lambda_ = 1

    avg_best = 0
    avg_smghl = 0
    avg_smgh = 0
    avg_naive = 0

    num_avgs = 50
    for _ in range(num_avgs):
        benefit = np.random.rand(n,m,T)

        print(f"Run {_}/{num_avgs}", end='\r')

        #SMHGL centralized, lookahead = 2
        multi_auction = MultiAuction(benefit, None, T, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        avg_smghl += ben/num_avgs

        #SMGH
        multi_auction = MultiAuction(benefit, None, 1, lambda_=lambda_)
        multi_auction.run_auctions()
        ben, nh = multi_auction.calc_benefit()
        avg_smgh += ben/num_avgs

        #Naive
        ben, nh = solve_naively(benefit, lambda_)
        avg_naive += ben/num_avgs

        #Optimal
        ben, _ = find_true_optimal_ass(benefit, init_ass, lambda_)
        avg_best += ben/num_avgs

    fig = plt.figure()
    plt.bar(["Naive","SMGH", "SMGHL", "Optimal"], [avg_naive, avg_smgh, avg_smghl, avg_best])
    plt.title(f"Average benefit across {num_avgs} runs, n={n}, m={m}, T={T}")

    plt.show()