import numpy as np
from dist_auction_algo_josh import Auction

def run_perturbed_auction(n, m, pert_scale):
    # Create first auction with random benefit matrix
    auction = Auction(n, m, eps=0.01)
    dist_benefit = auction.run_auction()
    _, _, cent_benefit = auction.solve_centralized()

    dist_from_opt = (cent_benefit - dist_benefit)/cent_benefit

    prev_prices = auction.agents[0].public_prices
    perturbed_benefits = auction.benefits + np.random.normal(scale=pert_scale, size=(n, m))
    perturbed_benefits = np.clip(perturbed_benefits, 0, 1)
    p_auction = Auction(n, m, benefits=perturbed_benefits, prices=prev_prices, eps=0.01)
    p_dist_benefit = p_auction.run_auction()
    # p_dist_benefit = p_auction.run_reverse_auction_for_asymmetric()
    _, _, p_cent_benefit = p_auction.solve_centralized()

    p_dist_from_opt = (p_cent_benefit - p_dist_benefit)/p_cent_benefit

    iter_diff_pct = (p_auction.n_iterations - auction.n_iterations) / auction.n_iterations

    dist_from_opt_diff = p_dist_from_opt - dist_from_opt

    if p_dist_from_opt > n*auction.eps:
        n_eps_violation = True
    else:
        n_eps_violation = False

    return dist_from_opt_diff, iter_diff_pct, n_eps_violation

def eval_perturbed_auctions(num_auctions, pert_scale):
    np.random.seed(0)
    num_violations = 0
    total_pct_iter_diff = 0
    total_opt_diff = 0
    for auct_num in range(num_auctions):
        print(f"Running auction {auct_num}/{num_auctions}", end='\r')
        opt_diff, pct_iter_diff, violation = run_perturbed_auction(100, 110, pert_scale)
        if violation: num_violations += 1
        total_pct_iter_diff += pct_iter_diff
        total_opt_diff += opt_diff
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Number of violations of n-epsilon optimality: {num_violations}")
    print(f"Average percent runtime difference: {total_pct_iter_diff/num_auctions}")
    print(f"Average distance from optimality difference: {total_opt_diff/num_auctions}")

if __name__ == "__main__":
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Perturbation of 0.001")
    eval_perturbed_auctions(100,0.005)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Perturbation of 0.01")
    eval_perturbed_auctions(100,0.01)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Perturbation of 0.1")
    eval_perturbed_auctions(100,0.1)