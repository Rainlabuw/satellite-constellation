from auction import Auction, AuctionAgent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_iters_warm_start = []
    n_iters_from_scratch = []

    benefits = []
    opt_benefits = []

    n_agents = 10
    n_tasks = 10

    for _ in range(100):
        print(f"Running auction {_}",end='\r')
        b = np.random.rand(n_agents,n_tasks)
        new_task = np.random.rand(n_agents,1)

        #R
        orig_a = Auction(n_agents, n_tasks, benefits=b)
        ri, ci = orig_a.solve_centralized()
        orig_a.run_auction()
        
        
        total_b = np.hstack((b, new_task))

        #Rn non warm-started auction
        a = Auction(n_agents, n_tasks+1, benefits=total_b)
        a.run_auction()
        ri, ci = a.solve_centralized()
        opt_benefits.append(total_b[ri, ci].sum())
        n_iters_from_scratch.append(a.n_iterations)

        benefits.append(a.total_benefit_hist)

        #RUn warm started algo
        a = Auction(n_agents, n_tasks+1, benefits=total_b)
        # for i in range(n_agents):
        #     a.agents[i].choice = ci[i]

        for orig_agent, agent in zip(orig_a.agents, a.agents):
            agent.public_prices = np.hstack((orig_agent.public_prices, np.array([0])))
            agent.choice = orig_agent.choice

        a.run_auction()

        n_iters_warm_start.append(a.n_iterations)
    
    print(f"Average number of iterations for warm-started auction: {np.mean(n_iters_warm_start)}")
    print(f"Average number of iterations for non warm-started auction: {np.mean(n_iters_from_scratch)}")

    #Plot a histogram of the number of iterations
    fig, axes = plt.subplots(2,1,sharex=True, sharey=True)
    bins = np.arange(0, max(max(n_iters_warm_start), max(n_iters_from_scratch)), 2.5)
    axes[0].hist(n_iters_from_scratch, bins=bins, label="Non warm-started")
    axes[0].set_title("No warm-start")
    axes[0].set_ylabel("Number of occurences")

    axes[1].hist(n_iters_warm_start, bins=bins, label="Warm-started")
    axes[1].set_title("Warm-start")
    axes[1].set_ylabel("Number of occurences")
    axes[1].set_xlabel("Number of iterations to optimal")
    axes[1].set_xlim((0,100))
    
    plt.show(block=False)

    plt.figure()
    for opt_ben, ben in zip(opt_benefits, benefits):
        plt.plot(np.array(ben)-opt_ben)

    plt.ylabel("Distance to optimality")
    plt.xlabel("Iteration")
    plt.show()