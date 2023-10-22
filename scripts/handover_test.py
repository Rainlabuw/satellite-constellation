import numpy as np
from dist_auction_algo_josh import Auction
from methods import *
from sequential_greedy import sequential_greedy
import networkx as nx
from matplotlib import pyplot as plt

def generate_benefits_over_time(n, m, t_final, num_tsteps):
    benefits = np.zeros((n,m,num_tsteps))
    for i in range(n):
        for j in range(m):
            #where is the benefit curve maximized
            time_center = np.random.uniform(0, t_final)

            #how wide is the benefit curve
            time_spread = np.random.uniform(0, t_final/2)

            #how high is the benefit curve
            benefit_scale = np.random.uniform(1, 2)

            #iterate from time zero to t_final with 100 steps in between
            for t_index, t in enumerate(np.linspace(0, t_final, num_tsteps)):
                #calculate the benefit at time t
                benefits[i,j,t_index] = benefit_scale*np.exp(-(t-time_center)**2/time_spread**2)
    return benefits

def calc_handover_penalty(assignments, lambda_):
    handover_pen = 0
    for i in range(len(assignments)-1):
        new_assign = assignments[i+1]
        old_assign = assignments[i]

        assign_diff = new_assign - old_assign
        handover_pen += -lambda_*np.sum(assign_diff**2)

    return handover_pen

def convert_agents_to_assignment_matrix(agents):
    assignment_matrix = np.zeros((len(agents), len(agents[0].benefits)))
    for i, agent in enumerate(agents):
        assignment_matrix[i, agent.choice] = 1
    return assignment_matrix

def add_handover_pen_to_benefit_matrix(benefits, prev_assign, lambda_):
    adjusted_benefits = np.where(prev_assign == 1, benefits, benefits - lambda_)
    return adjusted_benefits

if __name__ == "__main__":
    ns = [10, 25, 50, 100]
    # ns = [10, 15, 20]
    t_final = 50
    num_tsteps = 25

    naive_benefits = []
    naive_handover_benefits = []

    sequential_benefits = []
    sequential_handover_benefits = []

    sga_benefits = []
    sga_handover_benefits = []
    for n in ns:
        print(f"AGENT {n}")
        m = n
        lambda_ = 1
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        # np.random.seed(29)
        print(f"Seed {seed}")
        print(f"n: {n}, m: {m}, T: {num_tsteps}, lambda: {lambda_}")
        graph = nx.complete_graph(n)
        benefit_mats_over_time = generate_benefits_over_time(n, m, t_final, num_tsteps)
        #Add 2 lambda_+eps to the benefit matrix to ensure that it's always positive to complete
        #a task.
        # benefit_mats_over_time += 2*lambda_ + 0.01

        #solve each timestep independently
        assignment_mats = []
        benefits = []
        for k in range(num_tsteps):
            print(k, end='\r')
            a = Auction(n, m, benefits=benefit_mats_over_time[:,:,k], graph=graph)
            benefit = a.run_auction()

            assignment_mat = convert_agents_to_assignment_matrix(a.agents)
            assignment_mats.append(assignment_mat)

            benefits.append(benefit)
        
        handover_ben = sum(benefits) + calc_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep independently")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        naive_benefits.append(sum(benefits)/(n*num_tsteps))
        naive_handover_benefits.append(handover_ben/(n*num_tsteps))

        #solve each timestep sequentially
        assignment_mats = []
        benefits = []
        
        #solve first timestep separately
        a = Auction(n, m, benefits=benefit_mats_over_time[:,:,0], graph=graph)
        benefit = a.run_auction()

        assignment_mat = convert_agents_to_assignment_matrix(a.agents)
        assignment_mats.append(assignment_mat)
        benefits.append(benefit)

        prev_assignment_mat = assignment_mat
        for k in range(1, num_tsteps):
            print(k, end='\r')
            #Generate assignment for the task minimizing handover
            benefit_mat_w_handover = add_handover_pen_to_benefit_matrix(benefit_mats_over_time[:,:,k], prev_assignment_mat, lambda_)

            a = Auction(n, m, benefits=benefit_mat_w_handover, graph=graph)
            a.run_auction()
            choices = [ag.choice for ag in a.agents]

            assignment_mat = convert_agents_to_assignment_matrix(a.agents)
            assignment_mats.append(assignment_mat)

            prev_assignment_mat = assignment_mat

            #Calculate the benefits from a task with the normal benefit matrix
            benefit = benefit_mats_over_time[:,:,k]*assignment_mat

            benefits.append(benefit.sum())

        handover_ben = sum(benefits) + calc_handover_penalty(assignment_mats, lambda_)
        print("Solving sequentially, each timestep considering the last one")
        print(f"\tBenefit without considering handover: {sum(benefits)}")
        print(f"\tBenefit with handover penalty: {handover_ben}")

        sequential_benefits.append(sum(benefits)/(n*num_tsteps))
        sequential_handover_benefits.append(handover_ben/(n*num_tsteps))

        #solve each timestep sequentially with greedy
        sg_assignment_mats = sequential_greedy(benefit_mats_over_time, lambda_)
        sg_benefit = 0
        for k, sg_assignment_mat in enumerate(sg_assignment_mats):
            sg_benefit += (benefit_mats_over_time[:,:,k]*sg_assignment_mat).sum()

        handover_ben = sg_benefit + calc_handover_penalty(sg_assignment_mats, lambda_)

        print("Solving with greedy algorithm")
        print(f"\tBenefit without considering handover: {sg_benefit}")
        print(f"\tBenefit with handover penalty: {handover_ben}")
    
        sga_benefits.append(sg_benefit/(n*num_tsteps))
        sga_handover_benefits.append(handover_ben/(n*num_tsteps))

    print("done")
    print(naive_benefits)
    print(naive_handover_benefits)
    print(sequential_benefits)
    print(sequential_handover_benefits)
    print(sga_benefits)
    print(sga_handover_benefits)


    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Top subplot
    axs[0].set_title('Benefits without handover penalty')
    axs[0].set_xlabel('Number of agents')
    axs[0].set_ylabel('Total benefit')
    axs[0].bar(np.arange(len(naive_benefits)), naive_benefits, width=0.2, label='Naive')
    axs[0].bar(np.arange(len(sga_benefits))+0.2, sga_benefits, width=0.2, label='SGA')
    axs[0].bar(np.arange(len(sequential_benefits))+0.4, sequential_benefits, width=0.2, label='SMGH (Ours)')
    axs[0].set_xticks(np.arange(len(naive_benefits)))
    axs[0].set_xticklabels([str(n) for n in ns])
    axs[0].legend(loc='lower center')

    # Bottom subplot
    axs[1].set_title('Benefits with handover penalty')
    axs[1].set_xlabel('Number of agents')
    axs[1].set_ylabel('Total benefit')
    axs[1].bar(np.arange(len(naive_handover_benefits)), naive_handover_benefits, width=0.2, label='Naive')
    axs[1].bar(np.arange(len(sga_handover_benefits))+0.2, sga_handover_benefits, width=0.2, label='SGA')
    axs[1].bar(np.arange(len(sequential_handover_benefits))+0.4, sequential_handover_benefits, width=0.2, label='SMGH (Ours)')
    axs[1].set_xticks(np.arange(len(naive_handover_benefits)))
    axs[1].set_xticklabels([str(n) for n in ns])
    #add a legend to the bottom middle of the subplot
    axs[1].legend(loc='lower center')

    plt.show()

