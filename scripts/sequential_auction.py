import numpy as np
from methods import *
import networkx as nx
import scipy.optimize
from multiprocessing import Pool, cpu_count
from dist_auction_algo_josh import Auction
from scipy import ndimage

class MultiAuction(object):
    def __init__(self, benefits, curr_assignment, graph=None, prices=None, verbose=False):
        # benefit matrix for the next few timesteps
        self.benefits = benefits

        self.n_agents = benefits.shape[0]
        self.n_tasks = benefits.shape[1]
        self.n_timesteps = benefits.shape[2]

        #price matrix for the next few timesteps
        self.prices = prices
        
        self.graph = graph

        # current assignment
        self.curr_assignment = curr_assignment

        self.total_benefit_hist = []

    def init_auctions(self):
        """
        Given benefits from n timesteps, creates n(n+1)/2 auctions.

        i.e. benefits from timesteps 1 and 2 will spawn auctions for
        benefit matrices b_1, b_2, and (b_1+b_2)
        """
        #What we want to do is 
        for convolution_size in range(1, self.benefits.shape[-1]+1):
            conv_kernel = np.ones((self.n_agents, self.n_tasks, convolution_size))
            output = ndimage.convolve(self.benefits, conv_kernel)
            print(convolution_size)
            print(output)

if __name__ == "__main__":
    benefits = np.ones((3,3,3))

    multi_auction = MultiAuction(benefits, None)
    multi_auction.init_auctions()