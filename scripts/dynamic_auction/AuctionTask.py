import time
import numpy as np
from copy import deepcopy

class AuctionTask(object):
    def __init__(self, benefit) -> None:
        #Generate random location for tasks to identify it from other tasks.
        self.loc = np.random.rand()
        
        self.update_time = time.time()

        self.benefit = benefit
        self.agent_cost_mult = 1.0

        self.price = 0.0
        self.high_bidder = -1
    
    def copy_task_without_price(self):
        """
        Copies tasks, without the price.
        """
        new_task = deepcopy(self)

        new_task.price = 0.0

        return new_task

    def __repr__(self) -> str:
        return f"Task location {self.loc} (updated at {self.update_time}), benefit {self.benefit}, agent cost mult {self.agent_cost_mult}"