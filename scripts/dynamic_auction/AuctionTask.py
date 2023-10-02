import time
import numpy as np

class AuctionTask(object):
    def __init__(self, benefit) -> None:
        #Generate random location for tasks to identify it from other tasks.
        self.loc = np.random.rand()
        
        self.init_time = time.time()

        self.benefit = benefit
        self.agent_cost_mult = 1.0
    
    def __repr__(self) -> str:
        return f"Task location {self.loc} (updated at {self.init_time}), benefit {self.benefit}, agent cost mult {self.agent_cost_mult}"