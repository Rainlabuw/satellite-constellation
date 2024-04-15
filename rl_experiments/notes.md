#### 2/22/24
Initializing this directory.

Wanted to determine smaller scale constellation that would still be fully connected - settled on 10x10 w 4000km link distance after experiments.

Generate 10 sims of 1000 timesteps each for 1,000,000 total training timesteps. (standard haal_experiment1 orbital parameters.)
Saved as data/benefits_list.pkl, etc.

HAAL experiment 1 has an average task in view of 8.67, and experiment 2 has avg. 14.4. Thus, I think M=10 for a constellation of 10x10 should be fine.

Generated value function and policy pairs for just a single of the 10 sims, and it was already 40 Gb, so 10x that would be far too much.
For that reason, switching to on the fly sampling and training - pick random numbers corresponding to datasets and timesteps, generate the pairs right then,
and then train on that minibatch.

Created function for generating batches on the fly^. Am pretraining the vlaue network now. I think I need to reconsider how exactly the policy network
targets are being chosen - seems like we want the network to be predicting the discounted value of each task, not just the current value (after all, thats the whole premise of why this auction would perform well.) In that case, we need to develop a way of figuring out the counterfactual benefits that each agent would recieve from getting to do a given task in the first timestep, and then letting HAAL adjust.

Also: managed to get pytorch working with mls tensors, which has a significant speedup: ~33% in a task that was not only GPU bound. This suggests that it is quite a significant speedup over CPU operations :)

(*IDEA*: when we have these benefit matrices, convolve both in the task direction and in the agent direction. This can hopefully help us capture
things like "how many tasks are available for this agent" and "how many other agents can do this task?")

#### 2/23/24
Pretrained for 2000 batches with a learning rate of 0.005 rather than 0.01. Loss seemed to plateau at 0.025 for both the training and test set, so perhaps this is as good as it will get. Need to work on pretraining the policy network now.

#### 2/24/24
Implemented policy network pretraining, but was extremely slow to the point that it's unworkable, and didn't seem to be improving on the test set (although its probably too early to tell.)
Need to reimplement a threaded version of the policy pretraining, and if thats still not fast enough figure out a heuristic that can get us values which are good enough.

#### 2/25/24
Multithreaded version didnt actually cause a speedup, but a simplification (using the precalculated assignments after a certain point rather than recalculating) did.

Successfully trained a policy network to ~convergence (0.04 loss), but potentially still more room to train before overfitting. Default policy network has 10 filters, 64 hidden units.

Tested this policy on a real benefit matrix, and it was awful, but also significantly better than random assignments, which may actually be worse because that means that there's no bug, it's just an awful policy. Need to investigate slightly more tomorrow and ensure that there is indeed no bug.
RL: -2146, HAAL: 2361, NHA: 1553, Random: -4568

It will also probably be critical to speed up the evaluation of the RL policies, either by multithreading the NN evaluations, putting them on the GPU, or something else, given that this will be running in the environment loop many times.

#### 2/26/24
Decided to switch fully to generating artifical tasks so I can iterate much faster. In order to better generate these benefit matrices, did some experiments on the number of tasks in view at a given time.

Turns out, the tasks in view w/ benefits greater than 0.01 for the real 10x10 constellation case is ~3 at the median

![real_tasks](plots/task_in_view_dist_100sat.png)

Perhaps one issue is that satellites were bidding for far too many tasks? with M=10 and a weakly trained net, it didnt yet know to not bid as high on tasks that were actually just useless?

In any case, generated artificial benefits with this distribution:

![artificial_tasks](plots/task_in_view_dist_artificial.png)