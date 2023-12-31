# satellite-constellation

## Contents:
* `bertsekas_symmetric_auction_algo.py` is an implementation of Bertsekas' auction algorithm when num_agents = num_tasks
* `spencer_dist_auction.py` is an implementation of Zavalnos' distributed auction algorithm
* `classic_auction.py` is an object oriented implementation of Zavalnos' distributed auction algorithm
* `bertsekas_asymmetric_auction_algo.py` Bertsekas' auction algorithm for asymmetric 
instances.
* `experiments.py` contains all the experiments that have been run over the course of writing the paper
* `constellation_sim/` is a folder containing the orbital mechanics simulation 

## Instructions on getting access to repo.

1. Find a location in your compute for the repo
2. Type `git clone https://github.com/Rainlabuw/satellite-constellation.git`
    This creates a local repo on your computer called `main` and a special "hidden" repo called `origin/main`; the latter tracks the `main` branch on the remote repo called `origin` (in GitHub).
3. Type `git fetch origin main`.
    This command updates the special hidden local `origin/main` branch so it matches the remote `origin main` branch. This is to ensure that your `origin/main` is up to date in case someone recnetly updated the remote `origin main` branch. 
    **Note:** This command will never mess up any work you've done locally. 
4. Type `git merge origin/main main`
    **Warning:** This command can mess up any work you've done locally. This command merges your special hidden `origin/main` branch with your local `main` branch. If there are conflicts, this command will fail and you'll have to resolve them. 

## Starlink simulation steps
### Colab (Easy)
1. Go to this website "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle" to get the latest TLE for Starlink, and replace the old TLE file in the simulation folder called "starlink_tle.txt"
2. Change the path location of TLE file to your google drive location when hinted
3. Run the simulation step by step, should not encounter any package related error

### Local (Hard)
Take time to write but long story short its hard