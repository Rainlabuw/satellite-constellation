# satellite-constellation

## Contents:
* `common/` is a folder containing methods which are used across the project.
* `rl_constellation/` is a folder containing RL implementations
* `haal/` is a folder containing infrastructure and files for HAAL.
* `scripts/` is a folder containing various auction implementations and miscellaneous experiments
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
