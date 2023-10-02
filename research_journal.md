

# Journal
## 9/28/23
Josh showed presentations, separated task allocation algorithms into 3 categories: 
1. Distributed
2. Centralized
3. Combination of both

We're interested in the intersection. 

**To-do:**
* We need to develop a simple model and a simple problem that retains relevancy in actual satellite constellations. 

**Ideas:** 
* **Model idea**
	* **Constellation:**
		* A satellite constellation is modeled by a time-varying graph $\mathcal{G}(t)=([N],E(t))$, where $N$ is the number of satellites and $E \subset [N] \times [N]$ is the edge set. 
		* The map from satellite $i$ at time $t$ to its (lat., long., alt.) position is $X(i,t)=(\varphi,\lambda,r)$.
		* We have an additional node $0$, which represents the **ground** or **shared memory** of the satellite constellation. Each satellite has occasional access to the shared memory. This characterizes a satellite briefly passing a ground station to exchange information. The edges from satellites to ground is represented with the edge set $E_0 \subset \{0\} \times [N]$. 
	* **Tasks:**
		* Tasks are indexed by the integers $[M]$. Tasks could be cells (i.e. regions on the Earth) that need to be served or observed by a satellite, or a region in space that needs to be observed (e.g. a satellite in GEO), or anything.
		* The **assignment set** $\mathcal{A}(t) \subset [N] \times [M]$ represent which satellites can execute which tasks. Note that $\mathcal{A}(t)$ is a time-varying bipartite graph. 
		* Every task $j$ has a **price** $p_j(t) \geq 0$ represented by the need for that task to be executed. 
		* For every worker-task pair $(i,j) \in [N] \times [M]$, we have a benefit $\beta_{ij}(t) \in \mathbb{R}$ which represents the benefit of satellite $i$ executing task $j$. Note that $\beta_{ij}(t)<0$ is possible. We assume $(i,j) \not \in \mathcal{A}(t)$ implies $\beta_{ij}(t)=0$. This implies $(\mathcal{A}(t),\beta(t))$ is a  bipartite graph with time-varying weights. 
		* 