
# Journal

## 10/2/23
### Meeting notes

We have a general research direction. Suppose we have a benefit matrix $\beta \in \mathbb{R}^{M \times N}$. We can easily solve the optimal task assignment problem, getting an optimal assignment $\alpha^*(\beta)$. Suppose the benefit matrix changes a bit to $\beta^+$. How can we take advantage of the given $\alpha^*(\beta)$ to compute $\alpha^*(\beta^+)$, particularly in a fast and distributed way? We assume $\beta^+$ is not far from $\beta$. This is useful if $\beta$ is time-varying.

Could knowing $\alpha^*(\beta)$ allow us to solve $$\begin{align}\max_{\alpha} \sum_{i=1}^N \sum_{j=1}^M \alpha_{ij}\beta^+_{ij}-\|\alpha^*(\beta)-\alpha\|_1\end{align}$$ in a faster way? 

How do we use the auction algorithms to handle the possibility of adding/removing satellites or adding/removing tasks?

### Textbook notes
Some notes from Chapter 7 of Bertsekas' **Network Optimization: Continuous and Discrete Models**. These terms will serve as a language to help us communication ideas more effectively. 

We have $N$ satellites and $M$ tasks, and we assume $M \geq N$. The set of tasks assignable to satellite $i$ is denoted $A(i)$. The set of all assignable satellite-task pairs is denoted $\mathcal{A}=\{(i,j):j \in A(i)\}$, which we call the **assignment set**. Remark that $\mathcal{A}$ characterizes a **bipartite graph**.

An **assignment** $S \subset \mathcal{A}$ is a set of assignable satellite-task pairs such that for all $(i,j) \in S$, we have $(i,j') \in S \implies j=j'$ and $(i',j) \in S \implies i = i'$. If $|S|=n$, we say the assignment is **complete**. Otherwise, we say the assignment is **partial**.

If $n=m$, we say our problem is **symmetric**. If $N \leq M$, we say our problem is **asymmetric**. 

Every satellite-task pair $(i,j)$ is given a score $\beta_{ij} \geq 0$ that represents how much **benefit** we get if satellite $i$ executes task $j$. We call $\beta \in \mathbb{R}^{N \times M}$ the **benefit matrix**. Every task $j$ is given a **price** $p_j \geq 0$. We call the vector $p \in \mathbb{R}^M$ the price vector. Given agent $i$, there is set of the "best" tasks equal to $\arg \max_{j \in A(i)}(\beta_{ij}-p_j)$. Elements in this set will be denoted with $j_i \in [M]$. Similarly, given a task $j$, we have a set of the "best" agents for executing that task: $\arg \max_{i \in [N]:j \in A(i)} (\beta_{ij}-p_j)$. Elements of this set are denoted by $i_j \in [N]$. 

Given a (possibly partial) assignment $S$ and price vector $p$, we say $(S,p)$ satisfies $\epsilon$**-complementary slackness** ($\epsilon$-CS) if for every $(i,j) \in S$, task $j$ is within $\epsilon$ of being the "best" task for agent $i$. That is for all $j \in [M]$, we have $$\beta_{ij}-p_j \geq \max_{k \in A(i)} (\beta_{ik} - p_k)-\epsilon$$ In other words, given agent $i$ and the best task $j_i$ for agent $i$, if $(i,j) \in S$, then we must have $(\beta_{ij}-p_j) \geq (\beta_{ij_i}-p_{j_i})-\epsilon$. This holds for each agent $i$. 

$(S,p)$ is not necessarily the optimal assignment, but the idea is that if $(S,p)$ satisfies $\epsilon$-CS and is a complete assignment, then $(S,p)$ is within $epsilon$ "distance" of the true optimal assignment. 

The optimal assignment problem is as follows: $$\begin{align} \max_{\alpha_{ij}} U:=\sum_{i=1}^N \sum_{j=1}^M\beta_{ij}\alpha_{ij} \\ \text{s.t. } \alpha_{ij} \in \{0,1\} \\
\alpha \mathbb{1}=\mathbb{1} \\
\alpha^T \mathbb{1} \leq \mathbb{1}\end{align}$$

The second constraint implies each agent is assigned to a unique task, and no two agents are assigned to the same task. The last constraint implies that no two agents are assigned to the same task, but there could exist unassigned tasks (in the case $M > N$). 

Here, we represent assignments with a matrix $\alpha \in \R^{N \times M}$ instead of a set of assignable satellite-task pairs. 

This problem is easily solved using Kuhn's Hungarian method. It can also be solved via an auction, laid out in section 7.2.2 of Bertsekas. 

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