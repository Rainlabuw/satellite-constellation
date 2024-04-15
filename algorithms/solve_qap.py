import numpy as np
import tqdm

def qap_objective(qap_benefits, a, b, assigns):
    nT = qap_benefits.shape[0]
    mT = qap_benefits.shape[1]

    total_value = 0
    for i in range(nT):
        for j in range(mT):
            total_value += qap_benefits[i,j] * assigns[i, j]
    
    for i in tqdm.tqdm(range(nT)):
        for j in range(mT):
            if assigns[i,j] == 0:
                continue
            for k in range(nT):
                if a[i,k] == 0:
                    continue
                for l in range(mT):
                    total_value += a[i,k] * b[j,l] * assigns[i,j] * assigns[k,l]

    return total_value

def calc_delta_qap_objective(qap_benefits, a, b, i, j, k=None, l=None):
    nT = qap_benefits.shape[0]
    mT = qap_benefits.shape[1]
    
    if k is not None and l is not None:
        total_value = qap_benefits[i,j] + qap_benefits[k,l]
        total_value += a[i,k] * b[j,l]
        return total_value
    else:
        return qap_benefits[i,j]

def solve_w_qap_heuristic(benefits, lambda_, verbose=False):
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    nT = n * T
    mT = m * T

    #Expand the 3d benefits matrix into a nTxmT matrix with the benefits at each timestep on the diagonal
    qap_benefits = np.zeros((nT, mT))
    a = np.zeros((nT, nT))
    b = np.zeros((mT, mT))

    for i in range(T):
        qap_benefits[i*n:(i+1)*n, i*m:(i+1)*m] = benefits[:,:,i]

        if i < T-1:
            a[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = np.eye(n)
            a[(i+1)*n:(i+2)*n, i*n:(i+1)*n] = np.eye(n)

            b[i*m:(i+1)*m, (i+1)*m:(i+2)*m] = -lambda_/2*(np.ones((m,m)) - np.eye(m))
            b[(i+1)*m:(i+2)*m, i*m:(i+1)*m] = -lambda_/2*(np.ones((m,m)) - np.eye(m))

    #Order 0 to (nT-1) in a random order
    order = np.random.permutation(nT)
    assigns = np.zeros((nT, mT))

    #wrap this loop in tqdm for timing
    for i in tqdm.tqdm(order):
        best_firstassign_value = -np.inf
        best_firstassign = -1

        eqiv_timestep = i // n
        num_assigned_in_this_timestep = np.sum(assigns[:,eqiv_timestep*m:(eqiv_timestep+1)*m])
        if num_assigned_in_this_timestep < m:
            #Iterate through all possible assignments, and choose the best one
            for j in range(m*eqiv_timestep, m*(eqiv_timestep+1)):
                if verbose: print(f"\tChoosing best assignment: {j}/{mT}", end='\r')

                assign_value = calc_delta_qap_objective(qap_benefits, a, b, i, j)
                if assign_value > best_firstassign_value:
                    best_firstassign_value = assign_value
                    best_firstassign = j
        else:
            #Iterate through all possible assignments, and choose the best one
            for j in range(mT):
                if verbose: print(f"\tChoosing best assignment: {j}/{mT}", end='\r')

                assign_value = calc_delta_qap_objective(qap_benefits, a, b, i, j)
                if assign_value > best_firstassign_value:
                    best_firstassign_value = assign_value
                    best_firstassign = j

        #If that assignment already assigned
        if np.sum(assigns[:,best_firstassign]) == 1:
            prev_assigned_agent = np.argmax(assigns[:,best_firstassign])

            assigns[i, best_firstassign] = 1
            assigns[prev_assigned_agent, best_firstassign] = 0

            best_secondassign_value = -np.inf
            best_secondassign = -1
            #Iterate through unassigned tasks, looking for best one
            eqiv_timestep = prev_assigned_agent // n
            num_assigned_in_this_timestep = np.sum(assigns[:,eqiv_timestep*m:(eqiv_timestep+1)*m])
            if num_assigned_in_this_timestep < m:
                for j in range(m*eqiv_timestep, m*(eqiv_timestep+1)):
                    if verbose: print(f"\tChoosing best reassigned assignment: {j}/{mT}", end='\r')
                    if np.sum(assigns[:,j]) == 0:
                        second_assign_value = calc_delta_qap_objective(qap_benefits, a, b, i, best_firstassign, prev_assigned_agent, j)
                    
                        if second_assign_value > best_secondassign_value:
                            best_secondassign_value = second_assign_value
                            best_secondassign = j
            else:
                for j in range(mT):
                    if verbose: print(f"\tChoosing best reassigned assignment: {j}/{mT}", end='\r')
                    if np.sum(assigns[:,j]) == 0:
                        second_assign_value = calc_delta_qap_objective(qap_benefits, a, b, i, best_firstassign, prev_assigned_agent, j)
                    
                        if second_assign_value > best_secondassign_value:
                            best_secondassign_value = second_assign_value
                            best_secondassign = j

            assigns[prev_assigned_agent,best_secondassign] = 1
            if verbose: print(f"\tAssigned {i} to {best_firstassign}, reassigned {prev_assigned_agent} to {best_secondassign} for {best_secondassign_value} value.")
        else:
            assigns[i, best_firstassign] = 1
            if verbose: print(f"\tAssigned {i} to {best_firstassign} for {best_firstassign_value} value.")

    assigns_list = []
    for k in range(T):
        assigns_list.append(assigns[k*n:(k+1)*n,k*m:(k+1)*m])

    return assigns_list, qap_objective(qap_benefits, a, b, assigns)