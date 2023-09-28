import methods
import numpy as np 
import matplotlib.pyplot as plt
import control as ct
from scipy.linalg import solve_discrete_lyapunov

n = 4
m = 3

A, B = methods.StabilizingGainManifold.rand_controllable_matrix_pair(n, m)
Sigma = np.eye(n)
Q = np.eye(n)
R = np.eye(m)
M = methods.StabilizingGainManifold(A, B, Q, R, Sigma)
K0 = M.rand()
K_opt = M.dlqr()
alpha = .01
error_hist = []
tol = 1e-3
error = np.linalg.norm(K0 - K_opt)**2

K = K0
while error > tol:
    error_hist.append(error)
    K = K - alpha*M.grad_f(K)
    error = np.linalg.norm(K - K_opt)**2
    print(np.round(error,6),alpha)

plt.figure()
plt.semilogy(error_hist)
plt.grid()
plt.show()