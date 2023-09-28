import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import control as ct
import itertools
from typing import Tuple

class StabilizingGainManifold:
    def __init__(
            self,
            A: np.ndarray, 
            B: np.ndarray, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Sigma: np.ndarray
        ) -> None:
        self.A = A # state matrix
        self.B = B # input matrix
        self.Q = Q # state cost matrix
        self.R = R # input cost matrix
        self.n = A.shape[0] # state dim
        self.m = B.shape[1] # input dim
        self.Sigma = Sigma # init positions for cost function computation

    @staticmethod
    def spectral_radius(A: np.ndarray) -> float:
        """Computes the max magnitude eigenvalue of A
        
        A: (ndarray) Arbitrary matrix
        
        Returns: max magnitude eigenvalue of A"""
        evals = np.linalg.eig(A)[0]
        return np.max(np.abs(evals))

    @staticmethod
    def rand_stable_matrix(n: int) -> np.ndarray:
        """Returns a random Schur stable matrix."""
        evals = np.random.rand(n)*(
            2*np.random.randint(2, size=n) - 1
        )
        D = np.diag(evals)
        S = np.random.randn(n,n)
        return S@D@np.linalg.inv(S)

    @staticmethod
    def rand_controllable_matrix_pair(
        n: int, m: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            A = np.random.randn(n, n)
            B = np.random.randn(n, m)
            C = ct.ctrb(A,B)
            if np.linalg.matrix_rank(C) == n:
                return A, B
        
    @staticmethod
    def lyapunov_map(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Discrete Lyapunov map. Returns the unique solution X to 
        X = AXA' + Q.
        
        A: (ndarray(n,n)) Schur stable matrix
        Q: (ndarray(n,n)) arbitrary matrix
        
        Returns: ndarray(n,n)"""
        return solve_discrete_lyapunov(A, Q)
    
    @staticmethod
    def diff_lyapunov_map(
        A: np.ndarray, Q: np.ndarray, E: np.ndarray, F: np.ndarray
    ) -> np.ndarray:
        """Computes differential of lyapunov map along tangent vectors E and F.
        A: (ndarray(n,n)) Schur stable matrix
        Q, E, F: (ndarray(n,n)) arbitrary matrix
        
        Returns: (ndarray(n,n)) diff_lyapunov_map(A,Q)|_{E,F}"""
        return StabilizingGainManifold.lyapunov_map(
            A, E@StabilizingGainManifold.lyapunov_map(A, Q)@A.T + \
            A@StabilizingGainManifold.lyapunov_map(A, Q)@E.T + F
        )
    
    def randvec(self, K: np.ndarray) -> np.ndarray:
        V = np.random.randn(self.m, self.n)
        V = V/self.norm(K, V)
        return V
    
    def zerovec(self) -> np.ndarray:
        return np.zeros((self.m, self.n))
    
    def A_cl(self, K: np.ndarray) -> np.ndarray:
        return self.A - self.B@K
    
    def dlqr(self) -> np.ndarray:
        K, _, _ = ct.dlqr(self.A, self.B, self.Q, self.R)
        return K
    
    def rand(self) -> np.ndarray:
        """Returns a random gain matrix that stablizes (A,B)."""
        evals = np.random.rand(self.n)*(
            2*np.random.randint(2, size=self.n) - 1
        )
        K = ct.place(self.A, self.B, evals)
        return K
    
    def finite_horizon_LQR_cost(
            self, x0: np.ndarray, u: np.ndarray, Qf: np.ndarray, N: int
        ) -> float:
        """Computes finite horizon LQR cost starting at x0. 
        
        x0: ((n,)-vector) init state
        u: ((m,N)-matrix) control inputs
        N: (integer) horizon time
        
        Returns: (scalar) finite horizon LQR cost"""
        out = 0
        xk = x0.copy()
        for k in range(N):
            uk = u[:,k]
            out += 1/2*(xk.T@self.Q@xk + uk.T@self.R@uk)
            xk = self.A@xk + self.B@uk
        out += 1/2*xk.T@Qf@xk
        return out
    
    def E(self, i: int, j: int) -> np.ndarray:
        """Returns (i,j)th entry in the global frame. Returns m-by-n 
        matrix of zeros with (i,j)th entry set to 1.
        
        i: (int) integer between 0 and m - 1
        j: (int) integer between 0 and n - 1
        
        Returns: (ndarray(m,n)) The global frame tangent vector 
        """
        out = self.zerovec()
        out[i,j] = 1
        return out
    
    def P(self, K: np.ndarray) -> np.ndarray:
        """Computes the "P" operator, written in lemma 3.6 of LQR through the Lens 
        of First Order Methods: Discrete-time Case. It solves 
        A_cl'*P*A_cl + K'RK + Q = P for P. Also, if we set u = -Kx, then 
        J(x0, u) = x0'*P*x0. 

        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)

        Returns: (ndarray(n,n)) The unique solution P to 
        A_cl'*P*A_cl + K'RK + Q = P."""
        
        return self.lyapunov_map(self.A_cl(K).T, K.T@self.R@K + self.Q)
    
    def dP(self, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Differential of P at K along tangent vector V"""

        return self.diff_lyapunov_map(
            self.A_cl(K).T, self.Q + K.T@self.R@K,
            -V.T@self.B.T,
            V.T@self.R@K + K.T@self.R@V
        )
        
    def Y(self, K: np.ndarray) -> np.ndarray:
        """Special map from Prop III.3 in Policy Optimization over 
        Submanifolds. Returns L(A - BK, Sigma).
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)
        
        Returns: (ndarray(n,n)) output of L(A - BK, Sigma)
        """
        
        return self.lyapunov_map(self.A_cl(K), self.Sigma)
        
    def dY(self, K: np.ndarray, i1: int, i2: int) -> np.ndarray:
        """Differential of the Y(.) map evaluated on the local frame vector E_i. 
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)
        
        Returns: (ndarray(m,n)) dY(E_i) at K."""
        E_i = self.E(i1, i2) 
        
        return self.diff_lyapunov_map(
            self.A_cl(K), self.Sigma, -self.B@E_i, np.zeros((self.n, self.n))
        )
    
    def f(self, K: np.ndarray) -> float:
        """Computes the finite horizon cost function sum_{1:i:n} J_{Sigma_i}(K).
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)

        Returns: (scalar) The cost."""
        if self.spectral_radius(self.A - self.B@K) >= 1:
            return float("inf")
        return np.trace(self.P(K)@self.Sigma)/2
    
    def df_dE(self, K: np.ndarray, i1: int, i2: int) -> float:
        return np.trace(
            self.dP(K, self.E(i1, i2))@self.Sigma
        )/2
  
    def grad_f(self, K: np.ndarray, Euclidean: bool=False) -> np.ndarray:
        """Gradient of the cost function f at K. 
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)
        Euclidean: (boolean) If true, returns Euclidean gradient. If false, returns
        the Riemannian gradient wrt the Lyapunov Riemannian metric. 

        Returns: (ndarray(m,n)) the gradient of f at K."""
        
        P_K = self.P(K)
        Y_K = self.Y(K)
        if Euclidean:
            return (self.R@K - self.B.T@P_K@self.A_cl(K))@Y_K
        else: 
            return self.R@K - self.B.T@P_K@self.A_cl(K)
 
    def hess_f(
        self, K: np.ndarray, V: np.ndarray, W: np.ndarray, Euclidean: bool=False
    ) -> float:
        A_cl_K = self.A_cl(K)
        grad_f_K = self.grad_f(K, Euclidean=Euclidean)
        S_KV = self.lyapunov_map(A_cl_K.T, V.T@grad_f_K + grad_f_K.T@V)
        S_KW = self.lyapunov_map(A_cl_K.T, W.T@grad_f_K + grad_f_K.T@W)
        H = self.inner(K, self.B.T@S_KW@A_cl_K, V)
        H += self.inner(
            K, 
            (self.R + self.B.T@self.P(K)@self.B)@V + self.B.T@S_KV@A_cl_K,
            W
        )
        if Euclidean:
            return H

        for i1, i2, j1, j2, k1, k2 in itertools.product(
            range(self.m), range(self.n), repeat=3
        ):
            H -= self.inner(
                K, 
                grad_f_K, 
                V[i1,i2]*W[j1,j2]*self.Gamma(K,i1,i2,j1,j2,k1,k2)*self.E(k1,k2)
            )
        return H
        
    def inner(self, K: np.ndarray, V: np.ndarray, W: np.ndarray) -> float:
        """Riemannian metric at K.

        K: (ndarray(m,n)) Stablizing feedback gain to (A,B) 
        V, W: (ndarray(m,n)) Tangent vectors, represented with arbitrary m-by-n 
        matrices 

        Returns: (scalar) <V, W>|_K
        """
        return np.trace(V.T@W@self.Y(K))

    def norm(self, K: np.ndarray, V: np.ndarray) -> float:
        return np.sqrt(self.inner(K, V, V))

    def g(self, K: np.ndarray, i1: int, i2: int, j1: int, j2: int) -> float:
        """Riemannian metric at K evaluated along global frame coordinate
        vectors.

        K: (ndarray(m,n)) Stablizing feedback gain to (A,B) 
        i1, j1: (int) integer beteen 0 and m - 1
        i2, j2: (int) integer between 0 and n - 1

        Returns: (scalar) Evaluation of <E_i1i2, E_j1j2>|_K."""
        if i1 == j1:
            Y_K = self.Y(K)
            return Y_K[j2, i2]
        else:
            return 0

    def inv_g(self, K: np.ndarray, i1: int, i2: int, j1: int, j2: int) -> float:
        """Simply the inverse of g(.). 
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)
        i1, j1: (int) integer beteen 0 and m - 1
        i2, j2: (int) integer between 0 and n - 1
        
        Returns: (scalar) g^ij
        """
        if i1 == j1:
            Y_K_inv = np.linalg.inv(self.Y(K))
            return Y_K_inv[j2, i2]
        else:
            return 0
        
    def dg_dE(
            self,
            K: np.ndarray, 
            i1: int, 
            i2: int, 
            j1: int, 
            j2: int, 
            k1: int,
            k2: int
        ) -> float:
        """Computes the rate of change of <E_i, E_j>|_K when K is shifted 
        along E_k for eps distance. Needed for testing validity of Christoffel 
        symbols.

        K: (ndarray(m,n)) Stablizing feedback gain to (A,B)
        i1,j1,k1: (int) integer beteen 0 and m - 1
        i2,j2,k2: (int) integer between 0 and n - 1

        Returns: (Scalar) d/dt|_{t=0} <E_i, E_j>|_c(t), where c(t) = K + t*E_k.
        """
        return np.trace(
            self.E(i1, i2).T@self.E(j1, j2)@self.diff_lyapunov_map(
                self.A_cl(K), 
                self.Sigma, 
                -self.B@self.E(k1, k2), 
                np.zeros((self.n, self.n))
            )
        )
    
    def Gamma(
            self, 
            K: np.ndarray, 
            i1: int, 
            i2: int,
            j1: int, 
            j2: int, 
            k1: int, 
            k2: int
        ) -> float:
        """Christoffel symbol of Riemannian metric in the local frame at K: 
        Gamma_ij^k at K.

        K: (ndarray(m,n)) Stablizing feedback gain to (A,B) 
        i1, j1, j2: (int) integers between 0 and m - 1 (incl.)
        i2, j2, k2: (int) Integers between 0 and n - 1 (incl.)

        Returns: (scalar) Evaluation of Christoffel symbol Gamma_ij^k at K."""
        gamma = 0
        for l1, l2 in itertools.product(range(self.m), range(self.n)):
            gamma_l1l2 = self.dg_dE(K, j1, j2, l1, l2, i1, i2) + \
                            self.dg_dE(K, i1, i2, l1, l2, j1, j2) - \
                            self.dg_dE(K, i1, i2, j1, j2, l1, l2)
            gamma_l1l2 *= 1/2*self.inv_g(K, k1, k2, l1, l2)
            gamma += gamma_l1l2
        return gamma

    def dGamma_dE(
            self, 
            K: np.ndarray, 
            i1: int, 
            i2: int, 
            j1: int, 
            j2: int,
            k1: int,
            k2: int, 
            l1: int,
            l2: int,
            eps: float=1e-8
        ) -> float:
        """Computes directional derivative of Christoffel symbols along E_l at K.
        
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B) 
        i1, j1, k1, l1: (int) integers between 0 and m - 1 (incl.)
        i2, j2, k2, l2: (int) Integers between 0 and n - 1 (incl.)
        eps: discretization

        Returns: (scalar) dGamma_ij^k/dE_l at K."""
        Gamma_K = self.Gamma(K, i1, i2, j1, j2, k1, k2)
        E_l = self.E(l1, l2)
        Gamma_K_shifted = self.Gamma(K + eps*E_l, i1, i2, j1, j2, k1, k2)
        return (Gamma_K_shifted - Gamma_K)/eps

    def Riemann_curvature_31(
            self, 
            K: np.ndarray, 
            i1: int, 
            i2: int, 
            j1: int, 
            j2: int, 
            k1: int, 
            k2: int,
            l1: int, 
            l2: int
        ) -> float:
        """Computes the coordiantes of the Riemann curvature tensor at K using the local frame co R_ijk^l. 
        K: (ndarray(m,n)) Stablizing feedback gain to (A,B) 
        i1, j1, k1, l1: (int) integers between 0 and m - 1 (incl.)
        i2, j2, k2, l2: (int) Integers between 0 and n - 1 (incl.)

        Returns: (scalar) R_ijk^l at K. 
        """
        R_ijk_l = self.dGamma_dE(K, j1, j2, k1, k2, l1, l2, i1, i2) - \
                    self.dGamma_dE(K, i1, i2, k1, k2, l1, l2, j1, j2)
                    
        for m1, m2 in itertools.product(range(self.m), range(self.n)):
            R_ijk_l += self.Gamma(K, j1, j2, k1, k2, m1, m2)* \
                        self.Gamma(K, i1, i2, m1, m2, l1, l2)
            R_ijk_l -= self.Gamma(K, i1, i2, k1, k2, m1, m2)* \
                        self.Gamma(K, j1, j2, m1, m2, l1, l2)
        return R_ijk_l

    def Riemann_curvature_40(
            self, 
            K: np.ndarray, 
            i1: int, 
            i2: int, 
            j1: int, 
            j2: int, 
            k1: int, 
            k2: int,
            l1: int, 
            l2: int
        ) -> float:
        out = 0
        for m1, m2 in itertools.product(range(self.m), range(self.n)):
            out += self.g(K, l1, l2, m1, m2)* \
            self.Riemann_curvature_31(K, i1, i2, j1, j2, k1, k2, m1, m2)
        return out
    
    def sectional_curvature(
            self,
            K: np.ndarray,
            i1: int,
            i2: int,
            j1: int,
            j2: int
        ) -> float:
        sc = self.Riemann_curvature_40(K, i1, i2, j1, j2, j1, j2, i1, i2)
        Ei_norm2 = self.g(K, i1, i2, i1, i2)
        Ej_norm2 = self.g(K, j1, j2, j1, j2)
        Ei_dot2_Ej = self.g(K, i1, i2, j1, j2)**2
        return sc/(Ei_norm2*Ej_norm2 - Ei_dot2_Ej)
