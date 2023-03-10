from dataclasses import dataclass
from functools import lru_cache
from scipy.linalg import expm , eig
from scipy.signal import lti , lsim
import numpy as np

from .checks import check_rank_matrix, is_diagonal
import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

import logging
logger = logging.getLogger(__name__)

@dataclass
class MdofSystem:
    M: np.ndarray 
    K: np.ndarray
    C: np.ndarray
    sys_name: str | None = None

    def __post_init__(self):
        self.n_dof = self.M.shape[0]
        self.check_system()


  
    @property
    def A(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof)), np.eye(self.n_dof)],
                    [-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C]])
    
    @property
    def B(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof))],
                     [np.linalg.inv(self.M)]])
    
    @property
    def C_o(self):
        return np.block([[np.eye(self.n_dof), np.zeros((self.n_dof, self.n_dof))],
                    [np.zeros((self.n_dof, self.n_dof)), np.eye(self.n_dof)],
                    [-np.linalg.inv(self.M) @ self.K, -np.linalg.inv(self.M) @ self.C]])
    
    @property
    def D(self):
        return np.block([[np.zeros((self.n_dof, self.n_dof))],
         [np.zeros((self.n_dof, self.n_dof))], 
         [np.eye(self.n_dof)@np.linalg.inv(self.M)]])

    @property
    def A_discrete(self):
        return expm(self.A * self.dt)

    @property
    def B_discrete(self):
        return np.linalg.solve(self.A_discrete - self.A, self.B)

    def __repr__(self):

        name = self.sys_name
        return ('name = {} \n\n'
                'Mass Matrix: \n'
                '{} \n\n'
                'Stiffness Matrix: \n'
                '{} \n\n'
                'Damping Matrix: \n'
                '{}'.format(name,pd.DataFrame(self.M), pd.DataFrame(self.K), pd.DataFrame(self.C)))

    def eigenvalues(self):
        return np.linalg.eigvals(self.A)[::2]

    def eigenvecs(self):
        return np.linalg.eig(self.A)[1][:self.n_dof,::2]

    def resonance_frequency(self):
        return np.abs(self.eigenvalues()/2/np.pi)

    def phi(self):
        phi = polyeig(self.K,self.C,self.M)[0]
        if not np.all(np.isclose(np.imag(phi),0,atol=1e-2)):
            logging.warning('Warning: imaginary part of phi is not zero')
        return np.real(phi)

    def phi_no_damping(self):
        phi = eig(self.K,self.M)[1]
        if not np.all(np.isclose(np.imag(phi),0,atol=1e-2)):
            logging.warning('Warning: imaginary part of phi is not zero')
        return np.real(phi)

    def damping_ratio(self):
        poles = self.eigenvalues()
        w_n = np.abs(poles)
        zeta = -np.real(poles)/w_n
        return zeta

    def rayleigh_damping_coef(self):
        zeta = self.damping_ratio()
        zeta = zeta[:2]
        w_n = np.abs(self.eigenvalues())
        
        Left = np.array([[1/w_n[0],w_n[0]],
                            [1/w_n[1],w_n[1]]])
        Right = np.array([[zeta[0]],[zeta[1]]])
        coef = np.linalg.solve(Left,2*Right)
        return coef

    
    def update_damping(self, method: str ='rayleigh',inplace: bool = True):
        if method == 'rayleigh':
            coef = self.rayleigh_damping_coef()
            C_rayleigh = coef[0]*self.M + coef[1]*self.K
            if inplace:
                self.C = C_rayleigh
            else:
                return C_rayleigh
        else:
            raise ValueError('Unknown damping method')



    def check_system(self):
        assert is_diagonal(self.M), "M is not diagonal"
        assert check_rank_matrix((self.M, self.K, self.C)), "M or K is not full rank"

    
    def transfer_function(self,omega:np.ndarray,i:int,j:int):
        s= 1j*omega
        M_s2 = np.einsum('ij,k->ijk', self.M, s**2)
        C_s = np.einsum('ij,k->ijk', self.C, s**1)
        K_s = np.einsum('ij,k->ijk', self.K, s**0)

        # Initialize the array to store the transfer function
        H_ij = []

        # Compute the transfer function for each frequency
        for s_i in range(len(s)):
            H_ij.append(np.linalg.inv((M_s2+C_s+K_s)[:,:,s_i])[i,j])

        return H_ij
    
    def project_modal(self, x):
        v = self.phi_no_damping()
        return v.T@x@v
    
    def modal_matrices(self):
        
        coef = self.rayleigh_damping_coef()
        omega_n = np.abs(self.eigenvalues())
        m_modal = self.project_modal(self.M)
        k_modal = self.project_modal(self.K)
        c_modal = self.project_modal(self.C)
        xi_modal = coef[0]/2*1/omega_n + coef[1]/2*omega_n

        return m_modal,k_modal,c_modal,xi_modal

        

          
        
        

    def simulate_homemade(self, t, u, x0=None):
        nsamples = len(t)
        noutputs = self.C_o.shape[0]
        x = np.zeros((nsamples, self.n_dof))
        y = np.zeros((nsamples, noutputs))

        x[0,:] = x0 if x0 is not None else np.zeros(self.n_dof)
        for i in range(1,nsamples):
            x[i,:] = self.A_discrete.dot(x[i-1,:]) + self.B_discrete.dot(u[i,:])
            y[i,:] = self.C_o.dot(x[i,:]) + self.D.dot(u[i,:])
        return y

    def simulate_lsim(self, u:np.ndarray, t:np.ndarray, x0=None): 
        sys = lti(self.A, self.B, self.C_o, self.D)
        return lsim(sys, u, t)

    def simulate_white_noise(self,t:np.ndarray,location:int = 7,amplitude:float = 300):
            nsamples = len(t)
            u = np.zeros((nsamples,self.n_dof))
            u[:,location] = np.random.normal(0,amplitude,(nsamples,))
            return u, self.simulate_lsim(t=t,u=u)


import numpy as np
from scipy import linalg

import numpy as np
from scipy import linalg

def polyeig(*A):
    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0???

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)


    """
    if len(A)<=0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes');

    n = A[0].shape[0]
    l = len(A)-1 
    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])
    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]          ]
        ]);
    # Solve generalized eigenvalue problem
    e, X = linalg.eig(C, D);
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    # Sort eigenvalues/vectors
    I = np.argsort(e)
    X = X[:,I][:,::2]
    e = e[I][::2]

    # Scaling each mode by max
    X 

    return X, e
