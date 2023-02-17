from dataclasses import dataclass
from functools import lru_cache
from scipy.linalg import expm
from scipy.signal import lti , lsim
import numpy as np

from .checks import check_rank_matrix, is_diagonal
import pandas as pd 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
@dataclass
class Mdof_system:
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
        return np.linalg.eigvals(self.A)
    def eigenvecs(self):
        return np.linalg.eig(self.A)

    def resonance_freqeuncy(self):
        return np.abs(self.eigenvalues()/2/np.pi)[::2]
        
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

    def simulate_white_noise(self,t,location:int = 7):
        nsamples = len(t)
        scale = np.random.uniform(0.5,30)
        amplitude = 10*np.random.uniform(0.5,3)
        u = np.zeros((nsamples,self.n_dof))
        u[:,location] = amplitude*np.random.normal(0,scale,(nsamples,))
        return u, self.simulate_lsim(t=t,u=u)