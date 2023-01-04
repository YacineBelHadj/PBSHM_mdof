
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import vibration_toolbox as vtb

def generate_system_variables(N:int=20, 
    m_mean:np.ndarray = 1e1*np.array([0.5318, 0.4040, 0.4101, 0.4123, 0.3960, 0.3809, 0.4086, 0.3798]),
    k_mean:np.ndarray = 1e3*np.array([1e-6, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70, 56.70]),
    c_mean:np.ndarray = np.array([8.746, 8.791, 8.801, 8.851, 8.714, 8.737, 8.549, 8.752]),
    m_std:float = 0.005, 
    k_std:float=0.5, 
    c_std:float=0.08):

    # Generate N sets of system parameters with random noise
    system_params = dict()
    for i in range(N):
        m_i = np.random.normal(m_mean, m_std)
        k_i = np.random.normal(k_mean, k_std)
        c_i = np.random.normal(c_mean, c_std)
        system_params[f'system_{i}']=np.stack((m_i, k_i, c_i))
    return system_params

def build_system_matrices(system_params: np.ndarray):
    m = system_params[0]
    k = system_params[1]
    c = system_params[2]

    # Number of masses
    n = m.shape[0]

    # Initialize mass matrix
    M = np.diag(m)
    # Initialize stiffness matrix
    K = np.zeros((n, n))
    for i in range(n-1):
        K[i, i] = k[i] + k[i+1]
        K[i, i+1] = K[i+1, i] = -k[i+1]

    K[-1, -1] = k[-1]
    
    # Initialize damping matrix
    C = np.zeros((n, n))
    for i in range(n-1):
        C[i, i] = c[i] + c[i+1]
        C[i, i+1] = C[i+1, i] = -c[i+1]

    C[-1, -1] = c[-1]

    return M, K, C

def check_rank_matrix(system: Tuple[np.ndarray]):
    """
    Checks if the matrix M is full rank.
    """
    for matrix in system:
        assert(np.linalg.matrix_rank(matrix) == matrix.shape[0])

def compute_modes(M: np.ndarray, K: np.ndarray, C: np.ndarray, n_modes: int = 3):
    """
    Computes the first n_modes of a system with mass matrix M, stiffness matrix K and damping matrix C.
    """
    # Solve eigenvalue problem
    omega_temp, phi = sp.linalg.eig(K,M)

    idx = np.argsort(omega_temp)
    # Sort eigenvalues and eigenvectors
    omega = omega_temp[idx]
    phi = phi[:, idx]

    omega = np.sqrt(omega)

    # Compute mode shapes
    return omega, phi



def check_orthogonality(M:np.ndarray,phi: np.ndarray):
    """
    Checks if the mode shapes are orthogonal.
    """
    mu = phi.T@M@phi
    np.allclose(mu-np.diag(np.diagonal(mu),0), np.zeros((M.shape[0], M.shape[0])))

def project_modal(M:np.ndarray, K:np.ndarray, C:np.ndarray, phi:np.ndarray):
    """
    Projects the system matrices onto the modal basis.
    """
    M_modal = phi.T@M@phi
    K_modal = phi.T@K@phi
    C_modal = phi.T@C@phi
    return M_modal, K_modal, C_modal

def is_diagonal(matrix: np.ndarray) -> bool:
    """
    Returns True if the matrix is diagonal, False otherwise.
    """
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if all off-diagonal elements are 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and np.abs(matrix[i, j]) > 1e-6:
                return False
    return True

def sdof_impulse_response(m:float,c:float,k:float,t:np.ndarray):

    xi= c/(2*np.sqrt(m*k))
    w_n = np.sqrt(k/m)
    w_n = np.sqrt(k/m)
    w_d = w_n*np.sqrt(1-xi**2)  
    h=1/(m*w_d)*np.exp(-xi*w_n*t)*np.sin(w_d*t)
    return h

def mdof_impulse_response(M:np.ndarray, K:np.ndarray, C:np.ndarray, phi:np.ndarray, t:np.ndarray):
    """
    Computes the impulse response of a system with mass matrix M, stiffness matrix K, damping matrix C and mode shapes phi.
    """
    M_modal, K_modal, C_modal = project_modal(M, K, C, phi)
    # Check if the matrices are diagonal


    # Initialize impulse response
    h = np.zeros((phi.shape[0], t.shape[0]))

    # Compute impulse response
    for i in range(phi.shape[0]):
        h[i, :] = sdof_impulse_response(M_modal[i, i], C_modal[i, i], K_modal[i, i], t)
    return h

def transfer_function(M:np.ndarray, K:np.ndarray, C:np.ndarray, omega:np.ndarray,i:int,j:int):
    s= 1j*omega
    n_dof= M.shape[0]
    M_s2 = np.einsum('ij,k->ijk', M, s**2)
    C_s = np.einsum('ij,k->ijk', C, s**1)
    K_s = np.einsum('ij,k->ijk', K, s**0)

    # Initialize the array to store the transfer function
    H_ij = []

    # Compute the transfer function for each frequency
    for s_i in range(len(s)):
        H_ij.append(np.linalg.inv((M_s2+C_s+K_s)[:,:,s_i])[i,j])

    return H_ij


def add_anomaly(system_params: dict, location: int, anomaly_type: str, anomaly_size: float = 0.1):
    """
    Adds an anomaly to the system specified in a specific location in the system_params dictionary.
    Available anomaly types: 'stiffness', 'mass', 'damping'
    """
    available_anomaly_types = {'mass':0, 'stiffness':1, 'damping':2}
    if anomaly_type not in available_anomaly_types.keys():
        raise ValueError(f"Invalid anomaly type. Choose from {available_anomaly_types.keys()}.")
    for key,values in system_params.items():
        system_params[key][available_anomaly_types[anomaly_type], location] *= (1-anomaly_size)
    return system_params

def main():
    sys=generate_system_variables() 

    M,K,C=build_system_matrices(sys['system_1'])
    check_rank_matrix((M,C,K))

#    check_orthogonality(M,phi)
#    M_prj, K_prj,C_prj=project_modal(M,K,C,phi)    
#    print(is_diagonal(M_prj))
#    print(is_diagonal(K_prj))
#    print(is_diagonal(C_prj))
    print(pd.DataFrame(M))
    print(pd.DataFrame(K))
    print(pd.DataFrame(C))

    omega, phi= compute_modes(M=M,K=K,C=C)
    print(pd.DataFrame(phi))
    print(pd.DataFrame(omega))

    w = np.linspace(0, 1000, 1000)
    H_3f = transfer_function(M=M, K=K, C=C, omega=w,i=0,j=7)
    fig, axes = plt.subplots(2,1)

    axes[0].semilogy(w,np.abs(H_3f))
    axes[0].grid(which='both', linestyle=':')
    axes[0].set_ylabel('Magnitude')
    axes[1].plot(w,np.angle(H_3f,deg=True))
    axes[1].set_ylabel('Angle (°)')
    axes[1].set_xlabel('Frequency (Hz)')

    axes[1].grid(which='both', linestyle=':')
    plt.show()
    plt.close()
if __name__=='__main__':
    main()