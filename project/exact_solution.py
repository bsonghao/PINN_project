import numpy as np
import torch
from scipy.special import roots_hermite

def exact_solution(inputs, nu=0.01/np.pi, N=100):
    """exact solution for Burgers's equation"""
    T = inputs[:,0].detach().numpy()
    X = inputs[:,1].detach().numpy()

    eta_base, w = roots_hermite(N)                  # (N,), (N,)

    # Scale eta per flattened (t, x) pair  →  shape: (N_t*N_x, N)
    eta = eta_base[None, :] * 2 * np.sqrt(nu * T[:, None])

    # (x - eta)  →  shape: (N_t*N_x, N)
    arg = X[:, None] - eta                     # (N_t*N_x, N)

    # Cole-Hopf kernel
    f_val   = np.exp(-np.cos(np.pi * arg) / (2 * np.pi * nu))  # (N_t*N_x, N)
    fun_num = np.sin(np.pi * arg) * f_val                         # (N_t*N_x, N)

    # Weighted quadrature sum along N axis  →  (N_t*N_x,)
    numerator   = np.sum(w[None, :] * fun_num, axis=-1)
    denominator = np.sum(w[None, :] * f_val,   axis=-1)

    u_exact = -numerator / denominator               # (N_t*N_x,)

    return torch.from_numpy(u_exact)
