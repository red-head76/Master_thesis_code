import numpy as np
import diagonalization
from support_functions import unpackbits, packbits, calc_idx_psi_0
from scipy.constants import hbar, e

# Arguments explained in output.py


def time_evo_sigma_z(t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                     seed, scaling, initial_state):
    """
    Computes the time evolution of the spin operator sigma_z for the given state

    Returns:
        exp_sig_z (array [tN, N]): array of expectation values of a chain with N sites and tN
                                   timesteps
    """
    total_spins = chain_length + central_spin
    dim = int(2**total_spins)
    idx_psi_0 = calc_idx_psi_0(initial_state, total_spins)
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    n_up = np.sum(unpackbits(idx_psi_0, total_spins))
    subspace_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - n_up))[0]
    psi_0 = psi_0[subspace_mask]
    eigenvalues, eigenvectors = diagonalization.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, n_up, seed, scaling)
    psi_z = np.arange(0, dim)
    sigma_z = (unpackbits(psi_z, total_spins) - 1/2)[subspace_mask]
    propagator = np.exp(-1j * np.outer(t, eigenvalues) / hbar * e * 1e-15)
    psi_t = (eigenvectors @ (propagator * (eigenvectors.T @ psi_0)).T).T

    return (np.abs(psi_t)**2) @ sigma_z


def time_evo(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
             central_spin, seed, scaling, initial_state):
    """
    Computes the time evolution of an inital state, based on given eigenvalues and vectors

    Returns:
        psi_t (complex [tN, dim_sub]): the state vector at each timestep
    """
    total_spins = chain_length + central_spin
    dim = int(2**total_spins)
    idx_psi_0 = calc_idx_psi_0(initial_state, total_spins)
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    n_up = np.sum(unpackbits(idx_psi_0, total_spins))
    subspace_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - n_up))[0]
    psi_0 = psi_0[subspace_mask]
    eigenvalues, eigenvectors = diagonalization.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, n_up, seed, scaling)

    # # non vectorized version. I just keep it for the sake of clarity
    # exp_sig_z = np.empty((len(t), total_spins))
    # # for time step in t
    # for time_i, ts in enumerate(t):
    #     # |psi_t> = U e^(iDt) U+ * |psi0>, with U = eigenvectors, D = diag(eigenvalues)
    #     psi_t = (eigenvectors @ np.diag(np.exp(1j * eigenvalues * ts)) @
    #              eigenvectors.conjugate().T @ psi0)
    #     exp_sig_z[time_i] = (np.inner(sigma_z.T, (np.abs(psi_t)**2)))

    # vectorized notation
    # Compute array with time vector multiplied with diagonal matrix with eigenvalues as entries
    # Factors 1 / hbar * e * 1e-15 transforms from eV to fs
    propagator = np.exp(-1j * np.outer(t, eigenvalues) / hbar * e * 1e-15)
    return (eigenvectors @ (propagator * (eigenvectors.T @ psi_0)).T).T
