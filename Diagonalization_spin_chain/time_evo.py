import numpy as np
import diagonalization
from support_functions import unpackbits, packbits
from scipy.constants import hbar, e


def time_evo_sigma_z(t, psi_0, chain_length, J, B0, A, spin_constant,
                     periodic_boundaries, central_spin):
    """
    Computes the time evolution of the spin operator sigma_z for the given state

    Args:
        t (array [tN]): array with tN timesteps
        psi0 (array [N]): the state at t0
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        spin_constant (bool): If true, than the function eig_values_vectors_spin_const is used,
                              eig_values_vectors otherwise.
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin, coupling
                                           to all other spins is used or not

    Returns:
        exp_sig_z (array [tN, N]): array of expectation values of a chain with N sites and tN
                                   timesteps

    """

    total_spins = chain_length + central_spin
    dim = np.int(2**total_spins)
    if spin_constant:
        eigenvalues, eigenvectors = diagonalization.eig_values_vectors_spin_const_all_subspaces(
            chain_length, J, B0, A, periodic_boundaries, central_spin)
    else:
        eigenvalues, eigenvectors = diagonalization.eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)
    psi_z = np.arange(0, dim)
    sigma_z = unpackbits(psi_z, total_spins) - 1/2

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
    propagator = np.exp(-1j * np.outer(t, eigenvalues) / hbar * e * 1e-15)
    # Old
    # psi_t = (eigenvectors @ (propagator[:, :, np.newaxis] * eigenvectors.T)) @ psi_0
    psi_t = (eigenvectors @ (propagator * (eigenvectors.T @ psi_0)).T).T

    return (np.abs(psi_t)**2) @ sigma_z


def time_evo_subspace(times, eigenvalues, eigenvectors, total_spins, initial_state="neel",
                      float_precision=32):
    """
    Computes the time evolution of an inital state

    Args:
        times (float [times])
        eigenvalues (float [dim_sub])
        eigenvectors (float [dim_sub, dim_sub])
        total_spins (int)
        inital (string, default: "neel"): the type of inital state. Possible arguments:
            "neel"
        float_precision (int, default: 32): determines the precision of floats. Possible
            arguments: 32, 64

    Returns:
        psi_t (complex [times, dim_sub]): the state vector at each timestep

    """
    if float_precision == 32:
        # float and complex dtype
        fdtype = np.float32
        cdtype = np.complex64
    elif float_precision == 64:
        fdtype = np.float64
        cdtype = np.complex128
    else:
        raise ValueError()
    dim = int(2**total_spins)
    # This mask filters out the states of the subspace
    subspace_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))[0]
    if initial_state == "neel":
        psi_0 = np.zeros(dim).astype(fdtype)
        psi_0[packbits(np.arange(total_spins) % 2)] = 1
        psi_0 = psi_0[subspace_mask]
    else:
        raise ValueError()
    eigenvectors = eigenvectors.astype(cdtype)

    propagator = np.exp(-1j * np.outer(times, eigenvalues) / hbar * e * 1e-15).astype(cdtype)
    # Old
    # return (eigenvectors @ (propagator[:, :, np.newaxis] * eigenvectors.T)) @ psi_0
    return (eigenvectors @ (propagator * (eigenvectors.T @ psi_0)).T).T
