import numpy as np
import diagonalization
from support_functions import unpackbits


def time_evo_sigma_z(t, psi0, chain_length, J, B0, A, spin_constant,
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
    exp_sig_z = np.empty((len(t), total_spins))
    if spin_constant:
        if central_spin:
            raise AttributeError(
                "With a central spin, total spin is not conserved in the chain. Therefore central_spin and spin_constant cannot be used at the same time.")
        eigenvalues, eigenvectors = diagonalization.eig_values_vectors_spin_const(
            chain_length, J, B0, periodic_boundaries, central_spin)
    else:
        eigenvalues, eigenvectors = diagonalization.eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)
    psi_z = np.arange(0, np.int(2**(total_spins)))
    sigma_z = unpackbits(psi_z, total_spins) - 1/2

    # # non vectorized version. I just keep it for the sake of clarity
    # # for time step in t
    # for time_i, ts in enumerate(t):
    #     # |psi_t> = U e^(iDt) U+ * |psi0>, with U = eigenvectors, D = diag(eigenvalues)
    #     psi_t = (eigenvectors @ np.diag(np.exp(1j * eigenvalues * ts)) @
    #              eigenvectors.conjugate().T @ psi0)
    #     exp_sig_z[time_i] = (np.inner(sigma_z.T, (np.abs(psi_t)**2)))

    # vectorized notation
    # Compute array with time vector multiplied with diagonal matrix with eigenvalues as entries
    exp_part = np.apply_along_axis(
        np.diag, 1, np.exp(1j * np.outer(t, eigenvalues)))
    psi_t = eigenvectors @ exp_part @ eigenvectors.conjugate().T @ psi0
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)

    return exp_sig_z
