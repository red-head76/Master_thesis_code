from matplotlib import animation
from scipy.constants import hbar, e
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sf
import diagonalization as diag
from time_evo import time_evo_sigma_z, time_evo_subspace


def generate_f_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                      spin_constant):
    """
    Calculates the f value f = 1 - <n| M1 |n><n| M1+ |n> / <n| M1+ M1 |n>

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        spin_const (bool): If true, the conservation of total spin is used
                        to construct respective subspaces. If False, full Hamiltonian is used.

    Returns:
        f_value (float)
    """

    # Get the eigenvectors n of the Hamiltonian
    if spin_constant:
        eigenvectors = diag.eig_values_vectors_spin_const_all_subspaces(
            chain_length, J, B0, A, periodic_boundaries, central_spin)[1]
    else:
        eigenvectors = diag.eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)[1]

    total_spins = chain_length + central_spin
    dim = np.int(2**(total_spins))

    # unvectorized notation step by step
    psi_z = np.arange(dim)
    sigma_z_ravelled = sf.unpackbits(psi_z, total_spins) - 1/2
    sigma_z = np.apply_along_axis(np.diag, 1, sigma_z_ravelled.T)
    M1 = sigma_z * np.exp(1j * 2 * np.pi *
                          np.arange(total_spins) / total_spins).reshape(-1, 1, 1)
    M1dagger = M1.transpose(0, 2, 1).conjugate()

    # # Old version
    # # Expectation value of M1, summed over all sites and take the diagonal entries
    # exp_M1_old = np.diag(np.sum(eigenvectors.T.conjugate()
    #                             @ M1 @ eigenvectors, axis=0))
    # exp_M1dagger_old = np.diag(np.sum(eigenvectors.T.conjugate()
    #                                   @ M1dagger @ eigenvectors, axis=0))
    # exp_M1daggerM1_old = np.diag(np.sum(eigenvectors.T.conjugate()
    #                                     @ M1dagger @ M1 @ eigenvectors, axis=0))
    # # f_values (for all states)
    # f_values_old = 1 - exp_M1_old * exp_M1dagger_old / exp_M1daggerM1_old

    exp_M1 = np.empty(dim, dtype=complex)
    exp_M1dagger = np.empty(dim, dtype=complex)
    exp_M1daggerM1 = np.empty(dim, dtype=complex)
    for n in range(dim):
        exp_M1[n] = np.sum(eigenvectors[n].T @
                           M1 @ eigenvectors[n], axis=0)
        exp_M1dagger[n] = np.sum(eigenvectors[n].T @
                                 M1dagger @ eigenvectors[n], axis=0)
        exp_M1daggerM1[n] = np.sum(eigenvectors[n].T @
                                   (M1dagger @ M1) @ eigenvectors[n], axis=0)

    f_value = np.mean(1 - exp_M1 * exp_M1dagger / exp_M1daggerM1)
    return f_value


def plot_f_fig2(chain_length, J, B0, periodic_boundaries, samples, save, verbose=True):
    """
    Plots the f values as done in Figure 2 in https://doi.org/10.1103/PhysRevB.82.174411

    Args:
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int or array (int)): Number of times data points should be generated
        verbose (bool, default=True): prints some extra information about where the process is

    Returns:
        If save: data (list [B0, mean_f_values]), None otherwise

    """
    mean_f_values = np.empty((np.size(chain_length), np.size(B0)))
    if np.size(samples) == 1:
        samples = np.ones(np.size(chain_length), dtype=np.int) * samples
    for i, N in enumerate(chain_length):
        if verbose:
            print(f"Calculating for chain length = {N}")
        for j, B in enumerate(B0):
            if verbose:
                print(f"B0 = {B}")
            f_value = 0
            for s in range(samples[i]):
                # print every 20% step
                if ((s*5 % (samples[i])) < 5) and verbose:
                    print(f"{s}/{samples[i]} samples done")
                f_value += generate_f_values(N, J, B, 0, periodic_boundaries, False, True)
            # Averaging over samples
            mean_f_values[i, j] = f_value / samples[i]

        yerrors = 1 / np.sqrt(samples[i] * 2**chain_length[i])
        plt.errorbar(B0, mean_f_values[i], yerr=yerrors, marker="o", capsize=5,
                     linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("f-value")
    plt.legend()
    if save:
        return [B0, mean_f_values]


def generate_g_values(rho0, times, chain_length, J, B0, A, periodic_boundaries, central_spin,
                      spin_constant):
    """
    Calculates the value g = <M1> (ρ(t)) / <M1> (ρ(0))

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        spin_const (bool): If true, the conservation of total spin is used
                        to construct respective subspaces. If False, full Hamiltonian is used.

    Returns:
        g_values (array (float) [tD]): the g_values at the given timesteps (tD in total)
    """

    # Get the eigenvectors n of the Hamiltonian
    if spin_constant:
        eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const_all_subspaces(
            chain_length, J, B0, A, periodic_boundaries, central_spin)
    else:
        eigenvalues, eigenvectors = diag.eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)

    total_spins = chain_length + central_spin
    dim = np.int(2**(total_spins))

    # unvectorized notation step by step
    psi_z = np.arange(dim)
    sigma_z_ravelled = sf.unpackbits(psi_z, total_spins) - 1/2
    sigma_z = np.apply_along_axis(np.diag, 1, sigma_z_ravelled.T)
    M1 = sigma_z * np.exp(1j * 2 * np.pi *
                          np.arange(total_spins) / total_spins).reshape([-1, 1, 1])
    exp_M1_rho0 = np.empty(dim)
    exp_M1_rhot = np.empty((times.size, dim))
    exp_M1_rhot_ext = np.empty((times.size, dim, dim), dtype=complex)

    # Expectation value of M1, summed over all sites and take the diagonal entries
    for n in range(dim):
        # Should be real anyways, but sometimes there can be numerical errors (of order ~10^-50)
        exp_M1_rho0[n] = np.real(np.sum(eigenvectors[n].T @
                                        (M1 @ rho0) @ eigenvectors[n], axis=0))
    for t_idx, t in enumerate(times):
        for n in range(dim):
            for m in range(dim):
                exp_M1_rhot_ext[t_idx, n, m] = (np.exp(-1j * (eigenvalues[n] - eigenvalues[m]) * t) *
                                                (eigenvectors[n].T @ rho0 @ eigenvectors[m]) *
                                                np.sum(eigenvectors[m].T @ M1 @ eigenvectors[n], axis=0))
    # Sum over m
    exp_M1_rhot = np.sum(exp_M1_rhot_ext, axis=2)
    # Average over state
    return np.mean(1 - exp_M1_rhot / exp_M1_rho0, axis=1)


def plot_g_value(rho0, times, chain_length, J, B0, periodic_boundaries, samples, save):
    """
    Plots the f values as done in Figure 2 in https://doi.org/10.1103/PhysRevB.82.174411

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int or array (int)): Number of times data points should be generated

    """
    # TODO: fixmeeeee, also time in real units
    fig = plt.figure(figsize=(chain_length.size * 6, 4))
    axes = [fig.add_subplot(1, chain_length.size, ax, projection='3d')
            for ax in range(1, chain_length.size + 1)]

    mean_g_values = np.empty((times.size, np.size(chain_length), np.size(B0)))
    if np.size(samples) == 1:
        samples = np.ones(np.size(chain_length), dtype=np.int) * samples
    for i, N in enumerate(chain_length):
        for j, B in enumerate(B0):
            g_value = np.zeros(times.size)
            for _ in range(samples[i]):
                g_value += generate_g_values(rho0[i], times, N, J, B,
                                             0, periodic_boundaries, False, True)
            # Averaging over samples
            mean_g_values[:, i, j] = g_value / samples[i]

        # yerrors = 1 / np.sqrt(samples[i] * 2**chain_length[i])
        t_idx, B0_idx = np.meshgrid(range(times.size), range(B0.size))
        axes[i].plot_surface(times[t_idx], B0[B0_idx],
                             mean_g_values[t_idx, i, B0_idx])
        axes[i].set_title(f"N = {N}")
        axes[i].set_xlabel("Time t")
        axes[i].set_ylabel("Magnetic field amplitude B0")
        axes[i].set_zlabel("g-value")
        # plt.errorbar(B0, mean_g_values[i], yerr=yerrors, marker="o", capsize=5,
        #              linestyle="--", label=f"N={N}")
    if save:
        return [times, mean_g_values]


def generate_fa_values(chain_length, J, B0, A, periodic_boundaries, central_spin):
    """
    Calculates the fa value fa = <n| Ma |n> <n| Ma+ |n>

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present

    Returns:
        fa_value (array (float) [chain_length])
    """
    total_spins = chain_length + central_spin
    eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, B0, A, periodic_boundaries, central_spin, n_up=total_spins//2)[1]
    dim = np.int(2**(total_spins))
    # change to fullspace representation
    subspace = np.where(np.logical_not(np.sum(sf.unpackbits(np.arange(dim), total_spins),
                                              axis=1) - total_spins//2))[0]
    eigenvectors = eigenvectors @ sf.create_basis_vectors(subspace, dim)

    psi_z = np.arange(dim)
    sigma_z = sf.unpackbits(psi_z, total_spins) - 1/2
    # Sigma_z at each side j times the phase factor of exp(2πi j/N a) for each side j and mode a
    # summed over sites
    Ma = (sigma_z.reshape(1, dim, chain_length) *
          np.exp(1j * 2 * np.pi * np.outer(np.arange(chain_length), np.arange(chain_length))
                 / chain_length).reshape(chain_length, 1, chain_length)).sum(axis=2)
    exp_Ma = Ma @ (eigenvectors**2).T
    exp_Ma_Madagger = (Ma * Ma.conj()) @ (eigenvectors**2).T

    # Return mean over all states
    return np.mean(np.real(exp_Ma * exp_Ma.conj() / exp_Ma_Madagger**2), axis=1)


def plot_fa_values(chain_length, J, B0, A, periodic_boundaries, central_spin, samples, save):
    """
    Plots the f values as done in Figure 2 in https://doi.org/10.1103/PhysRevB.82.174411

    Args:
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int or array (int)): Number of times data points should be generated
            for each number of samples there are binom(total_spins, total_spins // 2) data points


    Returns:
        If save: data (list [arange(chain_length), mean_fa_values]), None otherwise

    """
    # convert lists to int
    if chain_length.size > 1:
        raise Warning("chain length should be an integer, not a list when plotting fa")
    chain_length = chain_length[0]
    if samples.size > 1:
        raise Warning("samples should be an integer, not a list when plotting fa")
    samples = samples[0]
    mean_fa_values = np.empty((np.size(B0), chain_length))
    # Allow only one chain length for the moment
    # if np.size(samples) == 1:
    #     samples = np.ones(np.size(total_spins), dtype=np.int) * samples
    # for i, N in enumerate(chain_length):
    for j, B in enumerate(B0):
        fa_values = np.zeros(chain_length)
        for s in range(samples):
            print(f"Sample {s+1} / {samples}")
            fa_values += generate_fa_values(chain_length, J, B, A,
                                            periodic_boundaries, central_spin)
        # Averaging over samples
        mean_fa_values[j] = fa_values / samples
        yerrors = (1 / np.sqrt(samples))
        plt.errorbar(np.arange(chain_length, dtype=int), mean_fa_values[j],
                     yerr=yerrors, marker="o", capsize=5, linestyle="--", label=f"B0={B}")
    plt.xlabel("Fourier mode a")
    plt.ylabel("fa-value")
    plt.xticks(np.arange(chain_length))
    plt.title(f"fa_values for chain_length = {chain_length}")
    plt.legend(loc=4)
    if save:
        return [np.arange(chain_length), mean_fa_values]
