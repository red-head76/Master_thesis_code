from matplotlib import animation
from scipy.constants import hbar, e
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sf
import diagonalization as diag
from time_evo import time_evo_sigma_z, time_evo


def calc_half_chain_entropy_old(times, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                                seed, scaling):
    """
    Calculates the half chain entropy -tr(rho_a, ln(rho_a))

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly
        scaling (string): the scaling of the coupling A with the chain length.

    Returns:
        half_chain_entropy (array (float) [times])

    """

    total_spins = chain_length + central_spin
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))[0]

    eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries,
        central_spin, n_up=total_spins//2, seed=seed, scaling=scaling)
    psi_t = time_evo(times, eigenvalues, eigenvectors, total_spins)
    # This performs an outer product along axis 1
    rho_t = psi_t[:, :, np.newaxis] * psi_t.conj()[:, np.newaxis, :]
    # # Old way: go back to full space to calculate the partial trace. Even though there must
    # # be a smarter way to do this...
    # rho_t_fullspace = np.zeros((times.size, dim, dim), dtype=complex)
    # subspace_mask2D = np.meshgrid(subspace_mask, subspace_mask)
    # rho_t_fullspace[:, subspace_mask2D[1], subspace_mask2D[0]] = rho_t
    # # partial_trace over b -> rho_a(t)
    # rho_a_t = sf.partial_trace(rho_t_fullspace, total_spins//2)
    # Smarter way to to it:
    rho_a_t = sf.partial_trace_subspace(rho_t, subspace_mask, total_spins//2)
    # hce = -tr(rho_a ln(rho_a))
    #     = -tr(rho ln(rho)) = tr(D ln(D)), where D is the diagonalized matrix
    # should be real positive anyways, but to prevent complex casting warnings
    D = np.linalg.eigvalsh(rho_a_t)
    # ugly, but cuts out the the Runtime warning caused by of 0 values in log
    return -np.sum(D * np.log(D, where=D > 0, out=np.zeros(D.shape, dtype=D.dtype)), axis=1)


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


def sigma_E(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples,
            seed, scaling, save, initial_state):
    """
    Energy distribution of a given initial state

    Returns:
        If save: data (list [eigenvalues_save, E_distr_save]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_0 = psi_0[subspace_mask]
    if save:
        eigenvalues_save = np.empty((len(B0), (samples * subspace_mask.size)))
        E_distr_save = np.empty((len(B0), (samples * subspace_mask.size)))
    for j, B in enumerate(B0):
        eigenvalues = np.empty((samples, subspace_mask.size))
        eigenvectors = np.empty((samples, subspace_mask.size, subspace_mask.size))
        for i in range(samples):
            eigenvalues[i], eigenvectors[i] = diag.eig_values_vectors_spin_const(
                chain_length, J, J_xy, B, A, periodic_boundaries, central_spin, n_up)
        E_distr = np.abs(eigenvectors.transpose(0, 2, 1) @ psi_0)**2
        sort = eigenvalues.ravel().argsort()
        eigenvalues = eigenvalues.ravel()[sort]
        E_distr = E_distr.ravel()[sort]
        line, = plt.plot([], [])        # color dummy
        plt.hist(eigenvalues, weights=E_distr, bins=sturge_rule(eigenvalues.size), density=True,
                 histtype="step", label=f"W={B}", color=line.get_c())
        plt.vlines([eigenvalues[0], eigenvalues[-1]], 0, plt.axis()[3],
                   color=line.get_c(), ls=':')
        if save:
            eigenvalues_save[j] = eigenvalues
            E_distr_save[j] = E_distr
    plt.legend()
    if save:
        return [eigenvalues_save, E_distr_save]


def width_sigma_E(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples,
                  seed, scaling, save, initial_state):
    """
    Width of the energy distribution per sample of a given initial state

    Returns:
        If save: data (list [eigenvalues, E_distr]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_0 = psi_0[subspace_mask]
    width_means = np.empty(len(B0))
    width_stds = np.empty(len(B0))
    for j, B in enumerate(B0):
        eigenvalues = np.empty((samples, subspace_mask.size))
        eigenvectors = np.empty((samples, subspace_mask.size, subspace_mask.size))
        for i in range(samples):
            eigenvalues[i], eigenvectors[i] = diag.eig_values_vectors_spin_const(
                chain_length, J, J_xy, B, A, periodic_boundaries, central_spin, n_up)
        E_distr = np.abs(eigenvectors.transpose(0, 2, 1) @ psi_0)**2
        # w_avg = np.average(eigenvalues, weights=E_distr, axis=1)
        # w_std = np.sqrt(np.average((eigenvalues - w_avg[:, None])**2, weights=E_distr, axis=1))
        w_std2 = np.sqrt((np.average(eigenvalues, weights=E_distr, axis=1))**2 -
                         np.average(eigenvalues, weights=E_distr**2, axis=1))
        # print(-np.sqrt(w_var) / (np.sort(eigenvalues.ravel())
        #                          [0] - np.sort(eigenvalues.ravel())[-1]))
        # plt.errorbar(w_std.mean() , B, xerr=np.sqrt(w_std.std()), capsize=2)
        width_means[j] = w_std2.mean()
        width_stds[j] = w_std2.std()
    plt.plot(B0, width_means)
    if save:
        return [width_means, width_stds]


def plot_ds_deff(times, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples,
                 seed, scaling, save, initial_state):
    """
    Plots the value of 1/2 sqrt{ds^2/d^eff} of a given inital state (see
    10.1103/PhysRevE.79.061103)

    Returns:
        If save: data (list [B0, ds_deff_means, ds_deff_stds]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_0 = psi_0[subspace_mask]
    ds_deff_means = np.empty(len(B0))
    ds_deff_stds = np.empty(len(B0))
    distances_means = np.empty(len(B0))
    distances_stds = np.empty(len(B0))
    for j, B in enumerate(B0):
        eigenvalues = np.empty((samples, subspace_mask.size))
        eigenvectors = np.empty((samples, subspace_mask.size, subspace_mask.size))
        distances = np.empty(samples)
        for i in range(samples):
            eigenvalues[i], eigenvectors[i] = diag.eig_values_vectors_spin_const(
                chain_length, J, J_xy, B, A, periodic_boundaries, central_spin, n_up)
            psi_t = time_evo(times, chain_length, J, J_xy, B, A, periodic_boundaries,
                             central_spin, seed, scaling, initial_state)
            rho_t = psi_t[:, :, np.newaxis] * psi_t[:, np.newaxis, :]  # density matrix in subspace
            rho_a_t = sf.partial_trace_subspace(rho_t, subspace_mask, 1)  # density matrix of a
            omega_a = np.mean(rho_a_t, axis=0)                         # time average
            # distances[i] = np.mean(1/2 *
            #     np.diagonal(np.sqrt((rho_a - omega_a) @ (rho_a - omega_a)), axis1=1, axis2=2))
            distances[i] = np.mean(np.linalg.norm((rho_a_t - omega_a), ord="nuc", axis=(1, 2)))
        E_distr = np.abs(eigenvectors.transpose(0, 2, 1) @ psi_0)**2
        deff = 1 / np.sum(E_distr**4, axis=1)
        ds_deff_means[j] = np.mean(1/2 * np.sqrt(4 / deff))
        ds_deff_stds[j] = np.mean(1/4 * np.sqrt(4 / deff**3)) * deff.std()
        distances_means[j] = distances.mean()
        distances_stds[j] = distances.std()
    plt.errorbar(B0, ds_deff_means, ds_deff_stds, capsize=2, label="ds/deff")
    plt.errorbar(B0, distances_means, distances_stds, capsize=2, label=r"<D>_t")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Disorder W")
    if save:
        return [B0, ds_deff_means, ds_deff_stds, distances_means, distances_stds]
