import numpy as np
import matplotlib.pyplot as plt
import os
from time_evo import time_evo_sigma_z
from diagonalization import eig_values_vectors, eig_values_vectors_spin_const
from matplotlib import animation
from support_functions import packbits, unpackbits, partial_trace, save_data


def plot_time_evo(t, idx_psi_0, chain_length, J, B0, A, spin_constant,
                  periodic_boundaries, central_spin, save):
    """
    Plots the time evolution of the spin chain and the optional central spin

    Args:
        t (array [tN]): array with tN timesteps
        idx_psi_0 (int): the index of the state at t0
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
        save (string): If not False, the output is saved with the given filename.

    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    exp_sig_z = time_evo_sigma_z(t, psi_0, chain_length, J, B0, A, spin_constant,
                                 periodic_boundaries, central_spin)
    total_spins = chain_length + central_spin
    fig, ax = plt.subplots(total_spins, 1, figsize=(
        10, 1 + total_spins), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    for i in range(total_spins):
        if i != (total_spins - 1) or not central_spin:
            ax[i].plot(t, exp_sig_z.T[i])
        else:
            ax[i].plot(t, exp_sig_z.T[i], color="C1")
            ax[i].set_title("central spin")
        ax[i].axhline(0, color="black")
        ax[i].set_ylim(-0.55, 0.55)

        ax[-1].set_xlabel("Time t")
        if save:
            data = [t, exp_sig_z]
            params = {"plot type": "time_evo", "data structure": "[t, exp_sig_z]",
                      "idx_psi_0": idx_psi_0, "chain_length": chain_length, "J": J,
                      "B0": B0, "A": A, "spin_constant": spin_constant,
                      "periodic_boundaries": periodic_boundaries, "central_spin": central_spin}
            save_data(save, data, params)


def animate_time_evo(t, idx_psi_0, chain_length, J, B0, A, spin_constant,
                     periodic_boundaries, central_spin, save):
    """
    Animate the time evolution of the spin chain and the optional central spin

    Args:
        t (array [tN]): array with tN timesteps
        idx_psi_0 (int): the index of the state at t0
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
        save (string): If not False, the output is saved with the given filename.

    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    exp_sig_z = time_evo_sigma_z(t, psi_0, chain_length, J, B0, A, spin_constant,
                                 periodic_boundaries, central_spin)
    total_spins = chain_length + central_spin
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = plt.bar(np.arange(total_spins), exp_sig_z[0])
    ax.set_ylim(-0.6, 0.7)
    if central_spin:
        bars.patches[-1].set_color("C1")
        ax.axvline(total_spins - 1.5, color="black", linewidth=8)
        ax.annotate("central spin", (total_spins - 1.2, 0.65))

    def run(i):
        for j, bar in enumerate(bars):
            bar.set_height(exp_sig_z[i, j])
        return bars

    anim = animation.FuncAnimation(
        fig, run, frames=t.size, blit=True, interval=100)
    if save:
        data = [t, exp_sig_z]
        params = {"plot type": "anim_time_evo", "data structure": "[t, exp_sig_z]",
                  "idx_psi_0": idx_psi_0, "chain_length": chain_length, "J": J,
                  "B0": B0, "A": A, "spin_constant": spin_constant,
                  "periodic_boundaries": periodic_boundaries, "central_spin": central_spin}
        save_data(save, data, params, anim=anim)


# Histogram functions for r_value

def rice_rule(n):
    """
    Rule for calculating bins in a histogram (https://en.wikipedia.org/wiki/Histogram#Rice_Rule)
    In principle the choice can be arbitrary, but there should be a predetermined choice without
    looking at the data. This is some middle-way choice between square root rule and log rule.

    Args:
        n (int): number of data points

    Returns:
        nbins (int): number of bins to use according to rice rule

    """

    return np.int(np.ceil(2 * (n)**(1/3)))


def sturge_rule(n):
    """
    Rule for calculating bins in a histogram
    (https://en.wikipedia.org/wiki/Histogram#Sturges'_formula).
    In principle the choice can be arbitrary, but there should be a predetermined choice without
    looking at the data. This is give lower number of bins than rice rule

    Args:
        n (int): number of data points

    Returns:
        nbins (int): number of bins to use according to rice rule

    """
    return np.int(np.ceil(np.log2(n)) + 1)


def generate_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                      spin_constant, binning_func=sturge_rule):
    """
    Calculates the r value, the fraction of the difference of eigenvalues of the given Hamiltonian:
    r = min (ΔE_n, ΔE_n+1) / max (ΔE_n, ΔE_n+1)

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
        r_values (array (float) [2**total_spins - 2])

    """
    # Get the energy eigenvalues
    if spin_constant:
        E = eig_values_vectors_spin_const(
            chain_length, J, B0, A, periodic_boundaries, central_spin,
            only_biggest_subspace=True)[0]
    else:
        raise Warning("r_value with the fullspace H doesn't make sense!")
        E = eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)[0]

    E = np.sort(E)
    Delta_E = np.diff(E)
    Delta_E_shifted = np.roll(Delta_E, 1)
    Delta_min = np.min((Delta_E, Delta_E_shifted), axis=0)[1:]
    Delta_max = np.max((Delta_E, Delta_E_shifted), axis=0)[1:]
    # # Calculate the distribution approximately with a histogram
    # hist, bin_edges = np.histogram(Delta_min / Delta_max, bins=binning_func(Delta_min.size),
    #                                density=True, range=(0, 1))
    # bin_centers = (bin_edges + np.roll(bin_edges, 1))[1:] / 2

    # return hist * bin_centers
    return Delta_min / Delta_max


def plot_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                  spin_constant, samples, save):
    """
    Plots the histogram of r_values created by the given parameters.

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        spin_constant (bool): If true, the conservation of total spin is used to construct
                           respective subspaces. If False, full Hamiltonian is used.
        samples (int): Number of times data points should be generated for each number
                                    of samples there are (chain_length x chain_length - 2) data
                                    points
    """

    r_values = generate_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                                 spin_constant)
    for _ in range(samples - 1):
        r_values += generate_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                                      spin_constant)
    # Average over samples
    r_values /= samples
    plt.hist(r_values, bins=sturge_rule(r_values.size), density=True)
    # Average over states
    plt.plot([], [], ls="", label=f"Average r = {np.mean(r_values):.2f}")
    plt.xlabel("r")
    plt.ylabel(r"Density $\rho (r)$")
    plt.title(f"R values averaged over {samples} samples")
    plt.legend()
    if save:
        data = [r_values]
        params = {"plot type": "r_values", "data structure": "[r_values]",
                  "chain_length": chain_length, "J": J,
                  "B0": B0, "A": A, "spin_constant": spin_constant,
                  "periodic_boundaries": periodic_boundaries, "central_spin": central_spin}
        save_data(save, data, params)


def plot_r_fig3(chain_length, J, B0, periodic_boundaries, samples, save):
    """
    Plots the r values as done in Figure 3 in https://doi.org/10.1103/PhysRevB.82.174411

    Args:
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int or array (int)): Number of times data points should be generated
            for each number of samples there are (chain_length x chain_length - 2) data points

    """

    mean_r_values = np.empty((np.size(chain_length), np.size(B0)))
    if np.size(samples) == 1:
        samples = np.ones(np.size(chain_length), dtype=np.int) * samples
    for i, N in enumerate(chain_length):
        for j, B in enumerate(B0):
            r_values = generate_r_values(N, J, B,
                                         0, periodic_boundaries, False, True)
            for _ in range(samples[i] - 1):
                r_values += generate_r_values(N, J, B,
                                              0, periodic_boundaries, False, True)
            r_values /= samples[i]
            # Averaging over samples and states at the same time
            mean_r_values[i, j] = np.mean(r_values)
        yerrors = 1 / np.sqrt(samples[i])
        plt.errorbar(B0, mean_r_values[i], yerr=yerrors, marker="o", capsize=5,
                     linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("r-value")
    plt.legend()
    if save:
        data = [B0, mean_r_values]
        params = {"plot type": "r_values", "data structure": "[B0, mean_r_values]",
                  "chain_length": chain_length, "J": J,
                  "B0": B0, "periodic_boundaries": periodic_boundaries, "samples": samples}
        save_data(save, data, params)


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
        eigenvectors = eig_values_vectors_spin_const(
            chain_length, J, B0, A, periodic_boundaries, central_spin,
            only_biggest_subspace=False)[1]
    else:
        eigenvectors = eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)[1]

    total_spins = chain_length + central_spin
    dim = np.int(2**(total_spins))

    # unvectorized notation step by step
    psi_z = np.arange(dim)
    sigma_z_ravelled = unpackbits(psi_z, total_spins) - 1/2
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
                f_value += generate_f_values(N, J, B,
                                             0, periodic_boundaries, False, True)
            # Averaging over samples
            mean_f_values[i, j] = f_value / samples[i]

        yerrors = 1 / np.sqrt(samples[i] * 2**chain_length[i])
        plt.errorbar(B0, mean_f_values[i], yerr=yerrors, marker="o", capsize=5,
                     linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("f-value")
    plt.legend()
    if save:
        data = [B0, mean_f_values]
        params = {"plot type": "f_fig2", "data structure": "[B0, mean_f_values]",
                  "chain_length": chain_length, "J": J,
                  "B0": B0, "periodic_boundaries": periodic_boundaries, "samples": samples}
        save_data(save, data, params)


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
        eigenvalues, eigenvectors = eig_values_vectors_spin_const(
            chain_length, J, B0, A, periodic_boundaries, central_spin,
            only_biggest_subspace=False)
    else:
        eigenvalues, eigenvectors = eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)

    total_spins = chain_length + central_spin
    dim = np.int(2**(total_spins))

    # unvectorized notation step by step
    psi_z = np.arange(dim)
    sigma_z_ravelled = unpackbits(psi_z, total_spins) - 1/2
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


def plot_g_value(rho0, times, chain_length, J, B0, periodic_boundaries, samples):
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
    # TODO: fixmeeeee
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


def generate_fa_values(chain_length, J, B0, A, periodic_boundaries, central_spin):
    """
    Calculates the fa value fa = <n| Ma |n> <n| Ma+ |n>

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present

    Returns:
        fa_value (array (float) [chain_length])
    """
    eigenvectors = eig_values_vectors_spin_const(
        chain_length, J, B0, A, periodic_boundaries, central_spin,
        only_biggest_subspace=True)[1]

    N_states = eigenvectors.shape[0]
    total_spins = chain_length + central_spin
    dim = np.int(2**(total_spins))

    psi_z = np.arange(dim)
    sigma_z = unpackbits(psi_z, total_spins) - 1/2
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
    """
    # convert lists to int
    if chain_length.size > 1:
        raise Warning(
            "chain length should be an integer, not a list when plotting fa")
    chain_length = chain_length[0]
    if samples.size > 1:
        raise Warning(
            "samples should be an integer, not a list when plotting fa")
    samples = samples[0]
    total_spins = chain_length + central_spin
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

        yerrors = (
            1 / np.sqrt(samples))
        plt.errorbar(np.arange(chain_length, dtype=int), mean_fa_values[j],
                     yerr=yerrors, marker="o", capsize=5, linestyle="--", label=f"B0={B}")
    plt.xlabel("Fourier mode a")
    plt.ylabel("fa-value")
    plt.xticks(np.arange(chain_length))
    plt.title(f"fa_values for chain_length = {chain_length}")
    plt.legend(loc=4)
    if save:
        data = [np.arange(chain_length), mean_fa_values]
        params = {"plot type": "fa_values", "data structure": "[fa_modes, mean_fa_values]",
                  "chain_length": chain_length, "J": J, "A": A,
                  "B0": B0, "periodic_boundaries": periodic_boundaries,
                  "central_spin": central_spin, "samples": samples}
        save_data(save, data, params)


def plot_Sa_values(times, chain_length, J, B0, As, periodic_boundaries, samples, save):
    """
    Plots the Sa(t) values (see fig2 in http://arxiv.org/abs/1806.08316)

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int or array (int)): Number of times data points should be generated

    """
    # TODO: Samples currently not implemented
    total_spins = chain_length[0] + 1
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    sub_room_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))[0]
    # Starting with a Neel state , i.e. |up, down, up, down, ...>
    psi_0 = np.zeros(dim)
    psi_0[packbits(np.arange(total_spins) % 2)] = 1
    psi_0 = psi_0[sub_room_mask]

    for A in As:
        eigenvalues, eigenvectors = eig_values_vectors_spin_const(
            chain_length[0], J, B0[0], A, periodic_boundaries,
            central_spin=True, only_biggest_subspace=True)
        eigenvectors = (eigenvectors.T[sub_room_mask]).T
        # dimension of the biggest subspace
        dim_bss = eigenvectors.shape[0]
        # Time evolution of rho: rho(t) = U+ rho0 U with U = exp(-i H t)
        # U for all times, i.e. [times, dim, dim]
        # exp_part = np.apply_along_axis(
        #     np.diag, 1, np.exp(-1j * np.outer(times, eigenvalues)))
        # U = eigenvectors.T @ exp_part @ eigenvectors
        # rho_t = U.transpose(0, 2, 1).conjugate() @ rho_0 @ U
        exp_part = np.exp(1j * np.outer(times, eigenvalues))
        psi_t = (eigenvectors @
                 (exp_part.reshape(times.size, dim_bss, 1) * eigenvectors.T) @ psi_0)
        # This performs an outer product along axis 1
        rho_t = psi_t[:, :, np.newaxis] * psi_t.conj()[:, np.newaxis, :]
        # For now: go back to full space to calculate the partial trace. Even though there must
        # be a smarter way to do this...
        rho_t_fullspace = np.zeros((times.size, dim, dim), dtype=complex)
        sub_room_mask2D = np.meshgrid(sub_room_mask, sub_room_mask)
        rho_t_fullspace[:, sub_room_mask2D[1], sub_room_mask2D[0]] = rho_t
        # partial_trace over b -> rho_a(t)
        rho_a_t = partial_trace(rho_t_fullspace, total_spins//2)
        # Sa = -tr(rho_a ln(rho_a))
        #    = -tr(rho ln(rho)) = tr(D ln(D)), where D is the diagonalized matrix
        eigvals = np.linalg.eigvalsh(rho_a_t)
        # to prevent errors because of ln(0)
        eigvals += 1e-20
        Sa = -np.sum(eigvals * np.log(eigvals), axis=1)
        plt.plot(times, Sa, label=f"A={A}")

    plt.xlabel("time t")
    plt.ylabel("Sa(t)")
    plt.semilogx()
    plt.legend()
    if save:
        data = [times, Sa]
        params = {"plot type": "time_evo", "data structure": "[t, exp_sig_z]",
                  "chain_length": chain_length, "J": J, "B0": B0, "A": As,
                  "periodic_boundaries": periodic_boundaries, "samples": samples}
        save_data(save, data, params)


def calc_occupation_imbalance(times, chain_length, J, B0, A, periodic_boundaries, central_spin,
                              seed):
    """
    Calculates the occupation imbalance sum_odd s_z - sum_even s_z

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): coupling between the central spin and the spins in the chain
        periodic_boundaries (boolx): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        occupation_imbalance (array (float) [times])
    """
    total_spins = chain_length + central_spin
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    sub_room_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))
    eigenvalues, eigenvectors = eig_values_vectors_spin_const(
        chain_length, J, B0, A, periodic_boundaries, central_spin,
        only_biggest_subspace=True, seed=seed)
    eigenvectors = (eigenvectors.T[sub_room_mask]).T
    psi_z = np.arange(0, np.int(2**(total_spins)))[sub_room_mask]
    # discard last spin
    sigma_z = (unpackbits(psi_z, total_spins) - 1/2)[:, :-1]
    # Initialize in Neel state
    psi_0 = np.zeros(dim)
    psi_0[packbits(np.arange(total_spins) % 2)] = 1
    psi_0 = psi_0[sub_room_mask]
    exp_part = np.exp(1j * np.outer(times, eigenvalues))
    psi_t = eigenvectors @ (exp_part.reshape(times.size, eigenvalues.size, 1)
                            * eigenvectors.T) @ psi_0
    # discard central spin in exp_sig_z
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    # occupation imbalance mask: even minus odd sites
    occ_imbalance = np.where(np.arange(chain_length) %
                             2, exp_sig_z, -exp_sig_z).sum(axis=1)
    # and norm it to 1
    return occ_imbalance / (chain_length / 2)


def plot_occupation_imbalance(times, chain_length, J, B0, As, periodic_boundaries, central_spin,
                              samples, seed, save):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (boolx): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly
    """
    if save:
        occupation_imbalance_means = np.empty(
            (len(chain_length), len(As), len(B0), len(times)))
        occupation_imbalance_errors = np.empty(
            (len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                occupation_imbalance = np.zeros((samples[0], times.size))
                for sample in range(samples[i]):
                    occupation_imbalance[sample] = calc_occupation_imbalance(
                        times, N, J, B, A, periodic_boundaries, central_spin, seed)
                occupation_imbalance_mean = occupation_imbalance.mean(axis=0)
                yerrors = occupation_imbalance.std(
                    axis=0) / np.sqrt(samples[0])
                if save:
                    occupation_imbalance_means[i, a,
                                               b] = occupation_imbalance_mean
                    occupation_imbalance_errors[i, a, b] = yerrors
                plt.plot(times, occupation_imbalance_mean, label=f"N={N}")
                plt.fill_between(times, occupation_imbalance_mean + yerrors,
                                 occupation_imbalance_mean - yerrors, alpha=0.2)
                plt.title(
                    f"Occupation imbalance for \nJ={J}, B={B}, A={A}, central_spin={central_spin}")
    plt.xlabel("time")
    plt.semilogx()
    plt.ylabel("occupation imbalance")
    plt.legend(loc=1)
    if save:
        data = [times, occupation_imbalance_means,
                occupation_imbalance_errors]
        params = {"plot type": "occupation_imbalance", "data structure":
                  "[times, occupation_imbalance_means, occupation_imbalance_errors]",
                  "chain_length": chain_length, "J": J, "B0": B0, "A": As,
                  "periodic_boundaries": periodic_boundaries, "samples": samples, "seed": seed}
        save_data(save, data, params)


def calc_exp_sig_z_central_spin(times, chain_length, J, B0, A, periodic_boundaries, seed):
    """
    Calculates the expectation value for the central spin.

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (boolx): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        exp_sig_z (array (float) [times]): the expectation value for the central spin
    """
    total_spins = chain_length[0] + 1
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    sub_room_mask = np.where(np.logical_not(np.sum(unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))
    eigenvalues, eigenvectors = eig_values_vectors_spin_const(
        chain_length[0], J, B0, A, periodic_boundaries, True,
        only_biggest_subspace=True, seed=seed)
    eigenvectors = (eigenvectors.T[sub_room_mask]).T
    psi_z = np.arange(0, np.int(2**(total_spins)))[sub_room_mask]
    # discard last spin
    sigma_z = (unpackbits(psi_z, total_spins) - 1/2)[:, -1]
    # Initialize in Neel state
    psi_0 = np.zeros(dim)
    psi_0[packbits(np.arange(total_spins) % 2)] = 1
    psi_0 = psi_0[sub_room_mask]
    # e ^ i D t in shape (times, dim, dim)
    exp_part = np.apply_along_axis(
        np.diag, 1, np.exp(1j * np.outer(times, eigenvalues)))
    psi_t = eigenvectors @ exp_part @ eigenvectors.T @ psi_0
    # discard central spin in exp_sig_z
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    # and norm it to 1
    return exp_sig_z


def plot_exp_sig_z_central_spin(times, chain_length, J, B0, As, periodic_boundaries,
                                samples, seed, save):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (boolx): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly
    """
    # for saving the data
    if save:
        exp_sig_z_means = np.empty((len(As), len(B0), len(times)))
        exp_sig_z_errors = np.empty((len(As), len(B0), len(times)))
    for a, A in enumerate(As):
        for b, B in enumerate(B0):
            exp_sig_z = np.zeros((samples[0], times.size))
            for sample in range(samples[0]):
                exp_sig_z[sample] = calc_exp_sig_z_central_spin(
                    times, chain_length, J, B, A, periodic_boundaries, seed)
            exp_sig_z_mean = exp_sig_z.mean(axis=0)
            yerrors = exp_sig_z.std(axis=0) / np.sqrt(samples[0])
            if save:
                exp_sig_z_means[a, b] = exp_sig_z_mean
                exp_sig_z_errors = yerrors
            plt.plot(times, exp_sig_z_mean, label=f"B0={B}, A={A}")
            plt.fill_between(times, exp_sig_z_mean + yerrors,
                             exp_sig_z_mean - yerrors, alpha=0.2)
            plt.title(
                f"Expectation value of the central spin for \n N={chain_length[0]}, J={J}")
    plt.xlabel("time")
    plt.ylabel(r"$<S_z>$")
    plt.legend(loc=1)
    if save:
        data = [times, exp_sig_z_means,
                exp_sig_z_errors]
        params = {"plot type": "exp_sig_z_central_spin", "data structure":
                  "[times, exp_sig_z_means, exp_sig_z_errors]",
                  "chain_length": chain_length, "J": J, "B0": B0, "A": As,
                  "periodic_boundaries": periodic_boundaries, "seed": seed, "samples": samples}
        save_data(save, data, params)
