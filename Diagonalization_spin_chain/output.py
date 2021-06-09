import numpy as np
import matplotlib.pyplot as plt
import os
from time_evo import time_evo_sigma_z
from diagonalization import eig_values_vectors, eig_values_vectors_spin_const
from matplotlib import animation
from support_functions import unpackbits
from scipy.special import binom


def plot_time_evo(t, psi0, chain_length, J, B0, A, spin_constant,
                  periodic_boundaries, central_spin, save):
    """
    Plots the time evolution of the spin chain and the optional central spin

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
        save (string): If not False, the output is saved with the given filename.

    """

    exp_sig_z = time_evo_sigma_z(t, psi0, chain_length, J, B0, A, spin_constant,
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
            if not os.path.isdir("./Plots"):
                os.mkdir("./Plots")
            plt.savefig("./Plots/" + save)
    plt.show()


def animate_time_evo(t, psi0, chain_length, J, B0, A, spin_constant,
                     periodic_boundaries, central_spin, save):
    """
    Animate the time evolution of the spin chain and the optional central spin

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
        save (string): If not False, the output is saved with the given filename.

    """

    exp_sig_z = time_evo_sigma_z(t, psi0, chain_length, J, B0, A, spin_constant,
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
        if not os.path.isdir("./Plots"):
            os.mkdir("./Plots")
        writervideo = animation.FFMpegWriter(fps=10)
        anim.save("./Plots/" + save, writer=writervideo)
    plt.show()


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
                  spin_constant, samples):
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
    plt.show()


def plot_r_fig3(chain_length, J, B0, periodic_boundaries, samples):
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

        yerrors = 1 / np.sqrt(samples[i] * 2**chain_length[i])
        plt.errorbar(B0, mean_r_values[i], yerr=yerrors, marker="o", capsize=5,
                     linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("r-value")
    plt.legend()
    plt.show()


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


def plot_f_fig2(chain_length, J, B0, periodic_boundaries, samples, verbose=True):
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
            for each number of samples there are (chain_length x chain_length - 2) data points
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
    plt.show()


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
            for each number of samples there are (chain_length x chain_length - 2) data points

    """
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

    plt.show()


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
    sigma_z_ravelled = unpackbits(psi_z, total_spins) - 1/2
    sigma_z = np.apply_along_axis(np.diag, 1, sigma_z_ravelled.T)
    Ma = sigma_z * np.exp(1j * 2 * np.pi *
                          np.outer(np.arange(chain_length), np.arange(chain_length)) /
                          chain_length).reshape(chain_length, chain_length, 1, 1)
    Ma_dagger = Ma.transpose(0, 1, 3, 2).conjugate()

    # <n| Ma |n> <n| Ma_dagger |n>
    exp_fa = np.zeros(chain_length)
    for n in range(N_states):
        # Sum goes over the different sites
        exp_Ma = np.sum(eigenvectors[n].T @ Ma @ eigenvectors[n], axis=1)
        Ma_Ma_dagger = np.zeros(chain_length, dtype=complex)
        for i in range(chain_length):
            for j in range(chain_length):
                Ma_Ma_dagger += (eigenvectors[n].T @
                                 (Ma[:, i] @ Ma_dagger[:, j]) @ eigenvectors[n])
        exp_fa += np.real(exp_Ma * exp_Ma.conjugate()) / np.real(Ma_Ma_dagger)

    # average over states
    return exp_fa / N_states


def plot_fa_values(chain_length, J, B0, A, periodic_boundaries, central_spin, samples):
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
            1 / np.sqrt(samples * binom(total_spins, total_spins // 2)))
        plt.errorbar(np.arange(chain_length, dtype=int), mean_fa_values[j],
                     yerr=yerrors, marker="o", capsize=5, linestyle="--", label=f"B0={B}")
    plt.xlabel("Fourier mode a")
    plt.ylabel("fa-value")
    plt.title(f"fa_values for chain_length = {chain_length}")
    plt.legend()
    plt.show()
