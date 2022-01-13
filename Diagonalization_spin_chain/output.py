from matplotlib import animation
from scipy.constants import hbar, e
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sf
import diagonalization as diag
from time_evo import time_evo_sigma_z, time_evo_subspace

# Doesn't work on justus, so commented out
# import seaborn as sns
# sns.set_theme(context="paper")


def plot_time_evo(t, idx_psi_0, chain_length, J, J_xy, B0, A, spin_constant,
                  periodic_boundaries, central_spin, seed=False, scaling="sqrt", save=False):
    """
    Plots the time evolution of the spin chain and the optional central spin

    Args:
        t (array [tN]): array with tN timesteps
        idx_psi_0 (int): the index of the state at t0
        chain_length (int): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
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

    Returns:
        If save: data (list [time, exp_sig_z]), None otherwise

    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    exp_sig_z = time_evo_sigma_z(t, idx_psi_0, chain_length, J, J_xy, B0, A, spin_constant,
                                 periodic_boundaries, central_spin, seed, scaling)
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

        ax[-1].set_xlabel("Time in fs")
        if save:
            return [t, exp_sig_z]


def animate_time_evo(t, idx_psi_0, chain_length, J, J_xy, B0, A, spin_constant,
                     periodic_boundaries, central_spin, seed, scaling="sqrt", save="False"):
    """
    Animate the time evolution of the spin chain and the optional central spin

    Args:
        t (array [tN]): array with tN timesteps
        idx_psi_0 (int): the index of the state at t0
        chain_length (int): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
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


    Returns:
        If save: data (list [time, exp_sig_z]), None otherwise

    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    exp_sig_z = time_evo_sigma_z(t, idx_psi_0, chain_length, J, J_xy, B0, A, spin_constant,
                                 periodic_boundaries, central_spin, seed)
    total_spins = chain_length + central_spin
    np.random.seed(seed)
    B = np.random.uniform(-1, 1, chain_length)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Stem container containing markerline, stemlines, baseline
    stem_container = ax.stem(
        np.arange(total_spins), exp_sig_z[0], use_line_collection=True)
    ax.step(np.arange(total_spins), B, color="C2", where="mid")
    # ax.set_ylim(-0.6, 0.7)
    if central_spin:
        # stem_container[1][-1].set_color("C1")
        ax.axvline(total_spins - 1.5, color="black", linewidth=8)
        ax.annotate("central spin", (total_spins - 1.2, 0.65))

    def run(i):
        # Markers
        stem_container[0].set_ydata(exp_sig_z[i])
        stem_container[1].set_paths([np.array([[x, 0], [x, y]])
                                     for (x, y) in zip(np.arange(total_spins), exp_sig_z[i])])
        return stem_container

    anim = animation.FuncAnimation(
        fig, run, frames=t.size, blit=True, interval=100)
    if save:
        return [t, exp_sig_z]


def calc_eigvals_eigvecs_biggest_subspace(chain_length, J, J_xy, B0, A, periodic_boundaries,
                                          central_spin, seed, scaling):
    """
    Calculates the eigenvalues and vectors of the biggest subspace

    Args:
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

    Returns:
        eigenvalues (float [dim]), eigenvectors (float [dim, dim])
    """
    n_up = (chain_length[0] + central_spin) // 2
    return diag.eig_values_vectors_spin_const(chain_length[0], J, J_xy, B0[0], A[0], periodic_boundaries,
                                              central_spin, n_up, seed, scaling)


def calc_psi_t(times, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, seed, scaling,
               inital_state="neel"):
    """
    Calculates the time evolution of an initial state

    Args:
        times (float [times]): time array
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

    Returns:
        eigenvalues (float [dim]), eigenvectors (float [dim, dim])
    """
    total_spins = chain_length + central_spin
    n_up = (total_spins) // 2
    eigvals, eigvecs = diag.eig_values_vectors_spin_const(chain_length, J, J_xy, B0, A,
                                                          periodic_boundaries, central_spin,
                                                          n_up, seed, scaling)
    return time_evo_subspace(times, eigvals, eigvecs, total_spins, inital_state, 32)


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


def generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                      spin_constant, binning_func=sturge_rule):
    """
    Calculates the r value, the fraction of the difference of eigenvalues of the given Hamiltonian:
    r = min (ΔE_n, ΔE_n+1) / max (ΔE_n, ΔE_n+1)

    Args:
        chain_length (int): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
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
    total_spins = chain_length + central_spin
    if spin_constant:
        E = diag.eig_values_vectors_spin_const(
            chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, n_up=total_spins//2)[0]
    else:
        raise Warning("r_value with the fullspace H doesn't make sense!")
        E = diag.eig_values_vectors(
            chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin)[0]

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


def plot_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                  spin_constant, samples, scaling="sqrt", save=False):
    """
    Plots the histogram of r_values created by the given parameters.

    Args:
        chain_length (int): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
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

    Returns:
        If save: data (list [r_values]), None otherwise

    """

    r_values = generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                                 spin_constant)
    for _ in range(samples - 1):
        r_values += generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
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
        return [r_values]


def plot_r_fig3(chain_length, J, J_xy, B0, periodic_boundaries, samples, scaling="sqrt",
                save=False):
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

    Returns:
        If save: data (list [B0, mean_r_values]), None otherwise

    """

    mean_r_values = np.empty((np.size(chain_length), np.size(B0)))
    std_r_values = np.empty((np.size(chain_length), np.size(B0)))
    if np.size(samples) == 1:
        samples = np.ones(np.size(chain_length), dtype=np.int) * samples
    for i, N in enumerate(chain_length):
        for j, B in enumerate(B0):
            r_values = generate_r_values(N, J, J_xy, B, A[0], periodic_boundaries,
                                         central_spin, True)
            for _ in range(samples[i] - 1):
                r_values += generate_r_values(N, J, J_xy, B, A[0], periodic_boundaries,
                                              central_spin, True)
            # Averaging over samples
            r_values /= samples[i]
            # and states
            mean_r_values[i, j] = np.mean(r_values)
            std_r_values[i, j] = np.std(r_values)
        plt.errorbar(B0, mean_r_values[i], yerr=std_r_values[i]/np.sqrt(samples[i]), marker="o",
                     capsize=5, linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("r-value")
    # plt.ylim(0.36, 0.54)
    plt.legend()
    if save:
        return [B0, mean_r_values, std_r_values]


def calc_half_chain_entropy(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                            central_spin, seed, scaling):
    """
    Calculates the half chain entropy -tr_a(rho_a, ln(rho_a))

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
    psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
    splitted_subspace_mask = np.split(sf.unpackbits(
        subspace_mask, total_spins), [total_spins//2], axis=1)
    state_left = sf.packbits(splitted_subspace_mask[0])
    state_right = sf.packbits(splitted_subspace_mask[1])
    schmidt_basis_left = np.unique(state_left)
    schmidt_basis_right = np.unique(state_right)
    schmidt_state_left = (schmidt_basis_left == state_left[:, np.newaxis]).argmax(axis=1)
    schmidt_state_right = (schmidt_basis_right == state_right[:, np.newaxis]).argmax(axis=1)
    decompositioned = np.zeros((times.size, schmidt_basis_left.size, schmidt_basis_right.size),
                               dtype=complex)
    decompositioned[:, schmidt_state_left, schmidt_state_right] = psi_t
    singular_values = np.linalg.svd(decompositioned, compute_uv=False)
    return -np.sum(singular_values**2 * np.log(singular_values**2), axis=1)


def plot_half_chain_entropy(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                            central_spin, samples, seed, scaling, save):
    """
    Plots the Sa(t) values (see fig2 in http://arxiv.org/abs/1806.08316)

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): Whether or not a central spin is present
        samples (int or array (int)): Number of times data points should be generated
        seed (int): random seed for reducible results
        scaling (string): scaling of coupling constant A by chain length
        save (string): filename, if data needs to be saved

    Returns:
        If save: data (list [time, hce_mean, yerrors]), None otherwise

    """
    if save:
        hce_means = np.empty((len(chain_length), len(As), len(B0), len(times)))
        hce_stds = np.empty((len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                hce = np.zeros((samples[i], times.size))
                for sample in range(samples[i]):
                    hce[sample] = calc_half_chain_entropy(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin, seed, scaling)
                hce_mean = np.mean(hce, axis=0)
                hce_std = np.std(hce, axis=0)
                yerrors = hce_std / np.sqrt(samples[i])
                if save:
                    hce_means[i, a, b] = hce_mean
                    hce_stds[i, a, b] = hce_std
                plt.plot(times, hce_mean, label=f"A={A}, L={N}, B={B}")
                plt.fill_between(times, hce_mean + yerrors, hce_mean - yerrors, alpha=0.2)
    plt.ylim(0, 3.5)
    # plt.hlines(
    #     np.log(2**((chain_length[0] + central_spin)//2)), times[0], times[-1], color='black')
    plt.xlabel("Time in fs")
    plt.ylabel("Half chain entropy")
    plt.semilogx()
    if save:
        return [times, hce_means, hce_stds]


def plot_single_shot_half_chain_entropy(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                                        central_spin, samples, seed, scaling, save):
    """
    Plots the Sa(t) values (see fig2 in http://arxiv.org/abs/1806.08316)

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (array (int)): the length of the spin chain
        J (float): the coupling constant
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): Whether or not a central spin is present
        samples (int or array (int)): Number of times data points should be generated
        seed (int): random seed for reducible results
        scaling (string): scaling of coupling constant A by chain length
        save (string): filename, if data needs to be saved

    Returns:
        If save: data (list [time, hce_mean, yerrors]), None otherwise

    """
    hces = np.empty((samples[0], len(times)))
    for sample in range(samples[0]):
        hces[sample] = calc_half_chain_entropy(
            times, chain_length[0], J, J_xy, B0[0], As[0], periodic_boundaries, central_spin,
            sample+1, scaling)
        plt.plot(times, hces[sample], label=f"seed={sample+1}")
    # plt.hlines(
    #     np.log(2**((chain_length[0] + central_spin)//2)), times[0], times[-1], color='black')
    plt.ylim(0, 3.5)
    plt.xlabel("Time in fs")
    plt.ylabel("Half chain entropy")
    plt.semilogx()
    plt.legend()
    if save:
        return [times, hces]


def calc_occupation_imbalance(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                              central_spin, seed, scaling):
    """
    Calculates the occupation imbalance sum_odd s_z - sum_even s_z

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

    Returns:
        occupation_imbalance (array (float) [times])
    """
    total_spins = chain_length + central_spin
    dim = int(2**total_spins)
    eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
        n_up=total_spins//2, seed=seed, scaling=scaling)
    psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
    # This mask filters out the states of the biggest subspace
    subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))[0]
    psi_z = np.arange(0, np.int(2**(total_spins)))[subspace_mask]
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)
    # discard central spin in exp_sig_z
    if central_spin:
        sigma_z = sigma_z[:, :-1]
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    # occupation imbalance mask: even minus odd sites
    occ_imbalance = np.where(np.arange(chain_length) % 2, exp_sig_z, -exp_sig_z).sum(axis=1)
    # and norm it to 1
    return occ_imbalance / (chain_length / 2)


def plot_occupation_imbalance(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                              central_spin, samples, seed, scaling, save):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        If save: data (list [time, occupation_imbalance_means, occupation_imbalance_stds]),
                 None otherwise

    """
    if save:
        occupation_imbalance_means = np.empty((len(chain_length), len(As), len(B0), len(times)))
        occupation_imbalance_stds = np.empty((len(chain_length), len(As), len(B0), len(times)))
    if len(samples) == 1:
        samples = samples * len(chain_length)
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                occupation_imbalance = np.zeros((samples[i], times.size))
                for sample in range(samples[i]):
                    occupation_imbalance[sample] = calc_occupation_imbalance(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin,
                        seed=sample+1, scaling=scaling)
                occupation_imbalance_mean = occupation_imbalance.mean(axis=0)
                occupation_imbalance_std = occupation_imbalance.std(axis=0)
                yerrors = occupation_imbalance.std(axis=0) / np.sqrt(samples[i])
                if save:
                    occupation_imbalance_means[i, a, b] = occupation_imbalance_mean
                    occupation_imbalance_stds[i, a, b] = occupation_imbalance_std
                plt.plot(times, occupation_imbalance_mean, label=f"N={N}")
                plt.fill_between(times, occupation_imbalance_mean + yerrors,
                                 occupation_imbalance_mean - yerrors, alpha=0.2)
    # plt.title(f"Occupation imbalance for \nJ={J}, B={B}, A={A}, scaling={scaling}")
    plt.ylim(-0.2, 1.02)
    plt.xlabel("Time in fs")
    plt.semilogx()
    plt.ylabel("Occupation imbalance")
    plt.legend(loc=1)
    if save:
        return [times, occupation_imbalance_means, occupation_imbalance_stds]


def plot_single_shot_occupation_imbalance(times, chain_length, J, J_xy, B0, As,
                                          periodic_boundaries, central_spin, samples, seed,
                                          scaling, save):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

    Args:
        rho0 (array (float) [dim, dim]): the initial density matrix, where dim = 2**total_spins
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whether or not a central spin is present
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        If save: data (list [time, occupation_imbalance_means, occupation_imbalance_stds]),
                 None otherwise

    """
    occupation_imbalances = np.empty((samples[0], len(times)))
    if len(samples) == 1:
        samples = samples * len(chain_length)
    for sample in range(samples[0]):
        occupation_imbalances[sample] = calc_occupation_imbalance(
            times, chain_length[0], J, J_xy, B0[0], As[0], periodic_boundaries, central_spin,
            seed=sample+1, scaling=scaling)
        plt.plot(times, occupation_imbalances[sample], label=f"Seed={sample+1}")
    plt.ylim(-0.2, 1.02)
    plt.xlabel("Time in fs")
    plt.semilogx()
    plt.ylabel("OI")
    plt.legend(loc=1)
    if save:
        return [times, occupation_imbalances]


def calc_exp_sig_z_central_spin(times, chain_length, J, J_xy, B0, A, periodic_boundaries, seed, scaling):
    """
    Calculates the expectation value for the central spin.

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        exp_sig_z (array (float) [times]): the expectation value for the central spin

    """
    total_spins = chain_length + 1
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))
    eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin=True,
        n_up=total_spins//2, seed=seed, scaling=scaling)
    psi_t = time_evo_subspace(times, eigenvalues, eigenvectors, total_spins)
    psi_z = np.arange(0, np.int(2**(total_spins)))[subspace_mask]
    # only the last spin
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, -1]
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    return exp_sig_z


def plot_single_shot_exp_sig_z_central_spin(times, chain_length, J, J_xy, B0, As,
                                            periodic_boundaries, samples, seed, scaling, save):
    """
    Plots the expectation value for the central spin.

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        If save: data (list [time, exp_sig_z_means, exp_sig_z_errors]), None otherwise

    """
    # for saving the data

    exp_sig_zs = np.empty((samples[0], len(times)))
    if len(samples) == 1:
        samples = samples * len(chain_length)
    for sample in range(samples[0]):
        exp_sig_zs[sample] = calc_exp_sig_z_central_spin(
            times, chain_length[0], J, J_xy, B0[0], As[0], periodic_boundaries,
            seed=sample+1, scaling=scaling)
        plt.plot(times, exp_sig_zs[sample], label=f"Seed={sample+1}")
    plt.xlabel("Time in fs")
    plt.ylabel(r"$<S_z>$")
    plt.semilogx()
    plt.legend(loc=1)
    if save:
        return [times, exp_sig_zs]


def plot_exp_sig_z_central_spin(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                                samples, seed, scaling, save):
    """
    Plots the expectation value for the central spin.

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (array (int)[1]): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        If save: data (list [time, exp_sig_z_means, exp_sig_z_errors]), None otherwise

    """
    # for saving the data
    if save:
        exp_sig_z_means = np.empty(
            (len(chain_length), len(As), len(B0), len(times)))
        exp_sig_z_stds = np.empty(
            (len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                exp_sig_z = np.zeros((samples[i], times.size))
                for sample in range(samples[i]):
                    exp_sig_z[sample] = calc_exp_sig_z_central_spin(
                        times, N, J, J_xy, B, A, periodic_boundaries, seed, scaling)
                exp_sig_z_mean = exp_sig_z.mean(axis=0)
                exp_sig_z_std = exp_sig_z.std(axis=0)
                yerrors = exp_sig_z_std / np.sqrt(samples[i])
                if save:
                    exp_sig_z_means[i, a, b] = exp_sig_z_mean
                    exp_sig_z_stds[i, a, b] = exp_sig_z_std
                plt.plot(times, exp_sig_z_mean, label=f"N={N}")
                plt.fill_between(times, exp_sig_z_mean + yerrors,
                                 exp_sig_z_mean - yerrors, alpha=0.2)
    if len(As) == 1 and len(B0) == 1:
        plt.title(f"Expectation value of the central spin for \nJ={J}, A={As[0]}, B={B0[0]}")
    plt.xlabel("Time in fs")
    plt.ylabel(r"$<S_z>$")
    plt.semilogx()
    plt.legend(loc=1)
    if save:
        return [times, exp_sig_z_means, exp_sig_z_stds]


def calc_correlation(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     seed, scaling):
    """
    Calculates the correlation function sigma^2(t) from
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.114.100601
    (in comments, G_r(t) from page 7 of http://arxiv.org/abs/1610.08993)

    For more details on the implementation see Notes/Correlation_function.xopp

    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        Correlation sigma^2(t) (array (float) [chain_length, times])
    # Before: G_r(t) (array (float) [chain_length, times])

    """
    total_spins = chain_length + 1
    dim = np.int(2**total_spins)
    # This mask filters out the states of the biggest subspace
    subspace_mask = np.where(np.logical_not(np.sum(sf.unpackbits(np.arange(dim), total_spins),
                                                   axis=1) - total_spins//2))[0]
    eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin=True,
        n_up=total_spins//2, seed=seed, scaling=scaling)
    psi_z = np.arange(0, np.int(2**(total_spins)))[subspace_mask]
    # discard the central spin from sigma_z
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, :chain_length]
    psi_0 = np.zeros(dim)
    # # Initialize in Neel state
    # psi_0[sf.packbits(np.arange(total_spins) % 2)] = 1
    # # Initialize in 'domain wall state' |1 1 (...) 1 0 0 (...) 0>
    # psi_0 = np.zeros(dim)
    # unpacked_psi_0 = np.concatenate((np.ones(total_spins // 2),
    #                                  np.zeros(total_spins - total_spins // 2))).astype(np.int)
    # psi_0[sf.packbits(unpacked_psi_0)] = 1
    # Initialize in one spin up state: |1 0 (...) 0>
    psi_0 = psi_0[subspace_mask]
    exp_part = np.exp(1j * np.outer(times, eigenvalues) / hbar * e * 1e-15)
    psi_t = eigenvectors @ (exp_part.reshape(times.size, eigenvalues.size, 1)
                            * eigenvectors.T) @ psi_0
    S_0 = psi_0 @ sigma_z
    S_t = np.abs(psi_t)**2 @ sigma_z
    # Previously: G_r(t)
    # G = np.empty((chain_length, times.size))
    # for r in range(chain_length):
    #     # 4 * to 'norm' it to [-1, 1]
    #     G[r] = 4 * (np.roll(S_t, r, axis=1) * S_0).mean(axis=1)
    n = np.arange(0, chain_length)  # First entry is zero anyways
    return (n**2 * (S_t * S_0[0] - S_0 * S_0[0])).mean(axis=1)


def plot_correlation(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                     samples, seed, scaling, save):
    """
    Plots the correlation function sigma_squared(t) from page 7 of
    http://arxiv.org/abs/1610.08993


    Args:
        times (array (float) [tD]): the time array, where g should be calculated
        chain_length (int or array (int)): the length of the spin chain
        J (float): Spin chain coupling in z-direction
        J_xy (float): Spin chain coupling in xy-direction
        B0 (float or array (float)): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        As (float or array (float)): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (array (int)): Number of times data points should be generated
        seed (int): use a seed to produce comparable outcomes if False, then it is initialized
                    randomly

    Returns:
        If save: data (list [time, sigma_squared_means, sigma_squared_errors]), None otherwise

    """
    # for saving the data
    if save:
        # sigma_squareds for different chain length have different sizes,
        # I have to put them in a list.
        sigma_squared_means = np.empty((len(chain_length), len(As), len(B0), len(times)))
        sigma_squared_stds = np.empty((len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                sigma_squareds = np.zeros((samples[i], times.size))
                for sample in range(samples[i]):
                    sigma_squareds[sample] = calc_correlation(
                        times, N, J, J_xy, B, A, periodic_boundaries, seed, scaling)
                sigma_squared_mean = sigma_squareds.mean(axis=0)
                sigma_squared_std = sigma_squareds.std(axis=0)
                if save:
                    sigma_squared_means[i][a][b] = sigma_squared_mean
                    sigma_squared_stds[i][a][b] = sigma_squared_std
                plt.plot(times, sigma_squared_mean, label=f"N={N}")
                plt.fill_between(times, sigma_squared_mean + sigma_squared_std,
                                 sigma_squared_mean - sigma_squared_std, alpha=0.2)
    if len(As) == 1 and len(B0) == 1:
        plt.title(f"Correlation sigma^2(t) for \nJ={J}, A={As[0]}, B={B0[0]}")
    plt.xlabel("Time in fs")
    plt.ylabel(r"Correlation $\sigma^2(t)$")
    plt.semilogx()
    plt.legend(loc=1)
    if save:
        return [times, sigma_squared_means, sigma_squared_stds]
