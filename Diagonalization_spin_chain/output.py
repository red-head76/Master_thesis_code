from matplotlib import animation
from scipy.constants import hbar, e
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
import support_functions as sf
import diagonalization as diag
from time_evo import time_evo_sigma_z, time_evo

# Doesn't work on justus, so commented out
# import seaborn as sns
# sns.set_theme(context="paper")

"""
Common arguments for plot functions:
* t (array (float)): array with tN timesteps
* idx_psi_0 (int): the index of the state at t0
* chain_length (int or list(int)): the length of the spin chain
* J (float): Spin chain coupling in z-direction
* J_xy (float): Spin chain coupling in xy-direction
* B0 (float or list(float)): the B-field amplitude. Currently random initialized uniformly
                             between (-1, 1).
* A (float or list(float)): the coupling between the central spin and the spins in the chain
* periodic_boundaries (bool): determines whether or not periodic boundary
                              conditions are used in the chain.
* central_spin (bool): determines whether or not a central spin, coupling
                                  to all other spins is used or not
* seed (int): If nonzero, the given integer is used as seed
* scaling (string): scaling of A_0 ('sqrt', 'inverse' or 'none')
* save (string): If not False, the output is saved with the given filename.
"""


def plot_time_evo(t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, seed,
                  scaling, save, initial_state):
    """
    Plots the time evolution of the spin chain and the optional central spin

    Returns:
        If save: data (list [time, exp_sig_z]), None otherwise
    """
    total_spins = central_spin + chain_length
    exp_sig_z = time_evo_sigma_z(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                                 central_spin, seed, scaling, initial_state)
    fig, ax = plt.subplots(total_spins, 1, figsize=(5, (1 + total_spins)),
                           sharex=True, tight_layout=True)
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

    # fig, ax = plt.subplots(figsize=(8, 6), sharex=True, tight_layout=True)
    # for i in range(1):
    #     if i == total_spins and central_spin:
    #         ax.plot(t, exp_sig_z.T[i], color="black", label="central_spin")
    #         ax.legend()
    #     else:
    #         ax.plot(t, exp_sig_z.T[i])

    if save:
        return [t, exp_sig_z]


def plot_light_cone(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                    central_spin, seed, scaling, save, initial_state):
    exp_sig_z = time_evo_sigma_z(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                                 central_spin, seed, scaling, initial_state)
    plt.imshow(exp_sig_z, aspect="auto", interpolation="none")
    plt.yticks(np.arange(0, t.size, t.size//10),
               t[np.where(np.arange(t.size) % (t.size//10) == 0)].astype(int))
    if save:
        return [t, exp_sig_z]


def animate_time_evo(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     central_spin, seed, scaling, save, initial_state):
    """
    Animate the time evolution of the spin chain and the optional central spin

    Returns:
        If save: data (list [time, exp_sig_z]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    exp_sig_z = time_evo_sigma_z(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                                 central_spin, seed, scaling, initial_state)
    total_spins = chain_length + central_spin
    if seed:
        np.random.seed(seed)
    B = np.random.uniform(-1, 1, chain_length) * min(B0, 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    # Stem container containing markerline, stemlines, baseline
    stem_container = ax.stem(
        np.arange(total_spins), exp_sig_z[0], use_line_collection=True)
    ax.step(np.arange(chain_length), B, color="C2", where="mid")
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

    anim = animation.FuncAnimation(fig, run, frames=t.size, blit=True, interval=20)
    if save:
        return [t, exp_sig_z, anim]
    else:
        return [anim]


def animate_barplot(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                    central_spin, seed, scaling, save, initial_state):
    """
    Animate the time evolution of the spin chain and the optional central spin

    Returns:
        If save: data (list [time, exp_sig_z]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    exp_sig_z = time_evo_sigma_z(t, chain_length, J, J_xy, B0, A, periodic_boundaries,
                                 central_spin, seed, scaling, initial_state)
    total_spins = chain_length + central_spin
    if seed:
        np.random.seed(seed)
    B = np.random.uniform(-1, 1, chain_length) * min(B0, 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-0.5, total_spins + 0.5)
    ax.axis("off")
    barcol = ax.bar(np.arange(total_spins), np.sqrt(exp_sig_z[0] + 0.5))

    def run(i):
        for j, b in enumerate(barcol):
            # sqrt just for optical scaling
            b.set_height(np.sqrt(exp_sig_z[i, j] + 0.5))
        return barcol
    anim = animation.FuncAnimation(fig, run, frames=t.size, blit=True, interval=20)
    if save:
        return [t, exp_sig_z, anim]
    else:
        return [anim]


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


def generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, initial_state):
    """
    Calculates the r value, the fraction of the difference of eigenvalues of the given Hamiltonian:
    r = min (ΔE_n, ΔE_n+1) / max (ΔE_n, ΔE_n+1)

    Returns:
        r_values (array (float) [2**total_spins - 2])

    """
    # Get the energy eigenvalues
    total_spins = chain_length + central_spin
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    E = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, n_up)[0]

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
                  samples, initial_state, scaling="sqrt", save=False):
    """
    Plots the histogram of r_values created by the given parameters.

    Returns:
        If save: data (list [r_values]), None otherwise

    """

    r_values = generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                                 initial_state)
    for _ in range(samples - 1):
        r_values += generate_r_values(chain_length, J, J_xy, B0, A, periodic_boundaries,
                                      central_spin, initial_state)
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


def plot_r_fig3(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, initial_state,
                samples, scaling="sqrt", save=False):
    """
    Plots the r values as done in Figure 3 in https://doi.org/10.1103/PhysRevB.82.174411
    initial_state is only used to calculate the right subspace.
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
                                         central_spin, initial_state)
            for _ in range(samples[i] - 1):
                r_values += generate_r_values(N, J, J_xy, B, A[0], periodic_boundaries,
                                              central_spin, initial_state)
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
                            central_spin, seed, scaling, initial_state="neel"):
    """
    Calculates the half chain entropy -tr_a(rho_a, ln(rho_a))

    Returns:
        half_chain_entropy (array (float) [times])
    """
    total_spins = chain_length + central_spin
    dim = np.int(2**total_spins)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     central_spin, seed, scaling, initial_state)
    splitted_subspace_mask = np.split(sf.unpackbits(subspace_mask, total_spins), [total_spins//2],
                                      axis=1)
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
                            central_spin, samples, seed, scaling, save, initial_state="neel"):
    """
    Plots the Sa(t) values (see fig2 in http://arxiv.org/abs/1806.08316)

    Returns:
        If save: data (list [time, hce_mean, yerrors]), None otherwise

    """
    if save:
        hce_means = np.empty((len(chain_length), len(As), len(B0), len(times)))
        hce_stds = np.empty((len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                hce = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    hce[sample] = calc_half_chain_entropy(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin, seed, scaling,
                        initial_state)
                hce_mean = np.mean(hce, axis=0)
                hce_std = np.std(hce, axis=0)
                yerrors = hce_std  # / np.sqrt(samples[i])
                if save:
                    hce_means[i, a, b] = hce_mean
                    hce_stds[i, a, b] = hce_std
                plt.plot(times, hce_mean, label=f"B={B}")
                plt.fill_between(times, hce_mean + yerrors, hce_mean - yerrors, alpha=0.2)
    # plt.hlines(
    #     np.log(2**((chain_length[0] + central_spin)//2)), times[0], times[-1], color='black')
    plt.xlabel("Time in fs")
    plt.ylabel("Half chain entropy")
    plt.semilogx()
    plt.legend()
    if save:
        return [times, hce_means, hce_stds]


def plot_single_shot_half_chain_entropy(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                                        central_spin, samples, scaling, save, initial_state):
    """
    Plots the Sa(t) values (see fig2 in http://arxiv.org/abs/1806.08316)

    Returns:
        If save: data (list [time, hce_mean, yerrors]), None otherwise
    """
    hces = np.empty((samples[0], len(times)))
    for sample in range(samples[0]):
        hces[sample] = calc_half_chain_entropy(
            times, chain_length[0], J, J_xy, B0[0], As[0], periodic_boundaries, central_spin,
            sample+1, scaling, initial_state)
        plt.plot(times, hces[sample], label=f"seed={sample+1}")
    # plt.hlines(
    #     np.log(2**((chain_length[0] + central_spin)//2)), times[0], times[-1], color='black')
    plt.xlabel("Time in fs")
    plt.ylabel("Half chain entropy")
    plt.semilogx()
    plt.legend()
    if save:
        return [times, hces]


def calc_occupation_imbalance(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                              central_spin, seed, scaling, initial_state):
    """
    Calculates the occupation imbalance sum_odd s_z - sum_even s_z

    Returns:
        occupation_imbalance (array (float) [times])
    """
    total_spins = chain_length + central_spin
    dim = int(2**total_spins)

    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     central_spin, seed, scaling, initial_state)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
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
                              central_spin, samples, seed, scaling, save, initial_state):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

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
                occupation_imbalance = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    occupation_imbalance[sample] = calc_occupation_imbalance(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin,
                        seed, scaling, initial_state)
                occupation_imbalance_mean = occupation_imbalance.mean(axis=0)
                occupation_imbalance_std = occupation_imbalance.std(axis=0)
                yerrors = occupation_imbalance.std(axis=0)  # / np.sqrt(samples[i])
                if save:
                    occupation_imbalance_means[i, a, b] = occupation_imbalance_mean
                    occupation_imbalance_stds[i, a, b] = occupation_imbalance_std
                plt.plot(times, occupation_imbalance_mean, label=f"B={B}")
                plt.fill_between(times, occupation_imbalance_mean + yerrors,
                                 occupation_imbalance_mean - yerrors, alpha=0.2)
    plt.xlabel("Time in fs")
    plt.semilogx()
    plt.ylabel("Occupation imbalance")
    plt.legend(loc=1)
    if save:
        return [times, occupation_imbalance_means, occupation_imbalance_stds]


def plot_occupation_imbalance_plateau(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                                      central_spin, samples, seed, scaling, save, initial_state):
    """
    Plots the plateau values for given As

    Returns:
        If save: data (list [As, plateau_means, plateau_stds]), None otherwise
    """
    if save:
        occupation_imbalance_means = np.empty((len(chain_length), len(As), len(B0)))
        occupation_imbalance_stds = np.empty((len(chain_length), len(As), len(B0)))
    if len(samples) == 1:
        samples = samples * len(chain_length)
    for i, N in enumerate(chain_length):
        for b, B in enumerate(B0):
            occupation_imbalance = np.empty((samples[i], times.size))
            occupation_imbalance_mean = np.empty((len(As)))
            occupation_imbalance_std = np.empty((len(As)))
            for a, A in enumerate(As):
                for sample in range(samples[i]):
                    occupation_imbalance[sample] = calc_occupation_imbalance(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin,
                        seed, scaling, initial_state)
                # Mean and std over time and samples
                occupation_imbalance_mean[a] = occupation_imbalance.mean()
                occupation_imbalance_std[a] = occupation_imbalance.std()
            if save:
                occupation_imbalance_means[i, :, b] = occupation_imbalance_mean
                occupation_imbalance_stds[i, :, b] = occupation_imbalance_std
            plt.errorbar(As, occupation_imbalance_mean, occupation_imbalance_std,
                         capsize=2, label=f"L={N}, W={B}")
    plt.xlabel("Time in fs")
    plt.ylabel("Plateau value of occupation imbalance")
    plt.legend(loc=1)
    if save:
        return [occupation_imbalance_means, occupation_imbalance_stds]


def plot_occupation_imbalance_plateau_linfit(times, chain_length, J, J_xy, B0, As,
                                             periodic_boundaries, central_spin, samples, seed,
                                             scaling, save, initial_state):
    """
    Plots the plateau values for given As

    Returns:
        If save: data (list [As, plateau_means, plateau_stds]), None otherwise
    """
    if save:
        slopes_means = np.empty((len(chain_length), len(As), len(B0)))
        slopes_errors = np.empty((len(chain_length), len(As), len(B0)))
    if len(samples) == 1:
        samples = samples * len(chain_length)
    for i, N in enumerate(chain_length):
        for b, B in enumerate(B0):
            slopes = np.empty((samples[i]))
            slope_errors = np.empty((samples[i]))
            slopes_mean = np.empty((len(As)))
            slope_errors_mean = np.empty((len(As)))
            for a, A in enumerate(As):
                for sample in range(samples[i]):
                    occupation_imbalance = calc_occupation_imbalance(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin,
                        seed, scaling, initial_state)
                    # Make linfit check
                    result = linregress(times, occupation_imbalance)
                    slopes[sample] = result.slope
                    slope_errors[sample] = result.slope
                slopes_mean[a] = slopes.mean()
                slope_errors_mean[a] = slope_errors.mean()
            if save:
                slopes_means[i, :, b] = slopes_mean
                slopes_errors[i, :, b] = slope_errors_mean
            plt.errorbar(As, slopes_mean, slope_errors_mean, capsize=2, label=f"L={N}, W={B}")
    plt.xlabel("Time in fs")
    plt.ylabel("Slope of occupation imbalance plateau values")
    plt.legend(loc=1)
    if save:
        return [slopes_means, slopes_errors]


def plot_single_shot_occupation_imbalance(times, chain_length, J, J_xy, B0, As,
                                          periodic_boundaries, central_spin, samples, seed,
                                          scaling, save, initial_state):
    """
    Plots the occupation imbalance sum_odd s_z - sum_even s_z

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
            seed, scaling, initial_state)
        plt.plot(times, occupation_imbalances[sample], label=f"Seed={sample+1}")
    plt.xlabel("Time in fs")
    plt.semilogx()
    plt.ylabel("OI")
    plt.legend(loc=1)
    if save:
        return [times, occupation_imbalances]


def calc_exp_sig_z_central_spin(times, chain_length, J, J_xy, B0, A, periodic_boundaries, seed,
                                scaling, initial_state):
    """
    Calculates the expectation value for the central spin.

    Returns:
        exp_sig_z (array (float) [times]): the expectation value for the central spin

    """
    total_spins = chain_length + 1
    dim = np.int(2**total_spins)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     True, seed, scaling, initial_state)
    psi_z = np.arange(0, np.int(2**(total_spins)))[subspace_mask]
    # only the last spin
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, -1]
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    return exp_sig_z


def plot_single_shot_exp_sig_z_central_spin(
        times, chain_length, J, J_xy, B0, As, periodic_boundaries, samples, seed, scaling,
        save, inital_state):
    """
    Plots the expectation value for the central spin.

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
            seed=sample+1, scaling=scaling, initial_state=inital_state)
        plt.plot(times, exp_sig_zs[sample], label=f"Seed={sample+1}")
    plt.xlabel("Time in fs")
    plt.ylabel(r"$<S_z>$")
    plt.semilogx()
    plt.legend(loc=1)
    if save:
        return [times, exp_sig_zs]


def plot_exp_sig_z_central_spin(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                                samples, seed, scaling, save, initial_state):
    """
    Plots the expectation value for the central spin.

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
                exp_sig_z = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    exp_sig_z[sample] = calc_exp_sig_z_central_spin(
                        times, N, J, J_xy, B, A, periodic_boundaries, seed, scaling, initial_state)
                exp_sig_z_mean = exp_sig_z.mean(axis=0)
                exp_sig_z_std = exp_sig_z.std(axis=0)
                yerrors = exp_sig_z_std  # / np.sqrt(samples[i])
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


def calc_correlation(times, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                     seed, scaling, initial_state):
    """
    Calculates the correlation function sigma^2(t) from
    https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.114.100601
    (in comments, G_r(t) from page 7 of http://arxiv.org/abs/1610.08993)

    For more details on the implementation see Notes/Correlation_function.xopp
    Returns:
        Correlation sigma^2(t) (array (float) [chain_length, times])
    # Before: G_r(t) (array (float) [chain_length, times])

    """
    total_spins = chain_length + 1
    dim = np.int(2**total_spins)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    psi_0 = np.zeros(dim)
    psi_0[idx_psi_0] = 1
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_0 = psi_0[subspace_mask]
    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     central_spin, seed, scaling, initial_state)
    psi_z = np.arange(0, np.int(2**(total_spins)))[subspace_mask]
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, :chain_length]
    S_0 = psi_0 @ sigma_z
    S_t = np.abs(psi_t)**2 @ sigma_z
    # Previously: G_r(t)
    # G = np.empty((chain_length, times.size))
    # for r in range(chain_length):
    #     # 4 * to 'norm' it to [-1, 1]
    #     G[r] = 4 * (np.roll(S_t, r, axis=1) * S_0).mean(axis=1)
    n = np.arange(0, chain_length)  # First entry is zero anyways
    return (n**2 * (S_t * S_0[0] - S_0 * S_0[0])).mean(axis=1)


def plot_correlation(times, chain_length, J, J_xy, B0, As, periodic_boundaries, central_spin,
                     samples, seed, scaling, save, initial_state):
    """
    Plots the correlation function sigma_squared(t) from page 7 of
    http://arxiv.org/abs/1610.08993

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
                sigma_squareds = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    sigma_squareds[sample] = calc_correlation(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin, seed,
                        scaling, initial_state)
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


def plot_2_spin_up(t, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                   seed, scaling, save):
    """
    Initializes two spins up at pos_up1 and pos_up2, the rest spin down.
    Then the expectation value of Sz at these positions is added and plotted over time.

    Returns:
        If save: data (list [time, exp_sig_z_at_init]), None otherwise
    """
    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    pos_up1 = chain_length//2
    pos_up2 = chain_length//2 + 1
    idx_psi_0 = int(2**(pos_up1) + 2**pos_up2)
    exp_sig_z = time_evo_sigma_z(t, idx_psi_0, chain_length, J, J_xy, B0, A,
                                 periodic_boundaries, central_spin, seed, scaling)
    # Expectation values of Sz at the initial positions added
    if pos_up2 < pos_up1:
        pos_up1, pos_up2 = pos_up2, pos_up2
    exp_sig_z_at_init = exp_sig_z[:, pos_up1-1:pos_up2+2].sum(axis=1)
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(t, exp_sig_z_at_init)
    ax.set_xlabel("Time in fs")
    ax.set_ylabel("Sum $S_z$ at inital positions")
    ax.semilogx()
    plt.show()
    if save:
        return [t, exp_sig_z_at_init]


def plot_deff(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, samples,
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
    deff_means = np.empty(len(B0))
    deff_stds = np.empty(len(B0))
    distances_means = np.empty(len(B0))
    distances_stds = np.empty(len(B0))
    for j, B in enumerate(B0):
        eigenvalues = np.empty((samples, subspace_mask.size))
        eigenvectors = np.empty((samples, subspace_mask.size, subspace_mask.size))
        distances = np.empty(samples)
        for i in range(samples):
            eigenvalues[i], eigenvectors[i] = diag.eig_values_vectors_spin_const(
                chain_length, J, J_xy, B, A, periodic_boundaries, central_spin, n_up,
                seed, scaling)
        coeffs = eigenvectors.transpose(0, 2, 1) @ psi_0
        deff = 1 / np.sum(coeffs**4, axis=1)
        deff_means[j] = deff.mean()
        deff_stds[j] = deff.std()
    plt.errorbar(B0, deff_means, deff_stds, capsize=2, label=r"$d^{eff}$")
    plt.hlines((1, subspace_mask.size), plt.axis()[0], plt.axis()[1], color="black", ls=':')
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("Disorder W")
    if save:
        return [B0, deff_means, deff_stds]


def plot_eigenstates(chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, seed,
                     scaling, save, initial_state):
    """
    Plots the lowest eigenvectors in real space

    Returns:
        If save: data (array eigenvectors_amplitude), None otherwise
    """
    # amount of eigenstates shown
    n_eigenvectors = 5

    total_spins = central_spin + chain_length
    dim = np.array(2**total_spins, dtype=np.int)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    eigenvalues, eigenvectors = diag.eig_values_vectors_spin_const(
        chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin, n_up, seed)
    subspace = sf.calc_subspace(total_spins, n_up)
    basis = sf.unpackbits(subspace, total_spins)
    eigenvectors_amplitude = np.abs(eigenvectors[:n_eigenvectors]**2) @ basis
    fig, ax = plt.subplots(tight_layout=True)
    # print(eigenvalues[:n_eigenvectors])
    for i in range(n_eigenvectors):
        plt.plot(np.arange(total_spins), (eigenvectors_amplitude[i]))
    ax.set_xlabel("Site")
    ax.set_ylabel("Amplitude")
    if save:
        return [eigenvectors_amplitude]


def calc_exp_sig_z_single_spin(times, chain_length, J, J_xy, B0, A, periodic_boundaries, seed,
                               scaling, initial_state):
    """
    Calculates the expectation value for a single spin for
    periodic_boundaries = True
    initial_state = 1

    Returns:
        exp_sig_z (array (float) [times]): the expectation value for the spin

    """
    periodic_boundaries = True
    initial_state = "1"
    total_spins = chain_length + 1
    dim = np.int(2**total_spins)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     True, seed, scaling, initial_state)
    psi_z = subspace_mask
    # only one spin
    sigma_z = (sf.unpackbits(psi_z, total_spins) - 1/2)[:, 0]
    exp_sig_z = (np.abs(psi_t)**2 @ sigma_z)
    return exp_sig_z


def plot_exp_sig_z_single_spin(times, chain_length, J, J_xy, B0, As, periodic_boundaries, samples,
                               seed, scaling, save, initial_state):
    """
    Plots the expectation value for a single spin for
    periodic_boundaries = True
    initial_state = 1


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
                exp_sig_z = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    exp_sig_z[sample] = calc_exp_sig_z_single_spin(
                        times, N, J, J_xy, B, A, periodic_boundaries, seed, scaling, initial_state)
                exp_sig_z_mean = exp_sig_z.mean(axis=0)
                exp_sig_z_std = exp_sig_z.std(axis=0)
                yerrors = exp_sig_z_std  # / np.sqrt(samples[i])
                if save:
                    exp_sig_z_means[i, a, b] = exp_sig_z_mean
                    exp_sig_z_stds[i, a, b] = exp_sig_z_std
                plt.plot(times, exp_sig_z_mean, label=f"N={N}")
                plt.fill_between(times, exp_sig_z_mean + yerrors,
                                 exp_sig_z_mean - yerrors, alpha=0.2)
    if len(As) == 1 and len(B0) == 1:
        plt.title(f"Expectation value of the single spin for \nJ={J}, A={As[0]}, B={B0[0]}")
    plt.xlabel("Time in fs")
    plt.ylabel(r"$<S_z>$")
    plt.semilogx()
    plt.legend(loc=1)
    if save:
        return [times, exp_sig_z_means, exp_sig_z_stds]


def calc_single_spin_entropy(times, chain_length, J, J_xy, B0, A, periodic_boundaries, central_spin,
                             seed, scaling, initial_state):
    """
    Calculates the single spin entropy -tr_a(rho_a, ln(rho_a)) for 
    periodic_boundaries = True
    initial_state = 1

    Returns:
        half_chain_entropy (array (float) [times])

    """
    periodic_boundaries = True
    initial_state = "1"
    total_spins = chain_length + central_spin
    dim = np.int(2**total_spins)
    idx_psi_0 = sf.calc_idx_psi_0(initial_state, total_spins)
    n_up = sf.unpackbits(idx_psi_0, total_spins).sum()
    subspace_mask = sf.calc_subspace(total_spins, n_up)
    psi_t = time_evo(times, chain_length, J, J_xy, B0, A, periodic_boundaries,
                     central_spin, seed, scaling, initial_state)
    splitted_subspace_mask = np.split(sf.unpackbits(subspace_mask, total_spins), [2],
                                      axis=1)
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


def plot_single_spin_entropy(times, chain_length, J, J_xy, B0, As, periodic_boundaries,
                             central_spin, samples, seed, scaling, save, initial_state):
    """
    Plots the single spin entropy -tr_a(rho_a, ln(rho_a)) for
    periodic_boundaries = True
    initial_state = 1

    Returns:
        If save: data (list [time, sse_mean, yerrors]), None otherwise

    """
    if save:
        sse_means = np.empty((len(chain_length), len(As), len(B0), len(times)))
        sse_stds = np.empty((len(chain_length), len(As), len(B0), len(times)))
    for i, N in enumerate(chain_length):
        for a, A in enumerate(As):
            for b, B in enumerate(B0):
                sse = np.empty((samples[i], times.size))
                for sample in range(samples[i]):
                    sse[sample] = calc_single_spin_entropy(
                        times, N, J, J_xy, B, A, periodic_boundaries, central_spin, seed, scaling,
                        initial_state)
                sse_mean = np.mean(sse, axis=0)
                sse_std = np.std(sse, axis=0)
                yerrors = sse_std  # / np.sqrt(samples[i])
                if save:
                    sse_means[i, a, b] = sse_mean
                    sse_stds[i, a, b] = sse_std
                plt.plot(times, sse_mean, label=f"B={B}")
                plt.fill_between(times, sse_mean + yerrors, sse_mean - yerrors, alpha=0.2)
    plt.xlabel("Time in fs")
    plt.ylabel("Half chain entropy")
    plt.semilogx()
    plt.legend()
    if save:
        return [times, sse_means, sse_stds]
