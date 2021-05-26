import numpy as np
import matplotlib.pyplot as plt
import os
from time_evo import time_evo_sigma_z
from diagonalization import eig_values_vectors, eig_values_vectors_spin_const
from matplotlib import animation

# Plotting time evolution of the spin chain


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
        J (float, default: 2): the coupling constant
        B0 (float, default: 1): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float, default: 1): the coupling between the central spin and the spins in the chain
        periodic_boundaries (bool, default:True): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        samples (int, default: 1): Number of times data points should be generated for each number
                                    of samples there are (chain_length x chain_length - 2) data
                                    points
        s_const (bool, default: True): If true, the conservation of total spin is used to construct
                                       respective subspaces. If False, full Hamiltonian is used.
        central_spin (bool, default=True): determines whether or not a central spin is present

    Returns:
        r_values (array (float) [2**total_spins - 2])

    """
    # Get the energy eigenvalues
    if spin_constant:
        if central_spin:
            raise AttributeError(
                "With a central spin, total spin is not conserved in the chain. Therefore central_spin and spin_constant cannot be used at the same time.")
        else:
            E = eig_values_vectors_spin_const(
                chain_length, J, B0, periodic_boundaries)[0]
    else:
        E = eig_values_vectors(
            chain_length, J, B0, A, periodic_boundaries, central_spin)[0]

    E = np.sort(E)
    Delta_E = np.diff(E)
    Delta_E_shifted = np.roll(Delta_E, 1)
    Delta_min = np.min((Delta_E, Delta_E_shifted), axis=0)[1:]
    Delta_max = np.max((Delta_E, Delta_E_shifted), axis=0)[1:]
    # Calculate the distribution approximately with a histogram
    # hist, bin_edges = np.histogram(Delta_min / Delta_max, bins=binning_func(Delta_min.size),
    #                                density=True, range=(0, 1))
    # bin_centers = (bin_edges + np.roll(bin_edges, 1))[1:] / 2
    # # Calculate the expectation value
    # r_values[i] = np.mean(hist * bin_centers)
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

    total_spins = chain_length + central_spin
    rs = np.zeros(np.int(2**total_spins - 2))
    for _ in range(samples):
        rs += generate_r_values(chain_length, J, B0, A, periodic_boundaries, central_spin,
                                spin_constant, samples)
    # Average over samples
    rs_mean = rs / samples
    plt.hist(rs_mean, bins=sturge_rule(rs_mean.size), density=True)
    # Average over states
    plt.plot([], [], ls="", label=f"Average r = {np.mean(rs_mean):.2f}")
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

    Returns:
        mean_r_values (array (float) [chain_length.size, B0.size]): The results of the r_value for
            the different realizations.
    """

    mean_r_values = np.empty((np.size(chain_length), np.size(B0)))
    if np.size(samples) == 1:
        samples = np.ones(np.size(chain_length), dtype=np.int) * samples
    for i, N in enumerate(chain_length):
        for j, B in enumerate(B0):
            r_values = np.zeros(np.int(2**chain_length - 2))
            for sample in samples:
                r_values += generate_r_values(N, J, B,
                                              0, periodic_boundaries, False, True)
            # Averaging over samples and states at the same time
            mean_r_values[i, j] = np.mean(r_values)

        plt.plot(B0, mean_r_values[i], marker="o",
                 linestyle="--", label=f"N={N}")
    plt.xlabel("Magnetic field amplitude B0")
    plt.ylabel("r-value")
    plt.legend()
    plt.show()