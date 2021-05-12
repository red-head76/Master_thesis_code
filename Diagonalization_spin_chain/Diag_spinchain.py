import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
# Animation stuff
from matplotlib import animation
from IPython.display import HTML

# Parameters
# ________________________________________________________________________________
chain_length = 8
# dimension of hilbert space
dim = np.int(2**chain_length)
# Coupling constant
J = 2
# Magnetic field
B0 = 1
B = np.round(np.random.uniform(-1, 1, chain_length), 2)
plot = False

# Helper functions
# ________________________________________________________________________________


def pc(matrix, precision=0):
    """
    print clean
    Just to print out a matrix cleanly without unreadable datajunk
    """
    with np.printoptions(precision=precision, suppress=True):
        print(matrix)


def create_basis_vectors(indices, dimension=dim):
    """
    Creates an array of basis vectors according to given indicees

    Args:
        indices (array [N]): an array containing N integers
        dimension (int, default: dim): the dimension of the vector space

    Returns:
        basis_vectors (array [N, dim]): an array containing N basis vectors.

    """
    basis_vectors = np.zeros((len(indices), dim))
    for row, index in enumerate(indices):
        basis_vectors[row, index] = 1
    return basis_vectors

# New definitions of packbits and unpackbits are required because np.unpackbits can only handle
# uint8. This means it is restricted to a chain_length of 8.


def unpackbits(x, num_bits=chain_length):
    """
    Similar to np.unpackbits, but can also handle longer uints than uint8
    From: https://stackoverflow.com/a/51509307
    The arrays get zero padded on the right, which means x=3, num_bits=4 returns:
    (1 1 0 0)

    Args:
        x (array [N]): input array with integers
        num_bits (int, default: chain_length): number of bits

    Returns:
        unpacked_bits (array [N, chain_length]): array of unpacked bits

    """
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def packbits(x):
    """
    Similar to np.packbits, but can also handle longer uints than uint8
    Example: packbits([1, 1, 0]) = 0 * 1 + 1 * 2 + 1 * 4 = 6

    Args:
        x (array [N, chain_length]): input array of unpacked bits
        num_bits (int, default: chain_length): number of bits

    Returns:
        packed_bits (array [N]): an array of integer

    """
    mask = 2**np.arange(len(x) - 1, -1, -1)
    return np.inner(mask, x)


# Setup
# ________________________________________________________________________________
psi_z = np.arange(0, np.int(2**chain_length))
# array of states in sigma_z basis
psi_z = np.arange(0, np.int(2**chain_length))
# sigma_z operator
sigma_z = unpackbits(psi_z) - 1/2


def eig_values_vectors(J=2, B0=1):
    """
    Computes the eigenvalues and eigenvectors for the Heisenberg Hamiltonian
    (H = Sum_i J * S_i * S_i+1 + B_i S_i^z).

    Args:
        J (float, default: 2): the coupling constant
        B0 (float, default: 1): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).

    Returns:
        eigenvalues (array [chain_length])
        eigenvectors (array [chain_length, chain_length])
    """

    H = np.zeros((dim, dim))
    # For every state
    for state_index in range(dim):
        state = unpackbits(psi_z[state_index])
        # Check interaction with every other state
        for i in range(1, chain_length):
            # Ising term in the hamiltonian: J * Sum(S_i^z * S_i+1^z)
            if state[i] == state[i-1]:
                H[state_index, state_index] += J/4
            else:
                H[state_index, state_index] -= J/4
                # Ladder operator terms: J/2 * Sum(S_i^+ S_i+1^- + S_i^- S_i+1^+)
                # Method: Flip spins and then add 1/2 in the according term in the hamiltonian
                # Only do this, if S_i^z != S_i+1^z, otherwise the ladder operators give 0.
                flipmask = np.roll(unpackbits(np.array(3)), i-1)
                flipped_state = packbits(np.logical_xor(state, flipmask))
                H[state_index, flipped_state] = J/2

        # Outer magnetic field term: Sum(B_i S_i^z)
        H[state_index,
            state_index] += np.sum(B0 * B * (state - 1/2))

    # Use eigh for the calculation, since H is hermitian -> I hope for better efficiency
    return np.linalg.eigh(H)


def eig_values_vectors_spin_const(J=2, B0=1):
    """
    Computes the eigenvalues and eigenvectors for the Heisenberg Hamiltonian
    (H = Sum_i J * S_i * S_i+1 + B_i S_i^z).
    This function makes use of the fact, that the Hamiltonian can be divided
    into subspaces according to the total number of spin-up states.

    Args:
        J (float, default: 2): the coupling constant
        B0 (float, default: 1): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).

    Returns:
        eigenvalues (array [chain_length])
        eigenvectors (array [chain_length, chain_length])
    """
    # Create subspaces
    subspaces = [0] * (chain_length + 1)
    for i in range(len(subspaces)):
        subspaces[i] = np.zeros((np.int(binom(chain_length, i))), dtype=np.int)

    # Fill subspaces
    # For every possible number of spin-up
    for n in range(len(subspaces)):
        # Check every psi to fit in that space
        sub_counter = 0
        for psi in psi_z:
            n_up = np.sum(unpackbits(psi))
            if n_up == n:
                subspaces[n][sub_counter] = psi
                sub_counter += 1

    eigenvalues = np.zeros(dim)
    eigenvectors = np.zeros((dim, dim))

    # Generate Hamiltonian for each subspace
    for psi_sub in subspaces:
        dim_sub = len(psi_sub)
        H_sub = np.zeros((dim_sub, dim_sub))
        # For every state
        for state_index in range(len(psi_sub)):
            state = unpackbits(psi_sub[state_index])
            for i in range(1, chain_length):
                # Ising term in the hamiltonian: J * Sum(S_i^z * S_i+1^z)
                if state[i] == state[i-1]:
                    H_sub[state_index, state_index] += J/4
                else:
                    H_sub[state_index, state_index] -= J/4
                    # Ladder operator terms: J/2 * Sum(S_i^+ S_i+1^- + S_i^- S_i+1^+)
                    # Method: Flip spins and then add 1/2 in the according term in the hamiltonian
                    # Only do this, if S_i^z != S_i+1^z, otherwise the ladder operators give 0.
                    flipmask = np.roll(unpackbits(np.array(3)), i-1)
                    flipped_state = packbits(np.logical_xor(state, flipmask))
                    sub_index_flipped_state = np.where(
                        psi_sub == flipped_state)[0][0]
                    H_sub[state_index, sub_index_flipped_state] = J/2
            # Outer magnetic field term: Sum(B_i S_i^z)
            H_sub[state_index,
                  state_index] += np.sum(B0 * B * (state - 1/2))

        # Diagonalization of subspace
        eigenvalues_sub, eigenvectors_sub = np.linalg.eigh(H_sub)
        # Enter this into full space
        eigenvalues[psi_sub] = eigenvalues_sub
        # Generate eigenvector with respect to full space basis
        eigenvectors_fullspace = eigenvectors_sub @ create_basis_vectors(
            psi_sub)
        eigenvectors[psi_sub] = eigenvectors_fullspace

    return eigenvalues, eigenvectors


def time_evo_sigma_z(t, psi0, J=2, B0=1, spin_constant=True):
    """
    Computes the time evolution of the spin operator sigma_z for the given state

    Args:
        t (array [tN]): array with tN timesteps
        psi0 (array [N]): the state at t0
        spin_constant (bool): If true, than the function eig_values_vectors_spin_const is used,
                              eig_values_vectors otherwise.
        J (float, default: 2): the coupling constant
        B0 (float, default: 1): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).

    Returns:
        exp_sig_z (array [tN, N]): Array of expectation values of a chain with N sites and tN
        timesteps

    """

    exp_sig_z = np.empty((len(t), chain_length))
    if spin_constant:
        eigenvalues, eigenvectors = eig_values_vectors_spin_const(J, B0)
    else:
        eigenvalues, eigenvectors = eig_values_vectors(J, B0)

    # non vectorized version. I just keep it for the sake of clarity
    # for time step in t
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

    return (exp_sig_z)


psi0 = np.zeros(dim)
psi0[1] = 1
t = np.linspace(0, 10, 100)
evo = time_evo_sigma_z(t, psi0)

if plot:
    fig, ax = plt.subplots(chain_length, 1, figsize=(
        10, 1 + chain_length), sharex=True)
    if not (all(B) == 0):
        plt.text(0, (chain_length) * 5/3 - 1.5, f"B = {B}")
    plt.subplots_adjust(hspace=0.5)
    for i in range(chain_length):
        ax[i].axhline(0, color="black")
        ax[i].plot(t, evo.T[i])
        ax[i].set_ylim(-0.55, 0.55)
        ax[chain_length-1].set_xlabel("Time t")
        # plt.savefig("N5_P4_B.png")
    plt.show()


def animate_spins(evo):
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = plt.bar(np.arange(chain_length), evo[0])
    frames = evo.shape[0]

    def run(i):
        for j, bar in enumerate(bars):
            bar.set_height(evo[i, j])
        return bars

    anim = animation.FuncAnimation(fig, run, frames=frames, blit=True)

    plt.show()


animate_spins(evo)
