import numpy as np
import matplotlib.pyplot as plt

# Parameters
# ________________________________________________________________________________
chain_length = 3
# dimension of hilbert space
dim = np.int(2**chain_length)
# Coupling constant
J = 2
# Magnetic field
B = 0 * np.random.uniform(-1, 1, chain_length)

# Setup
# ________________________________________________________________________________
# array of states in sigma_z basis
# if chain_length % 8 != 0, there will be zero padding until the byte is full
psi_z = np.arange(0, np.int(2**chain_length), dtype=np.uint8)

H = np.zeros((dim, dim))

# For every state
for state_index in range(dim):
    state = np.unpackbits(psi_z[state_index])
    shifted_state = np.unpackbits(np.right_shift(psi_z, 1)[state_index])
    # Going backwards through the array, because there is zero-padding to the left, which means
    # 15 = 0 0 0 0 1 1 1 1
    for i in range(-1, -(chain_length), -1):
        # Ising term in the hamiltonian: J * Sum(I_i^z * I_i+1^z)
        # Method: shift the array one element to the right and compare to the original indexwise
        if state[i] == shifted_state[i]:
            H[state_index, state_index] += J/4
        else:
            H[state_index, state_index] -= J/4
            # Ladder operator terms: J/2 * Sum(I_i^+ I_i+1^- + I_i^- I_i+1^+)
            # Method: Flip spins and then add 1/2 in the according term in the hamiltonian
            # Only do this, if I_i^z != I_i+1^z, otherwise the ladder operators give 0.
            flipmask = np.unpackbits(
                np.array(np.left_shift(3, -(i+1)), dtype=np.uint8))
            flipped_state = np.packbits(np.bitwise_xor(state, flipmask))
            H[state_index, flipped_state] = J/2
    # Outer magnetic field term: Sum(B_i I_i^z)
    # Method: Just add diagonal terms to the hamiltonian.
    H[state_index, state_index] += np.sum(B * (state[-chain_length:] - 1/2))

# Use eigh for the calculation, since H is hermitian -> I hope for better efficiency
eigenvalues, eigenvectors = np.linalg.eigh(H)


# TIME EVO DOESNT WORK YET


def time_evo_sigma_z(t, psi0):
    """
    Computes the time evolution of the spin operator sigma_z for the given state

    Args:
        t (array [tN]): array with tN timesteps
        psi0 (array [N]): the state at t0

    Returns:
        exp_sig_z (array [tN, N]): Array of expectation values of a chain with N sites and tN
        timesteps

    """
    # for time step in t
    for time__i, ts in enumerate(t):
        exp_sig_z = np.empty((len(t), chain_length), dtype=np.complex64)
        for i in range(chain_length):
            psi_t = (np.exp(-1j * eigenvalues[i] * ts) *
                     np.inner(np.outer(eigenvectors[i], eigenvectors[i]), psi0))
            sigma_z = psi_t[-chain_length:] - 1/2
            exp_sig_z[time__i] = (sigma_z *
                                  np.inner(psi_t.conjugate(), psi_t))
    return exp_sig_z


psi0 = np.unpackbits(psi_z[0])
t = np.linspace(0, 1e-3, 100)
evo = time_evo_sigma_z(t, psi0)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(t, np.abs(evo.T[0]))
ax2.plot(t, np.abs(evo.T[1]))
ax3.plot(t, np.abs(evo.T[2]))
ax3.set_xlabel("Time t")
plt.show()
