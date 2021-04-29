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
B = np.random.standard_normal(chain_length)

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
