import numpy as np
from scipy.special import binom
from support_functions import packbits, unpackbits, create_basis_vectors


def eig_values_vectors(chain_length, J, B0, A, periodic_boundaries, central_spin):
    """
    Computes the the Heisenberg Hamiltonian without coupling
    to a central spin: H = Sum_i (J * S_i * S_i+1 + B_i S_i^z).

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        A (float): the coupling between the spins in the chain and the central spin
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whethere or not a central spin is present

    Returns:
        eigenvalues (float array [total_spins]): the eigenvalues of the Hamiltonian
        eigenvectors (float array [total_spins, total_spins]): the eigenvectors
    """

    # Setup
    dim = int(2**chain_length)
    # Basis states of sigma_z operator
    psi_z = np.arange(0, dim)

    H = np.zeros((dim, dim))
    # Create a new random B-field for every instance
    B = np.round(np.random.uniform(-1, 1, chain_length), 2)
    # For every state
    for state_index in range(dim):
        state = unpackbits(psi_z[state_index], chain_length)
        if periodic_boundaries:
            start = -1
        else:
            start = 0
        # Check interaction with every other spin
        for i in range(start, chain_length - 1):
            # Ising term in the hamiltonian: J * Sum(S_i^z * S_i+1^z)
            if state[i] == state[i+1]:
                H[state_index, state_index] += J/4
            else:
                H[state_index, state_index] -= J/4
                # Ladder operator terms: J/2 * Sum(S_i^+ S_i+1^- + S_i^- S_i+1^+)
                # Method: Flip spins and then add 1/2 in the according term in the hamiltonian
                # Only do this, if S_i^z != S_i+1^z, otherwise the ladder operators give 0.
                flipmask = np.roll(unpackbits(3, chain_length), i)
                flipped_state = packbits(np.logical_xor(state, flipmask))
                H[state_index, flipped_state] = J/2

        # Outer magnetic field term: Sum(B_i S_i^z)
        H[state_index,
            state_index] += np.sum(B0 * B * (state - 1/2))

    if central_spin:
        # Extend to one more spin and add the interaction with the central spin
        dim_ext = int(dim * 2)
        psi_z_ext = np.arange(0, dim_ext)

        H_ext = np.zeros((dim_ext, dim_ext))
        H_ext[:dim, :dim] = H
        H_ext[dim:, dim:] = H

        # Now treat the last spin as the new central spin
        for state_index in range(dim_ext):
            state = unpackbits(psi_z_ext[state_index], chain_length+1)
            for i in range(chain_length):
                # Central coupling term
                if state[i] == state[-1]:
                    H_ext[state_index, state_index] += A/chain_length/4
                else:
                    H_ext[state_index, state_index] -= A/chain_length/4
                    flipmask = np.zeros(chain_length+1, dtype=np.bool)
                    flipmask[i] = 1
                    flipmask[-1] = 1
                    flipped_state = packbits(np.logical_xor(state, flipmask))
                    H_ext[state_index, flipped_state] = A/chain_length/2
        return np.linalg.eigh(H_ext)

    return np.linalg.eigh(H)


def eig_values_vectors_spin_const(chain_length, J, B0, periodic_boundaries, central_spin):
    """
    Computes the the Heisenberg Hamiltonian without coupling
    to a central spin: H = Sum_i (J * S_i * S_i+1 + B_i S_i^z).

    Args:
        chain_length (int): the length of the spin chain
        J (float): the coupling constant
        B0 (float): the B-field amplitude. Currently random initialized uniformly
                                between (-1, 1).
        periodic_boundaries (bool): determines whether or not periodic boundary
                                                  conditions are used in the chain.
        central_spin (bool): determines whethere or not a central spin is present

    Returns:
        eigenvalues (float array [total_spins]): the eigenvalues of the Hamiltonian
        eigenvectors (float array [total_spins, total_spins]): the eigenvectors
    """

    dim = np.int(2**chain_length)
    psi_z = np.arange(0, dim)
    # Create subspaces
    subspaces = [0] * (chain_length + 1)
    for i in range(len(subspaces)):
        subspaces[i] = np.zeros(np.int(binom(chain_length, i)), dtype=np.int)

    # Fill subspaces
    # For every possible number of spin-up
    for n in range(len(subspaces)):
        # Check every state to fit in that space
        sub_counter = 0
        for state in psi_z:
            n_up = np.sum(unpackbits(state, chain_length))
            if n_up == n:
                subspaces[n][sub_counter] = state
                sub_counter += 1

    # Create a new random B-field for every instance
    B = np.round(np.random.uniform(-1, 1, chain_length), 2)

    eigenvalues = np.zeros(dim)
    eigenvectors = np.zeros((dim, dim))

    # Generate Hamiltonian for each subspace
    for psi_sub in subspaces:
        dim_sub = len(psi_sub)
        H_sub = np.zeros((dim_sub, dim_sub))
        # For every state
        for state_index in range(len(psi_sub)):
            state = unpackbits(psi_sub[state_index], chain_length)
            if periodic_boundaries:
                start = -1
            else:
                start = 0

            for i in range(start, chain_length - 1):
                # Ising term in the hamiltonian: J * Sum(S_i^z * S_i+1^z)
                if state[i] == state[i+1]:
                    H_sub[state_index, state_index] += J/4
                else:
                    H_sub[state_index, state_index] -= J/4
                    # Ladder operator terms: J/2 * Sum(S_i^+ S_i+1^- + S_i^- S_i+1^+)
                    # Method: Flip spins and then add 1/2 in the according term in the hamiltonian
                    # Only do this, if S_i^z != S_i+1^z, otherwise the ladder operators give 0.
                    flipmask = np.roll(unpackbits(3, chain_length), i)
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
            psi_sub, dim)
        eigenvectors[psi_sub] = eigenvectors_fullspace

    return eigenvalues, eigenvectors
