import numpy as np
from matplotlib.pyplot import savefig
from json import dump
from os.path import isdir
from os import mkdir
from matplotlib import animation
from shutil import copy


def pc(matrix, precision=1):
    """
    print clean
    Just to print out a matrix cleanly without unreadable data junk
    """
    with np.printoptions(precision=precision, suppress=True):
        print(matrix)


def create_basis_vectors(indices, dimension):
    """
    Creates an array of basis vectors according to given indices

    Args:
        indices (array [N]): an array containing N integers
        dimension (int, default: dim): the dimension of the vector space

    Returns:
        basis_vectors (array [N, dim]): an array containing N basis vectors.

    """
    basis_vectors = np.zeros((len(indices), dimension))
    for row, index in enumerate(indices):
        basis_vectors[row, index] = 1
    return basis_vectors

# New definitions of packbits and unpackbits are required because np.unpackbits can only handle
# uint8. This means it is restricted to a chain_length of 8.


def unpackbits(x, num_bits):
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
    x = np.array(x)
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def packbits(x):
    """
    Similar to np.packbits, but can also handle longer uints than uint8
    Example: packbits([1, 1, 0, 0]) = 1 * 1 + 1 * 2 + 0 * 4 + 0 * 8 = 3

    Args:
        x (array [N, chain_length]): input array of unpacked bits
        num_bits (int, default: chain_length): number of bits

    Returns:
        packed_bits (array [x.shape[0]]): an array of integer

    """
    mask = 2**np.arange(x.shape[-1])
    return mask @ x.transpose()


def partial_trace_subspace(rho_sub, subspace_mask, spins_a, calc_rho_a=True):
    """
    Calculates the partial trace for the given rho given in a subpace with constant spin and
    dimension of new subspace a and b, where the chain is divided.
    Explanation see Notes/partial_trace_calc.pdf

    Args:
        rho_sub (array (float) [dim, dim] or [t, dim, dim]): full density matrix with dimensions of
                            the full hamiltonian. With optional time dimension in front.
        subspace_mask (array (float)) indices of states with const spin n in terms of the fullspace
        spins_a (int): spins in subspace a.
        calc_rho_a (bool, default: True): determines whether the partial trace over b
                    (if True, results in rho_a) or over a (results in rho_b) should be calculated.

    Returns:
        rho_a (array (float) [dim_a, dim_a] or [t, dim_a, dim_a]): the partial trace over b
                (or the other way around if calc_rho_a=False).
    """
    # maximum amount of spins used in subspace
    max_spins = np.int(np.floor(np.log2(np.max(subspace_mask))) + 1)
    # Split the entries of the mask into entries of the subspaces a and b (unpacked)
    splitted_idx = np.split(unpackbits(subspace_mask, max_spins), [spins_a], axis=1)
    # pack bits back in the individual subspaces a and b
    packed_idx = np.array((packbits(splitted_idx[0]), packbits(splitted_idx[1])))
    # density matrix with only entries from the subspace of const. total spin
    # in terms subspaces a and b (i.e. in form of rho_{(a1, b1), (a2, b2)})
    rho_idx_sub = np.rollaxis(np.array(
        (np.meshgrid(packed_idx[0], packed_idx[0]),
         np.meshgrid(packed_idx[1], packed_idx[1]))), 1)
    if calc_rho_a:
        # then the second index should be equal (diagonal in b)
        # the entry at pos_idx shows the position of the entry in the partial trace
        equal_idx = 1
        pos_idx = 0
    else:
        equal_idx = 0
        pos_idx = 1
    # indices of the entries that contribute to the partial trace
    trace_mask = np.where(rho_idx_sub[0, equal_idx] == rho_idx_sub[1, equal_idx])
    # indices where the entries should go in the partial trace
    new_pos = np.array((rho_idx_sub[0, pos_idx][trace_mask], rho_idx_sub[1, pos_idx][trace_mask]))
    # partial trace (finally)
    rho_a = np.zeros(rho_sub.shape, dtype=complex)
    for idx in range(new_pos.shape[1]):
        rho_a[..., new_pos[0, idx], new_pos[1, idx]] +=\
            rho_sub[..., trace_mask[0][idx], trace_mask[1][idx]]
    return rho_a


def partial_trace(rho, spins_a, calc_rho_a=True):
    """
    Calculates the partial trace for the given rho and dimensions of the subspaces
    Explanation see Notes/partial_trace_calc.pdf

    Args:
        rho (array (float) [dim, dim] or [t, dim, dim]): full density matrix with dimensions of
                            the full hamiltonian. With optional time dimension in front.
        spins_a (int): spins in subspace a.
        rho_a (bool, default: True): determines whether the partial trace over b (if True, results
                                     in rho_a) or over a (results in rho_b) should be calculated.

    Returns:
        rho_a (array (float) [dim_a, dim_a] or [t, dim_a, dim_a]): the partial trace over b
                (or the other way around if rho_a=False).
    """
    axis_offset = np.size(rho.shape) - 2
    dim = rho.shape[-1]
    dim_a = int(2**spins_a)
    dim_b = int(dim / dim_a)
    if axis_offset:
        rhojkjk = rho.reshape(-1, dim_a, dim_b, dim_a, dim_b)
    else:
        rhojkjk = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    # If rho_a is true, axis 1 and 3 (+offset) are taken, axis 0 and 2 otherwise
    return np.diagonal(rhojkjk, axis1=rho_a+axis_offset,
                       axis2=(2+rho_a+axis_offset)).sum(axis=2+axis_offset)


def save_data(filename, data, config_file, anim=False, fps=10):
    """
    Saves the data to a given plot with a given filename. There is one file for the plot, the data
    of the plot and the parameters used.

    Args:
        filename (string): The filename used for saving
        data (array): the data of the plot
        params (dict): the parameters used to create the plot
        anim (bool, default: False): Determines if the output is an animation
        fps (int, default=10): set the frames per second of an animation

    """

    if not isdir("./Plots"):
        mkdir("./Plots")
    save_path = "./Plots/" + filename
    if not anim:
        savefig(save_path)
    else:
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writervideo)

    np.savez(save_path, *data)
    copy(config_file, "./Plots/")
