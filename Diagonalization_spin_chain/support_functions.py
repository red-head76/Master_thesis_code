import numpy as np
from matplotlib.pyplot import savefig
from json import dump
from os.path import isdir
from os import mkdir
from matplotlib import animation


def pc(matrix, precision=1):
    """
    print clean
    Just to print out a matrix cleanly without unreadable datajunk
    """
    with np.printoptions(precision=precision, suppress=True):
        print(matrix)


def create_basis_vectors(indices, dimension):
    """
    Creates an array of basis vectors according to given indicees

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
        packed_bits (array [N]): an array of integer

    """
    mask = 2**np.arange(x.size)
    return np.inner(mask, x)


def partial_trace(rho, spins_a, rho_a=True):
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


def save_data(filename, data, params, anim=False, fps=10):
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
    with open(save_path + "_info.json", 'w') as jsonfile:
        dump(params, jsonfile)
