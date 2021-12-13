from shutil import copy
import os
from configparser import ConfigParser
from matplotlib.pyplot import savefig, close
from matplotlib import animation
import numpy as np


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
        rho_sub (array (float) [dim_sub, dim_sub] or [t, dim_sub, dim_sub]): density matrix with
            dimensions of the subspace with constant spin. With optional time dimension in front.
        subspace_mask (array (float)) indices of states with const spin n in terms of the fullspace
        spins_a (int): spins in subspace a (or subspace b if calc_rho_a=False)
        calc_rho_a (bool, default: True): determines whether the partial trace over b
                    (if True, results in rho_a) or over a (results in rho_b) should be calculated.

    Returns:
        rho_a (array (float) [dim_a, dim_a] or [t, dim_a, dim_a]): the partial trace over b
                (or the other way around if calc_rho_a=False).
    """
    # maximum amount of spins used in subspace
    if np.max(subspace_mask) == 0:
        max_spins = 1
    else:
        max_spins = int(np.floor(np.log2(np.max(subspace_mask))) + 1)
    if calc_rho_a:
        spins_subspace = spins_a
    else:
        spins_subspace = int(max_spins - spins_a)
    # Split the entries of the mask into entries of the subspaces a and b (unpacked)
    splitted_idx = np.split(unpackbits(subspace_mask, max_spins), [spins_a], axis=1)
    # pack bits back in the individual subspaces a and b
    # 0 and 1 are reversed here because of implementation of packed_bits
    packed_idx = np.array((packbits(splitted_idx[0]), packbits(splitted_idx[1])))
    # density matrix with only entries from the subspace of const. total spin
    # in terms subspaces a and b (i.e. in form of rho_{(a1, b1), (a2, b2)})
    rho_idx_sub = np.rollaxis(np.array((np.meshgrid(packed_idx[0], packed_idx[0], indexing='ij'),
                                        np.meshgrid(packed_idx[1], packed_idx[1], indexing='ij'))
                                       ), 1)
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
    # in case there is a time array
    if len(rho_sub.shape) == 3:
        rho_a = np.zeros((rho_sub.shape[0], int(2**spins_subspace),
                          int(2**spins_subspace)), dtype=complex)
    else:
        rho_a = np.zeros((int(2**spins_subspace), int(2**spins_subspace)), dtype=complex)
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
        calc_rho_a (bool, default: True): determines whether the partial trace over b
                    (if True, results in rho_a) or over a (results in rho_b) should be calculated.

    Returns:
        rho_a (array (float) [dim_a, dim_a] or [t, dim_a, dim_a]): the partial trace over b
                (or the other way around if calc_rho_a=False).
    """
    axis_offset = np.size(rho.shape) - 2
    dim = rho.shape[-1]
    dim_a = int(2**spins_a)
    dim_b = int(dim / dim_a)
    if axis_offset:
        rhojkjk = rho.reshape(-1, dim_a, dim_b, dim_a, dim_b)
    else:
        rhojkjk = rho.reshape(dim_a, dim_b, dim_a, dim_b)
    # If calc_rho_a is true, axis 1 and 3 (+offset) are taken, axis 0 and 2 otherwise
    return np.diagonal(rhojkjk, axis1=calc_rho_a+axis_offset,
                       axis2=(2+calc_rho_a+axis_offset)).sum(axis=2+axis_offset)


def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file
        (from https://thispointer.com/python-how-to-insert-lines-at-the-top-of-a-file/)"""
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for read_line in read_obj:
            write_obj.write(read_line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def save_data(save_path, data, config_file, time_passed, save_plot=True, picture_format="png",
              parallelized=False, anim=False, fps=10):
    """
    Saves the data to a given plot with a given filename. There is one file for the plot, the data
    of the plot and the parameters used.

    Args:
        filename (string): The filename used for saving
        data (array): the data of the plot
        config_file (string): the path of the .ini file
        time_passed (int): the time needed for the calculation
        save_plot (bool, default: True): Whether or not the plot should be saved
        anim (bool, default: False): Determines if the output is an animation
        fps (int, default=10): set the frames per second of an animation

    """
    save_path = save_path.rstrip('/')
    filename = save_path.split('/')[-1]
    dir_path = save_path[:-len(filename)]
    copy_config_path = dir_path + '/' + config_file.rstrip('/').split('/')[-1]
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if save_plot:
        if anim == False:
            savefig(save_path + '.' + picture_format)
        elif anim:
            writervideo = animation.FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writervideo)
    else:
        close()
    np.savez(save_path, *data)
    if not parallelized:
        copy(config_file, save_path + ".ini")
        t = int(time_passed)
        prepend_line(save_path + ".ini",
                     f"# Run time {t//3600}h:{(t%3600)//60}m:{t%60}s\n")
    else:
        t = int(time_passed)
        prepend_line(config_file,
                     f"# Run time {t//3600}h:{(t%3600)//60}m:{t%60}s\n")


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def read_config(config_file):
    config_object = ConfigParser(converters={"list": convert_list})
    config_object.read(config_file)
    return config_object
