import pdb
import sys
import os
from shutil import copy2
from configparser import ConfigParser
from math import ceil
from random import uniform


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    # if there is no config file
    if not os.path.isfile("config.ini"):
        raise Exception(
            "No 'config.ini' found. Create one using 'create_config.py'")
    config_file = "config.ini"

# Read config_file with a config_object
config_object = ConfigParser(converters={"list": convert_list})
config_object.read(config_file)

# System setup
System = config_object["System"]
central_spin = config_object.getboolean("System", "central_spin")
chain_length = config_object.getint("System", "chain_length")
total_spins = central_spin + chain_length
periodic_boundaries = config_object.getboolean("System", "periodic_boundaries")

# Coupling Constants
Constants = config_object["Constants"]
J = float(Constants["J"])
B = config_object.getfloat("Constants", "B")
A = config_object.getfloat("Constants", "A")
scaling = Constants["scaling"]

# Output option
Output = config_object["Output"]
filename = Output["filename"]
title = Output["title"]

# Other setup
Other = config_object["Other"]
timefinal = config_object.getfloat("Other", "timefinal")
timesteps = config_object.getfloat("Other", "timesteps")
n_combined_wf = str_to_int(config_object.getlist("Other", "n_combined_wf"))
wave_function_basis = str_to_int(
    config_object.getlist("Other", "wave_function_basis"))
central_spin_splitted = config_object.getboolean("Other", "central_spin_splitted")


# Functions to write operator file
# ______________________________________________________________________________


def define_section():
    return f"OP_DEFINE-SECTION\nTITLE\n{title}\nEND-TITLE\nEND-OP_DEFINE-SECTION\n\n"


def parameter_section():
    ps = "PARAMETER-SECTION\n"
    for i in range(1, chain_length+1):
        h_i = uniform(-1, 1)
        ps += (f"h{i} = {h_i}, eV\n")
    ps += f"B = {B}\n"
    ps += f"coupx = {J}, eV\n"
    # coupling x = coupling y
    ps += f"coupz = {J}, eV\n"
    if central_spin:
        ps += f"A = {A}\n"
    ps += "END-PARAMETER-SECTION\n\n"
    return ps


def hamiltonian_section():
    hs = "HAMILTONIAN-SECTION"
    # Declaring modes
    for i in range(1, total_spins + 1):
        if i % 10 == 1:
            hs += "\nmodes"
        hs += f" | v{i}"
    hs += "\n\n"

    # External field term sum_i h_i S^z_i
    for i in range(1, chain_length + 1):
        hs += f"B*h{i} |{i} q\n"
    hs += "\n"

    # Interaction term sum_i S_i S_i+1 (for all x, y, z)
    coupling_names = ["coupx", "coupx", "coupz"]
    operator_names = ["dq^2", "I*dq", "q"]
    for cn, on in zip(coupling_names, operator_names):
        for i in range(1, chain_length):
            hs += f"{cn} |{i} {on} |{i+1} {on}\n"
        if periodic_boundaries:
            hs += f"{cn} |{chain_length} {on} |{1} {on}\n"
        hs += "\n"

    # Interaction with central spin sum_i S_cs S_i (for all x, y, z)
    if central_spin:
        for cn, on in zip(coupling_names, operator_names):
            for i in range(1, chain_length + 1):
                hs += f"{cn} |{i} {on} |{total_spins} {on}\n"
            hs += "\n"

    hs += "END-HAMILTONIAN-SECTION\n\n"
    return hs


# Functions to write input file
# ______________________________________________________________________________
def run_section():
    rs = f"RUN-SECTION\nname = {title}\npropagate\ngridpop\ntitle = {title}\nsteps\n"
    rs += f"auto\nveigen\ntinit=0.0  tfinal={timefinal}  tout=0.1\nEND-RUN-SECTION\n\n"
    return rs


def operator_section():
    return f"OPERATOR-SECTION\nopname = {filename}\nEND-OPERATOR-SECTION\n\n"


def wave_function_basis_section():
    spfbs = "SPF-BASIS-SECTION\n"
    # Determine the allocation in combined wf
    if central_spin_splitted:
        spins_to_combine = chain_length
    else:
        spins_to_combine = total_spins
    if len(n_combined_wf) > 1:
        if sum(n_combined_wf) != spins_to_combine:
            raise ValueError(" + ".join(list(map(str, n_combined_wf)) +
                                        f" does not equal {spins_to_combine}, the amount of spins which should be combined into different wave functions."))
        else:
            n_cwf = n_combined_wf
            wfb = wave_function_basis
    elif len(n_combined_wf) == 1:
        # split into wf (approx.) equally
        split = int(ceil(spins_to_combine / n_combined_wf[0]))
        n_cwf = [split] * (spins_to_combine // split)
        if spins_to_combine % split:
            n_cwf += [spins_to_combine % split]
        wfb = wave_function_basis * len(n_cwf)
        # Switch the first and the last element. This way, the first group will always have a
        # consistent size, no matter if the central spin is active or not.
        n_cwf[0], n_cwf[-1] = n_cwf[-1], n_cwf[0]
    idx = 1
    for wf_idx in range(len(n_cwf)):
        for _ in range(n_cwf[wf_idx]):
            spfbs += f"v{idx}, "
            idx += 1
            # cut out last ', '
        spfbs = spfbs[:-2]
        spfbs += f" = {wfb[wf_idx]}\n"
    if central_spin_splitted:
        spfbs += f"v{total_spins} = 1\n"
    spfbs += "END-SPF-BASIS-SECTION\n\n"
    return spfbs


def primitive_basis_section():
    pbs = "PRIMITIVE-BASIS-SECTION\n"
    for i in range(1, total_spins + 1):
        pbs += f"v{i}  sin  2  -0.5  0.5  spin\n"
    pbs += "END-PRIMITIVE-BASIS-SECTION\n\n"
    return pbs


def integrator_section():
    return "INTEGRATOR-SECTION\nVMF\nABM = 6, 1.0D-8, 1.0D-6\nEND-INTEGRATOR-SECTION\n\n"


def init_wf_section():
    iwfs = "INIT_WF-SECTION\nbuild\n"
    # initialize in neel state:
    for i in range(1, total_spins + 1):
        if i % 2 == 0:
            iwfs += f"v{i}\tgauss\t0.5 \t0.0\t0.001\tpop=1\n"
        else:
            iwfs += f"v{i}\tgauss\t-0.5\t0.0\t0.001\tpop=1\n"
    iwfs += "end-build\nEND-INIT_WF-SECTION\n\n"
    return iwfs


# ______________________________________________________________________________
def write_inp_file(path):
    # write the input file
    with open(path + filename + ".inp", 'w') as f:
        f.write(run_section())
        f.write(operator_section())
        f.write(wave_function_basis_section())
        f.write(primitive_basis_section())
        f.write(integrator_section())
        f.write(init_wf_section())
        f.write("end-input\n")


def write_op_file(path):
    # Write the operator file
    with open(path + filename + ".op", 'w') as f:
        f.write(define_section())
        f.write(parameter_section())
        f.write(hamiltonian_section())
        f.write("END-OPERATOR\n")


def write_info_file(path):
    # copy the config file to the data directory
    copy2(config_file, path + "config")


def write_everything():
    if not os.path.isdir("input_files"):
        os.mkdir("input_files")
    if os.path.isfile("input_files/" + filename + ".inp"):
        raise Warning(
            f"file input_files/{filename}.inp does already exist. Choose another filename in {config_file}")
    else:
        write_inp_file("./input_files/")
        write_op_file("./input_files/")
    if not os.path.isdir("input_files/" + title):
        os.mkdir("input_files/" + title)
        write_info_file("input_files/" + title + "/")
    else:
        raise Warning(
            f"Directory \"input_files/\"{title} does already exist. Choose another title in {config_file}")


write_everything()
