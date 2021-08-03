import pdb
import sys
import os
import numpy as np
from shutil import copy2
from configparser import ConfigParser
from math import ceil


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
if scaling == "inverse":
    A = A / chain_length
elif scaling == "inverse_sqrt":
    A = A / np.sqrt(chain_length)
else:
    raise ValueError(f"'{scaling}' is not a possible scaling argument.")

# Output option
Output = config_object["Output"]
filename = Output["filename"]
title = Output["title"]
if title == "":
    title = filename

# Other setup
Other = config_object["Other"]
timefinal = config_object.getfloat("Other", "timefinal")
timestep = config_object.getfloat("Other", "timestep")
n_combined_wf = str_to_int(config_object.getlist("Other", "n_combined_wf"))
wave_function_basis = str_to_int(
    config_object.getlist("Other", "wave_function_basis"))
central_spin_splitted = config_object.getboolean("Other", "central_spin_splitted")
n_realizations = config_object.getint("Other", "n_realizations")
seed = config_object.getint("Other", "seed")


# Functions to write operator file
# ______________________________________________________________________________


def define_section():
    return f"OP_DEFINE-SECTION\nTITLE\n{title}\nEND-TITLE\nEND-OP_DEFINE-SECTION\n\n"


def parameter_section():
    ps = "PARAMETER-SECTION\n"
    if seed:
        np.random.seed(seed)
    for i in range(1, chain_length+1):
        h_i = np.random.uniform(-1, 1)
        ps += (f"h{i} = {h_i}, eV\n")
    ps += f"B = {B}\n"
    ps += f"coupx = {J}, eV\n"
    # coupling x = coupling y
    ps += f"coupz = {J}, eV\n"
    if central_spin:
        ps += f"coup_cs = {A}\n"
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
        for on in operator_names:
            for i in range(1, chain_length + 1):
                hs += f"coup_cs |{i} {on} |{total_spins} {on}\n"
            hs += "\n"

    hs += "END-HAMILTONIAN-SECTION\n\n"
    return hs


# Functions to write input file
# ______________________________________________________________________________
def run_section(job_name):
    rs = f"RUN-SECTION\nname = {job_name}\npropagate\ngridpop\ntitle = {job_name}\nsteps\n"
    rs += f"auto\nveigen\ntinit=0.0  tfinal={timefinal}  tout={timestep}\nEND-RUN-SECTION\n\n"
    return rs


def operator_section(job_name):
    return f"OPERATOR-SECTION\nopname = {job_name}\nEND-OPERATOR-SECTION\n\n"


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
        # split into wf approximately equally
        split = int(ceil(spins_to_combine / n_combined_wf[0]))
        n_cwf = [split] * (spins_to_combine // split)
        remainder = spins_to_combine % split
        if remainder:
            # By inserting the smallest group in the first place, it will always have a
            # consistent size, no matter if the central spin is active or not.
            n_cwf.insert(0, remainder)
        min_value = min(n_cwf)
        if 2**min_value < wave_function_basis[0]:
            print(
                f"A group with of {min_value} spins can maximally be described with {int(2**min_value)} basis functions, but not with {wave_function_basis[0]}")
            raise ValueError()
        wfb = wave_function_basis * len(n_cwf)
    idx = 1
    for wf_idx in range(len(n_cwf)):
        for _ in range(n_cwf[wf_idx]):
            spfbs += f"v{idx}, "
            idx += 1
            # cut out last ', '
        spfbs = spfbs[:-2]
        spfbs += f" = {wfb[wf_idx]}\n"
    if central_spin_splitted:
        spfbs += f"v{total_spins} = 2\n"
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
    for i in range(1, chain_length + 1):
        if i % 2 == 0:
            iwfs += f"v{i}\tgauss\t0.5 \t0.0\t0.001\tpop=1\n"
        else:
            iwfs += f"v{i}\tgauss\t-0.5\t0.0\t0.001\tpop=1\n"

    if central_spin:
        # always initialize the central spin the same (spin down), regardless of chain length
        iwfs += f"v{total_spins}\tgauss\t-0.5 \t0.0\t0.001\tpop=1\n"
    iwfs += "end-build\nEND-INIT_WF-SECTION\n\n"
    return iwfs


# ______________________________________________________________________________
def write_inp_file(path, job_name):
    # write the input file
    with open(path + ".inp", 'w') as f:
        f.write(run_section(job_name))
        f.write(operator_section(job_name))
        f.write(wave_function_basis_section())
        f.write(primitive_basis_section())
        f.write(integrator_section())
        f.write(init_wf_section())
        f.write("end-input\n")


def write_op_file(path):
    # Write the operator file
    with open(path + ".op", 'w') as f:
        f.write(define_section())
        f.write(parameter_section())
        f.write(hamiltonian_section())
        f.write("END-OPERATOR\n")


def write_info_file(path):
    # copy the config file to the data directory
    copy2(config_file, path + "config")


def write_everything():
    for realization in range(n_realizations):
        if n_realizations == 1:
            job_name = filename
        else:
            job_name = filename + "_" + str(realization)
        if not os.path.isdir("input_files"):
            os.mkdir("input_files")
        if os.path.isfile("input_files/" + job_name + ".inp"):
            raise Warning(
                f"file input_files/{job_name}.inp does already exist. Choose another job_name in {config_file}")
        else:
            # Try, because there might be wrong inputs
            try:
                write_inp_file("./input_files/" + job_name, job_name)
            except ValueError:
                os.remove("./input_files/" + job_name + ".inp")
                print(f"No files for {job_name} were produced.")
            else:
                write_op_file("./input_files/" + job_name)
                if n_realizations == 1:
                    job_dir = title
                else:
                    job_dir = title + "_" + str(realization)
                if not os.path.isdir("input_files/" + job_dir):
                    os.mkdir("input_files/" + job_dir)
                    write_info_file("input_files/" + job_dir + "/")
                else:
                    raise Warning(
                        f"Directory \"input_files/\"{title} does already exist. Choose another title in {config_file}")


write_everything()
