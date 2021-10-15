import numpy as np
import os
import sys
import shutil
from configparser import ConfigParser
import fileinput
from re import search


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


def replace_text(filename, text, replacement):
    with fileinput.FileInput(filename, inplace=True) as file:
        for line in file:
            print(line.replace(text, replacement), end='')


def strip_float(float_number):
    return str(float_number).rstrip('0').replace('.', '')


def create_subconfigs(config_name):
    # Read config_file with a config_object
    # and returns the path it puts it so the main function in this script can run them
    config_object = ConfigParser(converters={"list": convert_list})
    config_object.read("config_files/" + config_name)
    filename = config_object.get("Output", "filename").rstrip('/')
    real_filename = filename.split('/')[-1]
    path = filename[:-len(real_filename)]
    if not os.path.isdir(path):
        os.mkdir(path)
    Ls = str_to_int(config_object.getlist("System", "chain_length"))
    Bs = str_to_float(config_object.getlist("Constants", "B0"))
    As = str_to_float(config_object.getlist("Constants", "A"))
    Samples = str_to_int(config_object.getlist("Output", "samples"))
    for L in Ls:
        for B in Bs:
            for A in As:
                config_object["System"]["chain_length"] = str(L)
                config_object["Constants"]["B0"] = str(B)
                config_object["Constants"]["A"] = str(A)
                config_object["Output"]["filename"] = f"{filename}_L{L}_B{strip_float(B)}_A{strip_float(A)}"
                if len(Samples) != 1:
                    config_object["Output"]["samples"] = str(Samples[Ls.index(L)])
                # Write the new config file
                with open(f'{path}/{config_name[:-4]}_L{L}_B{strip_float(B)}_A{strip_float(A)}.ini',
                          'w') as conf:
                    config_object.write(conf)
    # create a 'ToPool' flagfile to sign that pooling needs to be done in this directory
    if not os.path.isfile(f"{path}/ToPool"):
        os.mknod(f"{path}/ToPool")
    return path


# Make "scan_ids" optional because i don't want to rewrite all the configs
def send_single_config(config_name):
    """
    This function creates a directory named config_name in "./Plots/" and places all
    sub-calculations for each sample in it while parallelizing the job by dividing the full job
    for all samples for one job for each sample.
    config_name is already the full path to the config_file, not just the filename itself
    """
    # Read config_file with a config_object
    config_object = ConfigParser(converters={"list": convert_list})
    config_object.read(config_name)
    samples = config_object.getint("Output", "samples")
    # filename = path + real_filename
    filename = config_object.get("Output", "filename").rstrip('/')
    real_filename = filename.split('/')[-1]
    path = filename[:-len(real_filename)]
    if not os.path.isdir(path):
        os.mkdir(path)
    # Copy original config
    shutil.copyfile(config_name, f"{config_name[:-4]}_config.ini")
    for i in range(samples):
        new_config_name = f"{config_name[:-4]}_{i}.ini"
        shutil.copyfile(config_name, new_config_name)
        # Set samples to one
        replace_text(new_config_name, f"samples = {samples}", "samples = 1")
        # Set new filename
        replace_text(new_config_name, f"filename = {filename}", f"filename = {filename}_{i}")
        # Set boolean parallelized to True (if it isn't the case yet)
        replace_text(new_config_name, "parallelized = False", "parallelized = True")
        if not os.path.isfile(f"{filename}_{i}.npz"):
            # sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
            os.system(
                f"sbatch --export=ALL,input={new_config_name}, -J {filename}_{i} start_job.sh")
            # os.system(f"python3 main.py {new_config_name}")
        else:
            print(f"{filename}_{i}.npz does already exist")
    os.remove(f"{config_name[:-4]}_config.ini")


if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./config_files")
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini":
            config_files.append(entry.name)
    for config_name in config_files:
        path = create_subconfigs(config_name)
        for sub_entry in os.scandir(path):
            if sub_entry.name[-4:] == ".ini" and not search("_\d+.ini", sub_entry.name):
                send_single_config(path + sub_entry.name)

else:
    for config_name in sys.argv[1:]:
        path = create_subconfigs(config_name)
        for sub_entry in os.scandir(path):
            if sub_entry.name[-4:] == ".ini" and not search("_\d+.ini", sub_entry.name):
                send_single_config(path + sub_entry.name)
