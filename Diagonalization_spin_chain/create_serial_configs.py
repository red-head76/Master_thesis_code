import os
import sys
from configparser import ConfigParser
from re import search


def convert_list(string):
    # Converts a string to a list of floats
    return ([i.strip() for i in string.split(',')])


def str_to_int(str_list):
    return [int(item) for item in str_list]


def str_to_float(str_list):
    return [float(item) for item in str_list]


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
                signature = f"_L{L}_B{strip_float(B)}_A{strip_float(A)}"
                config_object["Output"]["filename"] = filename + signature
                if len(Samples) != 1:
                    config_object["Output"]["samples"] = str(Samples[Ls.index(L)])
                # Write the new config file
                with open(f'config_files/{config_name[:-4]}{signature}.ini', 'w') as conf:
                    config_object.write(conf)


if len(sys.argv) == 1:
    print("Creating config files...")
    entries = os.scandir("./config_files")
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini" and not search("\d.ini", entry.name):
            config_files.append(entry.name)
    for config_name in config_files:
        create_subconfigs(config_name)

else:
    for config_name in sys.argv[1:]:
        create_subconfigs(config_name)
