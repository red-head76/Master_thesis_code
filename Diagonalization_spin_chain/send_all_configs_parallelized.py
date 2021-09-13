import numpy as np
import os
import sys
import shutil
from configparser import ConfigParser
import fileinput
import pdb


def send_single_config(config_name):
    """
    This function creates a directory named config_name in "./Plots/" and places all
    sub-calculations for each sample in it while parallelizing the job by dividing the full job
    for all samples for one job for each sample.
    """
    if not os.path.isdir("./Plots/" + config_name[:-4]):
        os.mkdir("./Plots/" + config_name[:-4])
    # Read config_file with a config_object
    config_object = ConfigParser()
    config_object.read("config_files/" + config_name)
    samples = config_object.getint("Output", "samples")
    filename = config_object.get("Output", "filename")
    # Set samples to one
    with fileinput.FileInput("./config_files/" + config_name, inplace=True) as file:
        for line in file:
            print(line.replace(f"samples = {samples}", "samples = 1"), end='')

    for i in range(samples):
        if i == 0:
            old_filename = filename
        else:
            old_filename = f"{config_name[:-4]}/{filename}_{i-1}"
        with fileinput.FileInput("./config_files/" + config_name, inplace=True) as file:
            for line in file:
                print(line.replace(f"filename = {old_filename}",
                                   f"filename = {config_name[:-4]}/{filename}_{i}"), end='')
        # sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
        os.system(
            f"sbatch --export=ALL,input=config_files/{config_name}, -J {filename}_{i} start_job.sh")

    # Restore original config
    with fileinput.FileInput("./config_files/" + config_name, inplace=True) as file:
        for line in file:
            print(line.replace(f"samples = 1", f"samples = {samples}"), end='')
    with fileinput.FileInput("./config_files/" + config_name, inplace=True) as file:
        for line in file:
            print(line.replace(f"filename = {config_name[:-4]}/{filename}_{samples-1}",
                               f"filename = {filename}"), end='')
    shutil.copyfile("./config_files/" + config_name, "./Plots/" +
                    config_name[:-4] + "/" + config_name)


def pool_data_files(config_name):
    path = "./Plots/" + config_name[:-4] + "/"
    config_object = ConfigParser()
    config_object.read("config_files/" + config_name)
    outputtype = config_object.get("Output", "outputtype")
    samples = config_object.getint("Output", "samples")
    filename = config_object.get("Output", "filename")
    if outputtype == "calc_eigvals_eigvecs":
        # data : eigenvalues [dim], eigenvectors [dim, dim]
        pooled_eigenvalues = []
        pooled_eigenvectors = []
        for entry in os.scandir(path):
            if entry.name[-4:] == ".npz":
                data = np.load(path + entry.name)
                pooled_eigenvalues.append(data["arr_0"])
                pooled_eigenvectors.append(data["arr_1"])
        pooled_data = {}
        np.savez(path + filename + "_pooled.npz", eigenvalues=np.array(pooled_eigenvalues),
                 eigenvectors=np.array(pooled_eigenvectors))
        # for entry in os.scandir(path):
        #     if entry.name[-4:] == ".npz" and entry.name[-10:] != "pooled.npz":
        #         os.remove(path + entry.name)
    else:
        raise Warning(f"Pooling of {outputtype} isn't implemented yet.")


if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./config_files")
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini":
            config_files.append(entry.name)
    for config_name in config_files:
        send_single_config(config_name)
        # pool_data_files(config_name)

else:
    for config_name in sys.argv[1:]:
        send_single_config(config_name)
        # os.system(
        #     f"sbatch --export=ALL,input=config_files/{config_name}, -J {config_name} start_job.sh")
