import os
import sys

# sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./config_files")
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini":
            config_files.append(entry.name)
    for filename in config_files:
        os.system(f"sbatch --export=ALL,input=config_files/{filename}, -J {filename} start_job.sh")

else:
    for filename in sys.argv[1:]:
        os.system(f"sbatch --export=ALL,input=config_files/{filename}, -J {filename} start_job.sh")
