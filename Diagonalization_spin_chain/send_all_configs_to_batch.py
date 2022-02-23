import os
import sys

# sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
if len(sys.argv) == 1:
    print("Submitting all files...")
    config_dir = "./config_files"
    entries = os.scandir(config_dir)
    config_files = []
    for entry in entries:
        if entry.name[-4:] == ".ini":
            config_files.append(entry.name)
    for filename in config_files:
        os.system(f"sbatch --export=ALL,input={config_dir}/{filename}, -J {filename} start_job.sh")
        # os.system(f"python3 main.py {config_dir}/{filename}")

else:
    for filename in sys.argv[1:]:
        os.system(f"sbatch --export=ALL,input=config_files/{filename}, -J {filename} start_job.sh")
