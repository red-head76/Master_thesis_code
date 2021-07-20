import sys
import os

# sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./input_files")
    input_files = []
    for entry in entries:
        if entry.name[-4:] == ".inp":
            input_files.append(entry.name)
    for filename in input_files:
        os.system(f"sbatch --export=ALL,input=input_files/{filename}, -J {filename} start_job.sh")


else:
    for filename in sys.argv[1:]:
        print(f"submitting {filename}")
        os.system(f"sbatch --export=ALL,input={filename}, -J {filename} start_job.sh")
