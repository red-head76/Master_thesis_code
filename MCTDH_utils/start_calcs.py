import sys
import os

# sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
if len(sys.argv) == 1:
    print("Submitting all files...")
    entries = os.scandir("./input_files")
    input_files = []
    for entry in entries:
        if entry.name[-4:] == ".inp":
            os.system(
                f"sbatch --export=ALL,input=input_files/{entry.name}, -J {entry.name} start_job.sh")

else:
    if "-c" in sys.argv:
        continue_job = True
    else:
        continue_job = False
    for arg in sys.argv[1:]:
        for entry in os.scandir("./input_files"):
            if entry.name[-4:] == ".inp" and arg in entry.name:
                if continue_job:
                    os.system(
                        f"sbatch --export=ALL,input=input_files/{entry.name}, -J {entry.name} continue_job.sh")
                else:
                    os.system(
                        f"sbatch --export=ALL,input=input_files/{entry.name}, -J {entry.name} start_job.sh")
