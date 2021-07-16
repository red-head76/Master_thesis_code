import sys
import os

# sbatch --export=ALL,input=*your_input_file1*.inp -J *name_of_job1* start_job.sh
for filenames in sys.argv[1:]:
    os.system(
        f"sbatch --export=ALL,input=input_files/{filenames}, -J {filenames} start_job.sh")
