import os
import shutil

for entry in os.scandir("./input_files/"):
    if entry.is_dir():
        finished = True
        for subentry in os.scandir("./input_files/" + entry.name + "/"):
            if subentry.name == "stop":
                finished = False
                break
        if finished:
            shutil.move("./input_files/" + entry.name, "../data/")
            shutil.move("./input_files/" + entry.name + ".inp", "../data/")
            shutil.move("./input_files/" + entry.name + ".op", "../data/")
