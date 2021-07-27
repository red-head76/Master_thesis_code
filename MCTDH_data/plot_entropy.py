import pdb
import numpy as np
import matplotlib.pyplot as plt
from read_output import read_output

# ["./run20", "./run40", "./run80"]
filenames = ["./20-5sp_20wf_delocalized", "./20-5sp_20wf_localized"]
data = [read_output(filename + "/output") for filename in filenames]
labels = ["delocalized", "localized"]
for dataset, label in zip(data, labels):
    # weights of subroom 1
    ws1 = dataset["weights"][0]
    # cut out the zero elements
    # ugly, but cuts out the the Runtime warning caused by of 0 values in log
    entropy = -np.sum(ws1 * np.log(ws1, where=ws1 > 0,
                                   out=np.zeros(ws1.shape)), axis=1)
    plt.plot(dataset["time"], entropy, label=label)
    plt.semilogx()
    plt.legend()
plt.hlines(np.log(20), 0, 100)
plt.show()
