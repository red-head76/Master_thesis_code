import numpy as np
import re

"""
This script provides
  * time: array with size (time)
  * e_tot: array with size (time)
  * e_corr: array with size (time)
  * delta_e: array with size (time)
  * weights: array of natural weights with shape (time, combined wf, weight basis state)
  * exp_q: array of expectation values <q> with shape (time, expectation values)
  * exp_dq: array of expectation values <dq> with shape (time, expectation values)
"""

# ______________________________________________________________________________
# pattern of expectation values:
exp_pattern = r"v\d+"
file_to_read = "outputtest"
# ______________________________________________________________________________


with open(file_to_read, 'r') as f:
    fulltext = f.read()

# Check if the process have finished and produced a valid output file
# via checking if the string "Total time" is in there
if not re.search("Total time", fulltext):
    # if not, than cut the file up to the last valid block
    last_valid_pos = list(re.finditer("-{78}", fulltext))[-1].end()
    fulltext = fulltext[:last_valid_pos]

# Check if pattern for expectation value matches correctly:
if not re.search(exp_pattern + r" +C\*", fulltext):
    raise ValueError(
        f"The search patter '{exp_pattern}' for the expectation values does not seem to occur!")

time = re.findall(r"Time += +(\d+.\d+)", fulltext)
e_tot = re.findall(r"E-tot += +(-?\d+.\d+)", fulltext)
e_corr = re.findall(r"E-corr += +(-?\d+.\d+)", fulltext)
delta_e = re.findall(r"Delta-E += +(-?\d+.\d+)", fulltext)

# Get number of natural weights for combined wave functions
# One full block of natural weights
nw_block = re.search(
    r"Natural weights \*1000 :\n" + exp_pattern + r"((?s:.*?)) Mode expectation", fulltext)
nw_individual_blocks = re.split(exp_pattern, nw_block.group(1))
# Number of combined wave functions:
n_cwf = len(nw_individual_blocks)
# Number of used basis states for the combined wave functions
n_bs = [len(re.findall(r"-?\d+\.\d*(?:E\-\d*)?", text))
        for text in nw_individual_blocks]

# Get number of expectation values
# One full block of mode expectation values and variances:
mev_block = re.search("values and variances :\n((?s:.*?))-{2,}", fulltext)
# number of expectation values
n_exp_vals = mev_block.group(1).count('\n')

# Natural weights *1000
weight_blocks = re.finditer(
    exp_pattern + r"\s+C\*:(?:[CE\*\d\.\-> ]+\n)+", fulltext)
# use inside weight block
weights = []
for matched_block in weight_blocks:
    weights_in_block = re.findall(
        r"-?\d+\.\d*(?:E\-\d*)?", matched_block.group(0))
    weights.extend(weights_in_block)
weights = np.array(weights, dtype=float).reshape(-1, np.sum(n_bs))
# split into each block of combined wave functions:
# [:-1] because the last split always returns an empty array
weights = np.split(weights, np.cumsum(n_bs), axis=1)[:-1]
# transpose it into format (time, combined_wf, basis_state)
weights = np.transpose(weights, (1, 0, 2))

# Block for expectation values and variances
exp_blocks = re.finditer(
    r"(?:" + exp_pattern + r"\s+:[ <>\-dq\.=\dn]*\n)+", fulltext)
exp_q = []
exp_dq = []
for matched_block in exp_blocks:
    exp_q_in_block = re.findall(r"<q>=\s*(\-?\d\.\d*)", matched_block.group(0))
    exp_q.extend(exp_q_in_block)
    exp_dq_in_block = re.findall(
        r"<dq>=\s*(\-?\d\.\d*)", matched_block.group(0))
    exp_dq.extend(exp_dq_in_block)
exp_q = np.array(exp_q, dtype=float).reshape(-1, n_exp_vals)
exp_dq = np.array(exp_dq, dtype=float).reshape(-1, n_exp_vals)
