import re

with open("outputtest", 'r') as f:
    fulltext = f.read()


time = re.findall(r"Time[ ]+=[ ]+(\d+.\d+)", fulltext)
e_tot = re.findall(r"E-tot[ ]+=[ ]+(-?\d+.\d+)", fulltext)
e_corr = re.findall(r"E-corr[ ]+=[ ]+(-?\d+.\d+)", fulltext)
delta_e = re.findall(r"Delta-E[ ]+=[ ]+(-?\d+.\d+)", fulltext)

# Natural weights *1000
weight_blocks = re.finditer(r"v\d+\s+C\*:(?:[CE\*\d\.\-> ]+\n)+", fulltext)
# use inside weight block
weights = []
for matched_block in weight_blocks:
    blocktext = fulltext[matched_block.start(): matched_block.end()]
    weights_in_block = re.findall(r"-?\d+\.\d*(?:E\-\d*)?", blocktext)
    weights.extend(weights_in_block)

# block for expectation values and variances
exp_blocks = re.finditer(r"(?:v\d+\s+:[ <>\-dq\.=\dn]*\n)+", fulltext)
exp_q = []
exp_dq = []
for matched_block in exp_blocks:
    blocktext = fulltext[matched_block.start(): matched_block.end()]
    exp_q_in_block = re.findall(r"<q>=\s*(\-?\d\.\d*)", blocktext)
    exp_q.extend(exp_q_in_block)
    exp_dq_in_block = re.findall(r"<dq>=\s*(\-?\d\.\d*)", blocktext)
    exp_dq.extend(exp_dq_in_block)
