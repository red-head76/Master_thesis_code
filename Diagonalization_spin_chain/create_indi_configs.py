from create_config import create_config

config_object = create_config()

Ls = [8, 10, 12]
Js = [0]
J_xys = [0]
Bs = [1]
As = [1]

outputtypes = ["half_chain_entropy", "occupation_imbalance"]

# System setup
System = config_object["System"]
System["central_spin"] = "True"
System["chain_length"] = "14"
System["periodic_boundaries"] = "True"
System["spin_constant"] = "True"

# Coupling Constants
Constants = config_object["Constants"]
Constants["scaling"] = "sqrt"

# Output option
Output = config_object["Output"]
Output["samples"] = "2"
Output["show"] = "False"
Output["save_plot"] = "False"

# Other setup
Other = config_object["Other"]
Other["timestart"] = "0.1"
Other["timeend"] = "1000"
Other["timesteps"] = "300"
Other["seed"] = "0"

for L in Ls:
    for J in Js:
        for J_xy in J_xys:
            for B in Bs:
                for A in As:
                    for outputtype in outputtypes:
                        System["chain_length"] = str(L)
                        Constants["J"] = str(J)
                        Constants["J_xy"] = str(J_xy)
                        Constants["B0"] = str(B)
                        Constants["A"] = str(A)
                        Output["outputtype"] = f"plot_{outputtype}"
                        if outputtype == "half_chain_entropy":
                            signature = "hce"
                        elif outputtype == "occupation_imbalance":
                            signature = "oi"
                        if len(Ls) <= 1:
                            signature += f"_{L}"
                        signature += f"_{J}{J_xy}{B}{A}"
                        Output["filename"] = f"./Plots/individual_mechanisms/{signature}"
                        with open(f"./config_files/{signature}.ini", 'w') as conf:
                            config_object.write(conf)
