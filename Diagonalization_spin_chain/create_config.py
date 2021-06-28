from configparser import ConfigParser


def create_config():
    """
    Creates a 'config.ini' file with default parameters

    """

    config_object = ConfigParser(allow_no_value=True)
    intro_string = """# All parameters are explained in the creation file create_config.py."""
    config_object.set('DEFAULT', intro_string, None)

    config_object["System"] = {
        # If the system should have a central spin, coupling to all other spins in the chain
        "central_spin": "",
        # Total number of spins in the chain
        "chain_length": "3",
        # Periodic boundary conditions
        "periodic_boundaries": "True",
        # Whether the total spin of the system is constant or not
        "spin_constant": ""
    }

    config_object["Constants"] = {
        # Coupling between spins in the chain
        "J": "2",
        # Strength of the external B-field
        "B0": "1",
        # Coupling of the central spin with the spins in the chain
        "A": "1",
        # Determines the scaling of A. Options: "inverse" for A/N, or "sqrt" for A/sqrt(N)
        "scaling": "inverse"
    }

    config_object["Output"] = {
        # Available options: plot_time_evo, animate_time_evo, plot_r, plot_r_fig3, plot_f_fig2,
        #                    plot_fa, plot_sa, plot_occupation_imbalance,
        #                    plot_exp_sig_z_central_spin
        "outputtype": "plot",
        # If empty, then the output is not saved, otherwise it is stored to ./Plots/filename
        "filename": "",
        # In case of calculating the plot_r_values, the amount of created samples
        # In case of multiple instances, a list is also possible.
        "samples": "100",
        # Determines if the output should be displayed or not
        "show": "True"
    }

    config_object["Other"] = {
        # Index where psi 0 is nonzero
        "idx_psi_0": "1",
        "timeend": "10",
        "timesteps": "100",
        # In certain cases a random seed can be used to produce comparable outcomes. 0 equals None
        "seed": "0"
    }

    # Write the above sections to config.ini file
    with open('config.ini', 'w') as conf:
        config_object.write(conf)


if __name__ == "__main__":
    create_config()
