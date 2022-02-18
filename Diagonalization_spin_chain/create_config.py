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
        "central_spin": "False",
        # Total number of spins in the chain
        "chain_length": "3",
        # Periodic boundary conditions
        "periodic_boundaries": "True",
        # This is depracted, everything is calculated with spin constant on default
        # # Whether the total spin of the system is constant or not
        # "spin_constant": "True"
    }

    config_object["Constants"] = {
        # Spin chain coupling in z-direction
        "J": "1",
        # Spin chain coupling in xy-direction.
        "J_xy": "1",
        # Strength of the external B-field
        "B0": "1",
        # Coupling of the central spin with the spins in the chain
        "A": "1",
        # Determines the scaling of A. Options: "inverse" for A/N, or "sqrt" for A/sqrt(N)
        "scaling": "inverse"
    }

    config_object["Output"] = {
        # Available options: plot_time_evo, plot_light_cone, animate_time_evo, plot_r,
        #                    plot_r_fig3, plot_f_fig2,
        #                    plot_fa, plot_half_chain_entropy, plot_occupation_imbalance,
        #                    plot_exp_sig_z_central_spin, calc_eigvals_eigvecs, calc_psi_t,
        #                    plot_occupation_imbalance_plateau
        "outputtype": "plot",
        # If empty, then the output is not saved, otherwise it is stored to ./Plots/filename
        "filename": "",
        # In case of calculating the plot_r_values, the amount of created samples
        # In case of multiple instances, a list is also possible.
        "samples": "100",
        # Determines if the output should be displayed or not
        "show": "True",
        # Whether or not the plot should be saved
        "save_plot": "True",
        # Output format of the picture
        "picture_format": "png",
        # If True, the the configs won't be copied twice in save_data in support_functions
        "parallelized": "False",
    }

    config_object["Other"] = {
        # String ('neel', 'neel_inverse', 'domain_wall') or integer. In the latter case, the
        # state corresponding to unpackbits(initial_state, total_spins) will be used
        # For example: initial_state = 3, total_spins = 4 -> [1, 1, 0, 0] is used
        "initial_state": "1",
        # Time parameters (in fs)
        "timestart": "0",
        "timeend": "10",
        "timesteps": "100",
        # In certain cases a random seed can be used to produce comparable outcomes. 0 equals None
        "seed": "0"
    }
    return config_object


if __name__ == "__main__":
    config_object = create_config()
    # Write the above sections to config.ini file
    with open('config.ini', 'w') as conf:
        config_object.write(conf)
