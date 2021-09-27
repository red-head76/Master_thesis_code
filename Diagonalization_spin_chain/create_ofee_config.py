from configparser import ConfigParser


def create_ofee_config():
    """
    Creates a 'config.ini' file with default parameters

    """

    config_object = ConfigParser(allow_no_value=True)
    intro_string = """# All parameters are explained in the creation file create_ofee_config.py."""
    config_object.set('DEFAULT', intro_string, None)

    config_object["Output"] = {
        # Available options: plot_time_evo, animate_time_evo, plot_r, plot_r_fig3, plot_f_fig2,
        #                    plot_fa, plot_half_chain_entropy, plot_occupation_imbalance,
        #                    plot_exp_sig_z_central_spin, calc_eigvals_eigvecs, calc_psi_t
        "outputtype": "plot",
        # Paths of data to read
        "data_paths": "some_paths",
        # If empty, then the output is not saved, otherwise it is stored to given path
        "filename": "",
        # Determines if the output should be displayed or not
        "show": "True",
        # Whether or not the plot should be saved
        "save_plot": "True",
    }

    config_object["Other"] = {
        # Index where psi 0 is nonzero
        "idx_psi_0": "1",
        # Time parameters (in fs)
        "timestart": "0",
        "timeend": "10",
        "timesteps": "100",
    }

    # Write the above sections to config.ini file
    with open('ofee_config.ini', 'w') as conf:
        config_object.write(conf)


if __name__ == "__main__":
    create_ofee_config()
