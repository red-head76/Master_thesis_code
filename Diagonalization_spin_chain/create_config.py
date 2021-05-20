from configparser import ConfigParser


def create_config():
    """
    Creates a 'config.ini' file with default parameters

    """

    config_object = ConfigParser(allow_no_value=True)
    intro_string = """# All parameters are explained in the creation file create_config.py. \n# A False corresponds to an empty string, True is any string."""
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
        "A": "1"
    }

    config_object["Output"] = {
        # Available options: plot, animate, plot_r_values
        "outputtype": "plot",
        # If empty, then the output is not saved, otherwise it is stored to ./Plots/filename
        "filename": "",
        # In case of calculating the r_values, the amount of created samplings
        "sampling": "100"
    }

    config_object["Other"] = {
        # Index where psi0 is nonzero
        "idx_psi0": "1",
        "timespan": "10",
        "timesteps": "100"
    }

    # Write the above sections to config.ini file
    with open('config.ini', 'w') as conf:
        config_object.write(conf)


if __name__ == "__main__":
    create_config()
