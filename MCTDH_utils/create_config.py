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
    }

    config_object["Constants"] = {
        # Coupling between spins in the chain
        "J": "2",
        # Strength of the external B-field
        "B": "1",
        # Coupling of the central spin with the spins in the chain
        "A": "1",
        # Determines the scaling of A. Options: "inverse" for A/N, or "sqrt" for A/sqrt(N)
        "scaling": "inverse"
    }

    config_object["Output"] = {
        # Name of the output file
        "filename": "filename",
        # title is used in the mctdh log files. if left empty, it is set to the same value as 'filename'
        "title": "title"
    }

    config_object["Other"] = {
        # The endpoint of the calculation in fs
        "timefinal": "10",
        "timestep": "0.1",
        # The amount of basis wfs (spins) that will be combined
        # If N spins are in the system you can either enter a list, which contains the
        # number of wfs that should be combined, e.g. [n1, n2, n3] with n1+n2+n3 = N
        # or a single number A < N, then there will be A combined wfs (approx.) equally distributed
        # If "central_spin_splitted" is true, it will not add to the value N and treated separately.
        "n_combined_wf": 2,
        # how many basis states there should be taken for each combined wf
        # either a list with an item for each combined wf or a number for every wf
        "wave_function_basis": 40,
        # If the central spin is splitted up, it means it is represented in its own space
        # without combining it with other spins into a wave function
        "central_spin_splitted": "False"
    }

    # Write the above sections to config.ini file
    with open('config.ini', 'w') as conf:
        config_object.write(conf)


if __name__ == "__main__":
    create_config()
