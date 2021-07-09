import pprint
import argparse
import json


def main(casename):
    """ The function generates input files that are necessary for testing."""
    if casename == "colliding_blocks":
        input_dict = colliding_blocks()
    elif casename == "sullivan_and_patton":
        input_dict = sullivan_and_patton()
    elif casename == "stable_bubble":
        input_dict = stable_bubble()
    elif casename == "bomex":
        input_dict = bomex()
    elif casename == "rico":
        input_dict = rico()
    elif casename == "atex":
        input_dict = atex()
    elif casename == "testbed":
        input_dict = testbed()

    write_file(casename, input_dict)
    # Pretty print the output to the terminal
    pprint.pprint(input_dict)

    return


def write_file(casename, input_dict):
    with open("./" + casename + ".json", "w") as input_file_out:
        json.dump(input_dict, input_file_out, sort_keys=True, indent=4)
    return


def stable_bubble():

    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "stable_bubble"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [512, 4, 64]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [51200.0, 51200.0, 6400.0]

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 250.0
    input_dict[key]["timescale"] = 50.0

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "time"
    input_dict[key] = {}
    input_dict[key]["time_max"] = 600.0
    input_dict[key]["cfl"] = 0.6

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "Thermodynamics"
    input_dict[key] = {}
    input_dict[key]["type"] = "dry"

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    return input_dict


def colliding_blocks():
    """ data necessary for the simplest testing of the domain """
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "colliding_blocks"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [256, 256, 5]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [3, 3, 3]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [1000.0, 1000.0, 1000.0]

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 600.0
    input_dict[key]["frequency"] = 600.0

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    return input_dict


def sullivan_and_patton():
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "sullivan_and_patton"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [32, 32, 32]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [5120.0, 5120.0, 2048.0]

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 250.0
    input_dict[key]["timescale"] = 50.0

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "Thermodynamics"
    input_dict[key] = {}
    input_dict[key]["type"] = "dry"

    key = "microphysics"
    input_dict[key] = {}
    input_dict[key]["scheme"] = "base"

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 3600.0 * 3.0

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "restart"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0
    input_dict[key]["restart_simulation"] = False
    input_dict[key]["infile"] = ""

    key = "fields"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    return input_dict


def bomex():
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "bomex"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [64, 64, 100]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [6400.0, 6400.0, 4000.0]

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "microphysics"
    input_dict[key] = {}
    input_dict[key]["scheme"] = "kessler"

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 1000.0
    input_dict[key]["timescale"] = 60.0

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 3600.0 * 6.0

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    key = "restart"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0
    input_dict[key]["restart_simulation"] = False
    input_dict[key]["infile"] = ""

    key = "fields"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0

    return input_dict


def atex():
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "atex"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [64, 64, 100]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [6400.0, 6400.0, 4000.0]

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "microphysics"
    input_dict[key] = {}
    input_dict[key]["scheme"] = "kessler"

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 1000.0
    input_dict[key]["timescale"] = 60.0

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 3600.0 * 6.0

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    key = "restart"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0
    input_dict[key]["restart_simulation"] = False
    input_dict[key]["infile"] = ""

    key = "fields"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0

    return input_dict


def rico():
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "rico"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [128, 128, 100]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [6400.0, 6400.0, 4000.0]

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "microphysics"
    input_dict[key] = {}
    input_dict[key]["scheme"] = "kessler"

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 1000.0
    input_dict[key]["timescale"] = 60.0

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 3600.0 * 24.0

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    key = "restart"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0
    input_dict[key]["restart_simulation"] = False
    input_dict[key]["infile"] = ""

    key = "fields"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0

    return input_dict


def testbed():
    input_dict = {}

    key = "meta"
    input_dict[key] = {}
    input_dict[key]["casename"] = "testbed"
    input_dict[key]["simname"] = input_dict[key]["casename"]
    input_dict[key]["output_directory"] = "./"
    input_dict[key]["random_seed"] = 1

    key = "grid"
    input_dict[key] = {}
    # Set the number of grid points in the domain
    input_dict[key]["n"] = [144, 144, 200]
    # Set the number of halo points in each direct
    input_dict[key]["n_halo"] = [4, 4, 4]
    # Set the domain length, dx will be determined from n and L
    input_dict[key]["l"] = [14400.0, 14400.0, 5000.0]

    key = "scalar_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"
    input_dict[key][input_dict[key]["type"]] = {}

    key = "momentum_advection"
    input_dict[key] = {}
    input_dict[key]["type"] = "weno5"

    key = "sgs"
    input_dict[key] = {}
    input_dict[key]["model"] = "smagorinsky"
    input_dict[key][input_dict[key]["model"]] = {}
    input_dict[key][input_dict[key]["model"]]["cs"] = 0.17
    input_dict[key][input_dict[key]["model"]]["prt"] = 1.0 / 3.0

    key = "microphysics"
    input_dict[key] = {}
    input_dict[key]["scheme"] = "kessler"

    key = "damping"
    input_dict[key] = {}
    input_dict[key]["vars"] = ["u", "v", "w", "s"]
    input_dict[key]["depth"] = 1000.0
    input_dict[key]["timescale"] = 60.0

    key = "time"
    input_dict[key] = {}
    input_dict[key]["cfl"] = 0.6
    input_dict[key]["time_max"] = 3600.0 * 12.0

    key = "stats"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 60.0
    input_dict[key]["modules"] = []

    key = "restart"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0
    input_dict[key]["restart_simulation"] = False
    input_dict[key]["infile"] = ""

    key = "fields"
    input_dict[key] = {}
    input_dict[key]["frequency"] = 600.0

    key = "testbed"
    input_dict[key] = {}
    input_dict[key]["input_filepath"] = "sgp_inputs.nc"
    input_dict[key]["momentum_forcing"] = "geostrophic"

    key = "towers"
    input_dict[key] = {}
    input_dict[key]["location"] = []
    input_dict[key]["frequency"] = 600.0

    return input_dict


LIST_OF_CASES = [
    "colliding_blocks",
    "stable_bubble",
    "sullivan_and_patton",
    "bomex",
    "rico",
    "atex",
    "testbed",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Aarg paraser for generating input files."
    )
    parser.add_argument("casename")
    args = parser.parse_args()

    assert args.casename in LIST_OF_CASES

    main(args.casename)
