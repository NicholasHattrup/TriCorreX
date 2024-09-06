import sys
import os
import numpy as np
import subprocess
from TriCorreX.utils.dat2array import dat2array


def main():
    # Check if exactly one argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py {g|l|b}")
        sys.exit(1)

    # Assign the positional parameter to a variable
    arg = sys.argv[1]

    # Assign directories based on the argument
    if arg == "g":
        directories = ["gas"]
    elif arg == "l":
        directories = ["liquid"]
    elif arg == "b":
        directories = ["gas", "liquid"]
    else:
        print("Invalid argument. Usage: python script.py {g|l|b}")
        sys.exit(1)

    # Run the simulations
    for directory in directories:
        print(f"Running {directory} simulation...")
        subprocess.run(["python", "lammps.py", "template.txt", f"{directory}/lj_{directory}.json", directory], check=True)
        os.chdir(directory)
        subprocess.run(["mpirun", "-np", "4", "lmp_mpi", "-in", "input.lammps"], stdout=open("output.log", "w"), check=True)
        configs = dat2array("configs.dat") # x y z xu yu zu
        configs_wrapped = configs[:, :, :3]
        configs_unwrapped = configs[:, :, 3:]
        np.save("configs_w.npy", configs_wrapped)
        np.save("configs_uw.npy", configs_unwrapped)
        os.chdir("..")

if __name__ == "__main__":
    main()

