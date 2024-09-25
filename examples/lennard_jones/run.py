import sys
import argparse
import os
import numpy as np
import subprocess
import yaml
from TriCorreX.utils.dat2array import dat2array
from TriCorreX.correlations.g2 import compute_g2
from TriCorreX.transport.einstein import compute_msd, compute_diffusion

def main():
    parser = argparse.ArgumentParser(description='Run LAMMPS simulations and post-process the results.')

    parser.add_argument('--dir', type=str, nargs="*", help='Path to the directory containing the input files.', default=None)
    parser.add_argument('--g', action='store_true', help='Run the gas simulation.')
    parser.add_argument('--l', action='store_true', help='Run the liquid simulation.')
    parser.add_argument('--b', action='store_true', help='Run both gas and liquid simulations.')
    parser.add_argument('--key', type=str, help='job file keyword', default=None)

    args = parser.parse_args()

    jobs = []
    if args.dir:
        jobs = args.dir
    if args.g:
        jobs.append('gas')
    if args.l:
        jobs.append('liquid')
    if args.b:
        jobs.append(['gas', 'liquid'])
    if not jobs:
        sys.exit('No jobs specified. Exiting...')
    key = args.key
    if not key:
        key = 'input.yaml'

    cwd = os.getcwd()
    # Run the simulations
    for job in jobs:
        # Find all input files in job directory
        files = [f for f in os.listdir(job) if os.path.isfile(os.path.join(job, f)) and key in f and '.yaml' in f]
        print(key, files)
        if not files:
            print(f"No input files found in {job}. Exiting...")
            continue
        print(f"Running {len(files)} simulation(s) in {job}...")
        for idx, f in enumerate(files):
            subprocess.run(["python", "lammps.py", "template.txt", f"{job}/{f}", job, "--output", f"{f[:-5]}.lammps"], check=True)
            params = yaml.load(open(f"{job}/{f}"), Loader=yaml.FullLoader)
            time_step = params["run_settings"]["time_step"]
            os.chdir(job)
            subprocess.run(["mpirun", "-np", "8", "lmp_mpi", "-in", f"{f[:-5]}.lammps"], stdout=open("output.log", "w"), check=True)
            print(f"{idx+1} simulation completed.")
            configs, times = dat2array("configs.dat") # x y z xu yu zu
            times -= times[0]
            times *= time_step
            configs_wrapped = configs[:, :, :3]
            configs_unwrapped = configs[:, :, 3:]
            np.save("configs_w.npy", configs_wrapped)
            np.save("configs_uw.npy", configs_unwrapped)
            print('Calculating g2...')
            params = yaml.load(open(f"params.yaml"), Loader=yaml.FullLoader)
            L, rho = params["L"], params["rho"]
            r_max, n_bins = params['g2']['r_max'], params['g2']['n_bins']
            dist, rdf = compute_g2(configs_wrapped, L=L, R_max=r_max, num_bins=n_bins, rho=rho, workers=8)
            dist = (dist[:-1] + dist[1:])/2
            np.save(f"g2_{idx+1}.npy", np.column_stack((dist, rdf)))
            print('Calculating MSD...')
            msd = compute_msd(configs_unwrapped)
            np.save(f"msd_{idx+1}.npy", msd)
            print('Calculating diffusion coefficient...')
            D = compute_diffusion(msd, times)
            np.save(f"D_{idx+1}.npy", D)
            # clean up .dat files
            os.remove("configs.dat")
            os.remove("output.log")
            os.remove("configs_w.npy")
            os.remove("configs_uw.npy")
            os.chdir(cwd)

    print('Finished running and post-processing simulations.')
if __name__ == "__main__":
    main()

