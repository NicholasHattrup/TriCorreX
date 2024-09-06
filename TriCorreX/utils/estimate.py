import os 
import yaml
import numpy as np
import sys 
import argparse
from TriCorreX.correlations import g2
from TriCorreX.correlations import g3 
from TriCorreX.utils import transport


def load_configs(file_path):
    '''
    Load the configurations from the file_path
    '''
    return np.load(file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--file', type=str)
    parser.add_argument('--params', type=str, default='params.yaml')   
    parser.add_argument('--compute_msd', type=bool, default=False)
    parser.add_argument('--compute_D', type=bool, default=False)
    parser.add_argument('--time_step', type=float, default=None)
    parser.add_argument('--compute_g2', type=bool, default=False)
    parser.add_argument('--compute_g3', type=bool, default=False)
    args = parser.parse_args()
    directory = args.path
    configs_file = args.file
    params_file = args.params
    compute_msd = args.compute_msd
    compute_D = args.compute_D
    time_step = args.time_step
    compute_g2 = args.compute_g2
    compute_g3 = args.compute_g3
    # Load the configurations
    config_path = os.path.join(directory, configs_file)
    params_path = os.path.join(directory, params_file)
    configs = load_configs(config_path)
    params = yaml.safe_load(open(params_path, 'r'))
    L = params['params']['L']
    rho = params['params']['rho']
    # n_bins and R_max are specific to the g2 and g3 calculations

    # Compute the g2 and g3 correlations
    if compute_g2:
        n_bins = params['params']['g2']['n_bins']
        R_max = params['params']['g2']['r_max']
        if R_max is None:
            R_max = L/2
        rdf, dist = g2.compute_g2(configs, L, R_max, n_bins, rho)
        # Now given dist[i], reassign as dist[i] = (dist[i] + dist[i+1])/2
        dist = (dist[:-1] + dist[1:])/2
        np.save(os.path.join(directory, 'g2.npy'), np.column_stack((rdf, dist)))
    
    if compute_msd:
        msd = transport.compute_msd(configs)
        np.save(os.path.join(directory, 'msd.npy'), msd)
        if compute_D:
            if time_step is None:
                raise ValueError("Time step must be provided to compute the diffusion coefficient.")
            time = np.arange(len(msd)) * time_step
            D = transport.compute_diffusion(msd, time)
            np.save(os.path.join(directory, 'D.npy'), D)

        

    # TO-DO
    # Add g3 functionality 
    return 

if __name__ == '__main__':
    main()




