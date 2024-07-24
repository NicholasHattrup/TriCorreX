import os 
import yaml
import numpy as np
import sys 
import argparse
from TriCorreX.correlations import g2
from TriCorreX.correlations import g3 

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
    parser.add_argument('--compute_g2', type=bool, default=True)
    parser.add_argument('--compute_g3', type=bool, default=True)
    args = parser.parse_args()
    directory = args.path
    configs_file = args.file
    params_file = args.params
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

    # TO-DO
    # Add g3 functionality 
    return 

if __name__ == '__main__':
    main()




