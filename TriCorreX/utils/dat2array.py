import numpy as np 

def dat2array(file_path):
    with open(file_path, 'r') as file:
        header_lines = 0
        while True:
            line = file.readline()
            if line.startswith('ITEM: NUMBER OF ATOMS'):
                num_atoms = int(file.readline())
            if line.startswith('ITEM: ATOMS'):
                cols = line.split()[2:]
                header_lines += 1
                break
            header_lines += 1
        total_lines = header_lines
        while True:
            line = file.readline()
            if not line:
                break
            total_lines += 1
        total_configs = total_lines // (num_atoms + header_lines)
        file.seek(0)
        times = np.zeros(total_configs)
        data = np.zeros((total_configs, num_atoms, len(cols)-1))
        pos = np.zeros((num_atoms, len(cols)-1))
        config = 0
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith('ITEM: ATOMS'):
                for i in range(num_atoms):
                    line = file.readline().split()
                    idx, vals = int(line[0]), [float(val) for val in line[1:]]
                    pos[idx-1] = vals
                    # first column is atom id, ensure it is always ordered 
                data[config] = pos
                config += 1
        file.seek(0)
        config = 0
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith('ITEM: TIMESTEP'):
                times[config] = float(file.readline().split()[0])
                config += 1
    return data, times

