import json
import argparse
import os

def load_json(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def read_template(template_path):
    with open(template_path, 'r') as file:
        template = file.read()
    return template

def generate_script(template, config):
    # Flatten the configuration dictionary for easier access
    flat_config = {
        "inp_units": config["system_settings"]["units"],
        "inp_pair_style": config["system_settings"]["pair_style"],
        "inp_num_atoms": config["system_settings"]["num_atoms"],
        "inp_length": config["system_settings"]["length"],
        "inp_pos_seed": config["system_settings"]["pos_seed"],
        "inp_mass": config["potential_settings"]["mass"],
        "inp_epsilon": config["potential_settings"]["epsilon"],
        "inp_sigma": config["potential_settings"]["sigma"],
        "inp_temp": config["thermo_settings"]["temp"],
        "inp_damp": config["thermo_settings"]["damp"],
        "inp_time_step": config["run_settings"]["time_step"],
        "inp_vel_seed": config["run_settings"]["vel_seed"],
        "inp_equil_steps": config["run_settings"]["equil_steps"],
        "inp_prod_steps": config["run_settings"]["prod_steps"],
        "inp_sample_freq": config["sample_settings"]["sample_freq"],
        "inp_output_file": config["sample_settings"]["output_file"]
    }
    return template.format(**flat_config)

def save_script(script, output_path, output_file):
    with open(os.path.join(output_path, output_file), 'w') as file:
        file.write(script)

def main():
    parser = argparse.ArgumentParser(description='Generate LAMMPS script from JSON configuration and template.')
    parser.add_argument('template', type=str, help='Path to the LAMMPS script template file.')
    parser.add_argument('config', type=str, help='Path to the JSON configuration file.')
    parser.add_argument('path', type=str, help='Path to save the generated LAMMPS script.')
    parser.add_argument('--output', type=str, default='input.lammps', help='Filename to save the generated LAMMPS script.')

    args = parser.parse_args()

    config = load_json(args.config)
    template = read_template(args.template)
    script = generate_script(template, config)
    save_script(script, args.path, args.output)
    
    print(f"Generated script saved to {args.output} in {args.path}.")

if __name__ == "__main__":
    main()

