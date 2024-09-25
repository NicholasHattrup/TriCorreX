import yaml  # Replace json with yaml
import argparse
import os

def load_yaml(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # Use yaml.safe_load to load YAML safely
    return config

def read_template(template_path):
    with open(template_path, 'r') as file:
        template = file.read()
    return template

def generate_script(template, config):
    # Flatten the configuration dictionary and check for None values
    flat_config = {
        "inp_units": config["system_settings"].get("units"),
        "inp_pair_style": config["system_settings"].get("pair_style"),
        "inp_num_atoms": config["system_settings"].get("num_atoms"),
        "inp_length": config["system_settings"].get("length"),
        "inp_pos_seed": config["system_settings"].get("pos_seed"),
        "inp_mass": config["potential_settings"].get("mass"),
        "inp_epsilon": config["potential_settings"].get("epsilon"),
        "inp_sigma": config["potential_settings"].get("sigma"),
        "inp_temp": config["thermo_settings"].get("temp"),
        "inp_damp": config["thermo_settings"].get("damp"),
        "inp_time_step": config["run_settings"].get("time_step"),
        "inp_vel_seed": config["run_settings"].get("vel_seed"),
        "inp_equil_steps": config["run_settings"].get("equil_steps"),
        "inp_prod_steps": config["run_settings"].get("prod_steps"),
        "inp_sample_freq": config["sample_settings"].get("sample_freq"),
        "inp_output_file": config["sample_settings"].get("output_file")
    }

    # Raise an error if any required argument is None
    for key, value in flat_config.items():
        if value is None:
            raise ValueError(f"Missing required value for {key}")

    # Generate the script by formatting the template
    return template.format(**flat_config)


def save_script(script, output_path, output_file):
    with open(os.path.join(output_path, output_file), 'w') as file:
        file.write(script)

def main():
    parser = argparse.ArgumentParser(description='Generate LAMMPS script from YAML configuration and template.')
    parser.add_argument('template', type=str, help='Path to the LAMMPS script template file.')
    parser.add_argument('config', type=str, help='Path to the YAML configuration file.')
    parser.add_argument('path', type=str, help='Path to save the generated LAMMPS script.')
    parser.add_argument('--output', type=str, default='input.lammps', help='Filename to save the generated LAMMPS script.')

    args = parser.parse_args()

    config = load_yaml(args.config)  # Use load_yaml instead of load_json
    template = read_template(args.template)
    script = generate_script(template, config)
    save_script(script, args.path, args.output)
    
    print(f"Generated script saved as {args.output} in {os.path.abspath(args.path)}")

if __name__ == "__main__":
    main()
