import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Parameters:
    config_path (str): Path to the configuration file.

    Returns:
    dict: Configuration settings.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def generate_script(output_path, script_name, settings):
    """
    Generate a script with the provided settings.

    Parameters:
    output_path (str): Directory where the script will be saved.
    script_name (str): Name of the script file.
    settings (dict): Dictionary of settings and variables to include in the script.

    Returns:
    str: Path to the generated script.
    """
    script_content = "# Generated script\n"
    print(settings)
    for key, value in settings.items():
        script_content += f"{key} = {value}\n"

    script_path = os.path.join(output_path, script_name)
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)
    
    logger.info(f"Script generated at {script_path}")
    return script_path

def main(config_path, output_path):
    """
    Main function to load configuration and generate the script.

    Parameters:
    config_path (str): Path to the configuration file.
    output_path (str): Directory where the script will be saved.
    """
    config = load_config(config_path)
    print(config)
    
    script_name = config.get("script_name", "default_script.py")
    settings = config.get("settings", {})

    generate_script(output_path, script_name, settings)

if __name__ == "__main__":
    CONFIG_PATH = "settings.json"
    OUTPUT_PATH = "."
    
    
    main(CONFIG_PATH, OUTPUT_PATH)
