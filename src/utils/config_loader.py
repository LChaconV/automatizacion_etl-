import yaml
def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            print(f"DEBUG: Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Configuration file not found at '{config_path}'.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")