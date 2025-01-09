# Libs >>>
import os
import tomllib
from rich import print

# Modules >>>
from train_exper import train

# Conf >>>
config_file_name = "expers.toml"
Conf_base_dir = "."

# Config loader >>>
def load_config(raise_if_err: bool = True, file_name: str = None) -> dict:
    """
    Loads a configuration from a TOML file.

    Args:
        raise_if_err (bool, optional): If True, raises any exceptions that occur while loading the configuration. If False, returns a dictionary with an "Error" key containing the error message. Defaults to True.
        file_name (str, optional): The name of the TOML file to load. If not provided, the function will use the default file name.

    Returns:
        dict: The configuration loaded from the TOML file.
    """
    try:
        with open(os.path.join(Conf_base_dir, file_name), "rb") as file:
            # End
            return tomllib.load(file)
    except Exception as err:
        if raise_if_err:
            raise err
        else:
            # End
            return {"Error": str(err)}

# Main >>>
def main():
    # Load expers
    print(f"[bold green]Loading expers from {config_file_name}...")
    expers = load_config(file_name=config_file_name)
    print("[underline]Expers loaded successfully.")
    
    # Run the expers
    print("[bold green]Starting expers...")
    for exp_name, exp_conf in expers.items():
        print(f"[bold green]Starting exper {exp_name}...\n[default]---------------------------------------------------------------------------")
        train(exp_conf)
        print(f"[bold green]Finished exper {exp_name}.")

# Start >>>
if __name__ == "__main__":
    main()