# Libs >>>
import os
import tomllib
from rich.console import Console

# Modules >>>
from train_exper import train

# Conf >>>
config_dir = "./expers.toml"


# Main >>>
def main():
    # Making the console
    console = Console()

    # Load the config
    print("Loading the config...")
    with open(config_dir, "rb") as file:
        config = tomllib.load(file)

    # Looping through experiments
    print("Running the experiments...")
    for key in config:
        # Log
        console.rule(f"Running experiment: [bold green]{key}")
        # Run the train function
        train({"exper_name": key, **config[key]})


# Start >>>
if __name__ == "__main__":
    main()
