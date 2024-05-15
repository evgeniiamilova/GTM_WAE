from copy import deepcopy
from pathlib import Path

import click
import yaml

full_config = {
    "Data": {
        "dataset_path": "/path/to/your/csv",
        "full_dataset": "/path/to/your/pkl",
        "vocabulary_path": "/path/to/your/vocabulary",
        "logging_folder": "/path/to/your/training_logs",
        "weights_path": "/path/to/your/weights",
        "weights_name": "name_{val_rec_rate:.2f}",
        "sequence_column": "sequence",
        "label_column": None,
    },
    "Train": {
        "task": 'train',
        "batch_size": 500,
        "lv_dim": 256,
        "max_epochs": 1000,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.2,
        "lr": 5e-4,
        "vae": True,
        "vae_loss_type": "mmdrf",
        "vae_beta": 0.01,
        "lambda_logvar_l1": 1e-3,
        "lambda_logvar_kl": 1e-3,
    },
}

@click.command()
@click.option(
    "-o",
    "--output_dir",
    "output_dir",
    required=True,
    default=".",
    help="Path to the YAML file to create.",
    type=click.Path(path_type=Path)
)
def create_default_config(output_dir: Path):
    """
     A command-line interface function that, when invoked,
     writes the contents of the full_config object to a
     YAML file named "default_config.yaml".
     The purpose of this function is to create a YAML file
     that serves as the default configuration
    """
    if output_dir.exists() and output_dir.is_dir():
        output_path = output_dir.joinpath("default_config.yaml")
    else:
        raise ValueError("Path to the config does not exist.")

    with open(output_path, "w") as file:
        yaml.dump(full_config, file)

def check_config(loaded_config):
    updated_config = deepcopy(full_config)
    for i, v in loaded_config.items():
        for ii, vv in v.items():
            updated_config[i][ii] = vv  # update the default config with user entered parameters

    # check that the paths exist
    updated_config["Data"]["dataset_path"] = Path(updated_config["Data"]["dataset_path"]).resolve(strict=True)
    updated_config["Data"]["logging_folder"] = Path(updated_config["Data"]["logging_folder"]).resolve(strict=True)
    updated_config["Data"]["weights_path"] = Path(updated_config["Data"]["weights_path"]).resolve(strict=True)

    vocab_path = Path(updated_config["Data"]["vocabulary_path"])
    if not vocab_path.exists():
        with open(vocab_path, 'a'):  # Create the file if it doesn't exist
            pass
    updated_config["Data"]["vocabulary_path"] = vocab_path.resolve(strict=True)

    return updated_config


def read_config(config_path):
    with config_path.open("r") as file:
        try:
            config = yaml.safe_load(file)
        except (yaml.YAMLError) as e:
            print(f'Error loading YAML file: {e}')
            return None

        if config is None:
            print("Empty config file.")
            return None

    config = check_config(config)
    return config


if __name__ == '__main__':
    create_default_config()
