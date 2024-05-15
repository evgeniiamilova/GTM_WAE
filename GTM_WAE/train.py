import sys
from pathlib import Path
import random
import click
import torch
import pickle
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import os.path as osp


sys.path.append("/data1/karina/peptides/src1/models/")  # TODO: redo this crunch to installed module with setup.py
from GTM_WAE.preprocessing import SequenceDataset
from GTM_WAE.model import GTM_WAE
from GTM_WAE.config import read_config


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()  # Set the seed at the beginning of the training script

def save_to_pickle(data, file_path):
    print("saves")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def training_pipeline(config):
    if osp.exists(config["Data"]["full_dataset"]):
        print("Loading full_dataset...")
        full_dataset = load_from_pickle(config["Data"]["full_dataset"])
    else:
        print("Pickle files not found. Creating dataset and vocabulary...")
        full_dataset = SequenceDataset(file_name=config["Data"]["dataset_path"],
                                   sequence_column=config["Data"]["sequence_column"],
                                   vocabulary_path=config["Data"]["vocabulary_path"],
                                   label_column=config["Data"]["label_column"])
        save_to_pickle(full_dataset, config["Data"]["full_dataset"])
    print(full_dataset.max_seq_len)
    print(full_dataset.data.shape)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], torch.Generator().manual_seed(42))
    print(f'train_size: {len(train_dataset)}')
    print(f'val_size: {len(val_dataset)}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["Train"]["batch_size"], pin_memory=True,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["Train"]["batch_size"], pin_memory=True)

    model = GTM_WAE(
        vocab_size=full_dataset.vocabulary_size,
        max_len=full_dataset.max_seq_len,
        lv_dim=config["Train"]["lv_dim"],
        num_heads=config["Train"]["num_heads"],
        num_layers=config["Train"]["num_layers"],
        dropout=config["Train"]["dropout"],
        vae=config["Train"]["vae"],
        lr=config["Train"]["lr"],
        task='train'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger(save_dir="..", name=config["Data"]["logging_folder"])

    checkpoint = ModelCheckpoint(dirpath=config["Data"]["weights_path"], filename=config["Data"]["weights_name"],
                                 monitor="val_loss", mode="min")
    swa = StochasticWeightAveraging(swa_lrs=config["Train"]["lr"], swa_epoch_start=0.95)

    trainer = Trainer(accelerator='gpu', devices=[0], max_epochs=config["Train"]["max_epochs"],
                      callbacks=[lr_monitor, checkpoint, swa],
                      logger=logger, gradient_clip_val=1.0, log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    help="Path to the config YAML file.",
    type=click.Path(exists=True, path_type=Path),
)
def main(config_path):
    config_path = Path(config_path)
    if config_path.exists() and config_path.is_file():
        config = read_config(config_path)
        if config is not None:
            training_pipeline(config)
    else:
        ValueError("The config file does not exist.")


if __name__ == '__main__':
    main()
