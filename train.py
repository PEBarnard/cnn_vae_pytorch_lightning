import argparse
import pytorch_lightning as pl
from lightning_cnn_vae import ConvVAE
from config.model_param_config import model_config
import os
from argparse import ArgumentParser, Namespace
from os.path import realpath, dirname, join
from pytorch_lightning import Trainer, seed_everything
import sys

BASE_DIR = join(dirname(realpath(__file__)))
CONFIG_DIR = join(BASE_DIR, 'config')
RESULTS_DIR = join(BASE_DIR, 'results')

def main():
    seed_everything(42)
    args = parse_args(sys.argv[1:])
    train(**vars(args))

def train(**training_kwargs):
    model_parameters = model_config(CONFIG_DIR, 'model_param_config.yml')

    training_params = vars(parse_args())
    training_params.update(training_kwargs)

    hparams = dict(model_parameters=model_parameters)
    hparams.update(training_params)

    model = ConvVAE(**hparams)
    trainer = Trainer(**training_params)

    trainer.fit(model)


def parse_args(argv=None):
    argv = argv or []

    parser = ArgumentParser()
    parser = ConvVAE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    main()





