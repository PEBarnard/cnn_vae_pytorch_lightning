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
    # parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
    # # parser.add_argument('--result_dir', type=str, default='results', metavar='DIR',
    # #                     help='output directory')
    # parser.add_argument('--batch_size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 128)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: None')

    # model options
    # parser.add_argument('--latent_size', type=int, default=32, metavar='N',
    #                     help='latent vector size of encoder')

    args = parse_args()

    # torch.manual_seed(args.seed)

        # with torch.no_grad():
        #     sample = torch.randn(64, 32).to(device)
        #     sample = model.decode(sample).cpu()
        #     img = make_grid(sample)
        #     writer.add_image('sampling', img, epoch)
        #     save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
    # train
    model_parameters = model_config(CONFIG_DIR, 'model_param_config.yml')
    train(model_parameters, **vars(args))
    # model = ConvVAE(**model_parameters)
    # training_params = vars(parse_args())
    # trainer = Trainer(**training_params)
    #
    # trainer.fit(model)


def train(model_params, **training_kwargs):
    # model_config = make_config(**model_params)

    training_params = vars(parse_args())
    training_params.update(training_kwargs)

    # hparams = dict(model_config=model_config)
    # hparams.update(training_params)

    # model_parameters = model_config(CONFIG_DIR, 'model_param_config.yml')
    model = ConvVAE(**model_params)
    training_params = vars(parse_args())
    trainer = Trainer(**training_params)

    trainer.fit(model)


def parse_args(argv=None):
    argv = argv or []

    parser = ArgumentParser()

    # # add model specific args
    parser = ConvVAE.add_model_specific_args(parser)

    # add all the available trainer options to parser
    parser = Trainer.add_argparse_args(parser)

    # add other args
    # parser.add_argument('--save_top_k', type=int, default=1)

    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    main()





