import argparse
import pytorch_lightning as pl
from lightning_cnn_vae import ConvVAE
from config.model_param_config import model_config
import os
from argparse import ArgumentParser, Namespace
from os.path import realpath, dirname, join
from pytorch_lightning import Trainer, seed_everything
import sys
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from constants import BASE_DIR, CONFIG_DIR, RESULTS_DIR
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    seed_everything(42)
    args = parse_args(sys.argv[1:])
    trainer = train(**vars(args))
    test(trainer)

early_stop_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=1e-20,
                                    patience=25,
                                    verbose=True,
                                    mode='max')

checkpoint_callback = ModelCheckpoint(save_weights_only=False)

def train(**training_kwargs):
    print('\n------------ Model training in progress --------------\n')
    model_parameters = model_config(CONFIG_DIR, 'model_param_config.yml')
    training_params = vars(parse_args())
    training_params.update(training_kwargs)
    hparams = dict(model_parameters=model_parameters)
    hparams.update(training_params)
    exp_name = "conv1_"+str(model_parameters['cnn_vae']['vae_encoder_params']['conv1_out_channels']) + "_conv2_"+str(model_parameters['cnn_vae']['vae_encoder_params']['conv2_out_channels'])
    logger = TensorBoardLogger('logbook_cnnvae', 
                                name=exp_name)

    model = ConvVAE(**hparams)
    trainer = pl.Trainer(max_epochs=1000, 
                        callbacks=[early_stop_callback, checkpoint_callback], 
                        auto_lr_find=False,
                        checkpoint_callback=True,
                        fast_dev_run=False,
                        flush_logs_every_n_steps=100,
                        gpus=1,
                        min_epochs=1,
                        limit_train_batches=1.0,
                        limit_val_batches=1.0,
                        limit_test_batches=1.0,
                        logger=logger)

    trainer.fit(model)
    print('\n------------ Model training completed --------------\n')
    return trainer

def test(trainer):
    print('\n------------ Model testing in progress --------------\n')
    trainer.test()
    print('\n------------ Model testing completed --------------\n')


def parse_args(argv=None):
    argv = argv or []
    parser = ArgumentParser()
    parser = ConvVAE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    return args


if __name__ == '__main__':
    main()





