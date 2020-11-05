import torch
from torch import nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
import os
import gc
from argparse import ArgumentParser, Namespace
from torchvision.utils import save_image, make_grid


class Flatten(pl.LightningModule):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(pl.LightningModule):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(pl.LightningModule):

    def __init__(self, **hparams):
        super(ConvVAE, self).__init__()

        self.hparams = hparams
        self.mnist_train = None
        self.mnist_test = None
        self.mnist_val = None

        self.input_image_shape_c = self.hparams['model_parameters']['input_image_shape']['input_image_shape_c']
        self.input_image_shape_h = self.hparams['model_parameters']['input_image_shape']['input_image_shape_h']
        self.input_image_shape_w = self.hparams['model_parameters']['input_image_shape']['input_image_shape_w']

        # encoder
        self.latent_dim = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['latent_dim']
        self.conv1_out_channels = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['conv1_out_channels']
        self.conv2_out_channels = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['conv2_out_channels']
        self.linear1_out_layer = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['linear1_out_layer']
        self.kernel_size = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['kernel_size']
        self.stride = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['stride']
        self.padding = self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['padding']
        self.image_reduced_dim = int(self.input_image_shape_w / (self.kernel_size/self.stride))
        if self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['activation'] == 'relu':
            self.encoder_activation = nn.ReLU()
        elif self.hparams['model_parameters']['cnn_vae']['vae_encoder_params']['activation'] == 'sigmoid':
            self.encoder_activation = nn.Sigmoid()

        # decoder
        if self.hparams['model_parameters']['cnn_vae']['vae_decoder_params']['activation'] == 'relu':
            self.decoder_activation = nn.ReLU()
        elif self.hparams['model_parameters']['cnn_vae']['vae_decoder_params']['activation'] == 'sigmoid':
            self.decoder_activation = nn.Sigmoid()

        if self.hparams['model_parameters']['cnn_vae']['vae_decoder_params']['last_activation'] == 'relu':
            self.decoder_last_activation = nn.ReLU()
        elif self.hparams['model_parameters']['cnn_vae']['vae_decoder_params']['last_activation'] == 'sigmoid':
            self.decoder_last_activation = nn.Sigmoid()

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_image_shape_c,
                      self.conv1_out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            self.encoder_activation,
            nn.Conv2d(self.conv1_out_channels,
                      self.conv2_out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding),
            self.encoder_activation,
            Flatten(),
            nn.Linear(int(self.conv2_out_channels*(self.image_reduced_dim/2)**2),
                      self.linear1_out_layer),
            self.encoder_activation
        )

        # hidden => mu
        self.fc1 = nn.Linear(self.linear1_out_layer, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(self.linear1_out_layer, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.linear1_out_layer),
            self.decoder_activation,
            nn.Linear(self.linear1_out_layer,
                      int(self.conv2_out_channels*(self.image_reduced_dim/2)**2)),
            self.decoder_activation,
            Unflatten(self.conv2_out_channels,
                      int(self.image_reduced_dim/2), int(self.image_reduced_dim/2)),
            self.decoder_activation,
            nn.ConvTranspose2d(self.conv2_out_channels,
                               self.conv1_out_channels,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),
            self.decoder_activation,
            nn.ConvTranspose2d(self.conv1_out_channels,
                               self.input_image_shape_c,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               padding=self.padding),
            self.decoder_last_activation
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        # parser.add_argument('--data_dir', type=str, default=str(pathlib.Path('./data')))
        parser.add_argument('--num_workers', type=int, default=(int(os.cpu_count()/2)))
        parser.add_argument('--batch_size', type=int, default=128)
        # # optimizer args
        # # parser.add_argument('--optimizer_type', type=str, default='Adam')
        parser.add_argument('--learning_rate', type=float, default=3e-5)
        # parser.add_argument('--weight_decay', type=float, default=0.0)
        # # parser.add_argument('--look_ahead', action='store_true')
        # # parser.add_argument('--look_ahead_k', type=int, default=5)
        # # parser.add_argument('--look_ahead_alpha', type=float, default=0.5)
        parser.add_argument('--use_lr_scheduler', type=bool, default=True)
        parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.96)

        return parser

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction loss
        BCE = F.binary_cross_entropy(recon_x.view(-1, self.input_image_shape_h*self.input_image_shape_w),
                                     x.view(-1, self.input_image_shape_h*self.input_image_shape_w), reduction='sum')

        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def on_batch_end(self) -> None:
        gc.collect()

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        recon_batch, mu, logvar = self.forward(x)
        loss = self.loss(recon_batch, x, mu, logvar)
        loss /= train_batch[0].shape[0]
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def test_step(self, test_batch, batch_idx):
        x, _ = test_batch
        recon_batch, mu, logvar = self.forward(x)
        loss = self.loss(recon_batch, x, mu, logvar)
        loss /= test_batch[0].shape[0]
        logs = {'test_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        recon_batch, mu, logvar = self.forward(x)
        loss = self.loss(recon_batch, x, mu, logvar)
        loss /= val_batch[0].shape[0]

        out = {'val_loss': loss}

        if batch_idx == 0:
            # n = min(x.size(0), 8)
            # comparison = torch.cat([x[:n], recon_batch.view(64, 1, 28, 28)[:n]]).cpu()
            # img = make_grid(comparison)
            out['reconstruction'] = recon_batch.view(self.hparams["batch_size"],
                                                     self.input_image_shape_c,
                                                     self.input_image_shape_h,
                                                     self.input_image_shape_w)

        return out

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        res = outputs[0]['reconstruction']
        # log image reconstructions
        n = min(64, 8)
        recons = [res.cpu()[:n], res.cpu()[:n]]
        recons.append(res.cpu()[:n])
        recon = torch.cat(recons, 0)
        rg = torchvision.utils.make_grid(
            recon,
            nrow=n, pad_value=0, padding=1
        )
        # self.logger..add_image('recons', rg, self.current_epoch)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor()]) #,
                                        # transforms.Normalize((0.1307,), (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams["batch_size"], num_workers=self.hparams["num_workers"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        return optimizer
