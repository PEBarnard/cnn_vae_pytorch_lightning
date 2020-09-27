import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import shutil
from torchvision.utils import save_image, make_grid
import os
import pytorch_lightning as pl


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

    def __init__(self, latent_size, device):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size
        self.device = device
        # self.optimizer = optimizer
        # self.epochs = epochs

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

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

    def loss_function(self, recon_x, x, mu, logvar):
        # reconstruction loss
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def fit(self, train_loader, optimizer):
        self.train()
        train_loss = 0

        for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
            data = data.to(self.device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)

            loss = self.loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        return train_loss

    def predict(self, epoch, test_loader, writer, args):
        self.eval()
        test_loss = 0

        with torch.no_grad():
            for batch_idx, (data, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
                data = data.to(self.device)

                recon_batch, mu, logvar = self(data)

                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

                if batch_idx == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]).cpu()
                    img = make_grid(comparison)
                    writer.add_image('reconstruction', img, epoch)
                    save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)

        return test_loss

    def save_checkpoint(self, state, is_best, outdir='results'):
        checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
        best_file = os.path.join(outdir, 'model_best.pth')
        torch.save(state, checkpoint_file)
        if is_best:
            shutil.copyfile(checkpoint_file, best_file)
