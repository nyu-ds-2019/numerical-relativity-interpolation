import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from loader import SingleChannelDataset

from model_conv import Encoder
from model_conv import Decoder

class PlaceholderModel(pl.LightningModule):

    def __init__(self, hparams, data_path):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = nn.MSELoss()

        self.hparams = hparams
        self.data_path = data_path

    def prepare_data(self):

        dataset = SingleChannelDataset(self.data_path)
        train_length = int(0.7 * len(dataset))
        val_length = int(0.15 * len(dataset))
        test_length = len(dataset) - train_length - val_length
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_length, val_length, test_length])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#         return DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    def forward(self, x1, x2):
        # in lightning, forward defines the prediction/inference actions
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        embedding = embedding1 + embedding2
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x1, x2, y = batch

        z = self(x1, x2)
        y_hat = self.decoder(z)

        loss = self.criterion(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i

        tensorboard_logs = {'train_loss': avg_loss}
        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL

        x1, x2, y = batch

        z = self(x1, x2)
        y_hat = self.decoder(z)

        # loss = self.criterion(y_hat, y)
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        # OPTIONAL
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['val_loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i

        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL

        x1, x2, y = batch

        z = self(x1, x2)
        y_hat = self.decoder(z)

        # loss = self.criterion(y_hat, y)
        loss = F.mse_loss(y_hat, y)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        # OPTIONAL
        if self.trainer.num_gpus == 0 or self.trainer.num_gpus == 1:
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        else:
            avg_loss = 0
            i = 0
            for dataloader_outputs in outputs:
                for output in dataloader_outputs['test_loss']:
                    avg_loss += output
                    i += 1

            avg_loss /= i

        tensorboard_logs = {'test_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}