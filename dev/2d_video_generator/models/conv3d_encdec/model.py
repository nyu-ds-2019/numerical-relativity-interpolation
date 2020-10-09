import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from .loader import SingleChannelDataset

from .model_conv_1 import Encoder
from .model_conv_1 import Decoder

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

class PlaceholderModel(pl.LightningModule):

    def __init__(self, hparams, data_path):
        super().__init__()
        self.encoder = Encoder(5, F.relu)
        self.decoder = Decoder(5, 72, F.relu)
        self.criterion = nn.MSELoss()

        self.hparams = hparams
        self.data_path = data_path
        
        self.best_loss = 10 ** 6

    def prepare_data(self):

        dataset = SingleChannelDataset(self.data_path)
        train_length = int(0.7 * len(dataset))
        val_length = int(0.15 * len(dataset))
        test_length = len(dataset) - train_length - val_length
#         self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_length, val_length, test_length])
        
        self.train_dataset = Subset(dataset, list(range(0, train_length)))
        self.val_dataset = Subset(dataset, list(range(train_length, train_length+val_length)))
        self.test_dataset = Subset(dataset, list(range(train_length+val_length, len(dataset))))
        
#         print(len(self.train_dataset)), print(len(self.val_dataset)), print(len(self.test_dataset))

    def configure_optimizers(self):
        # REQUIRED

        if self.hparams.lr_type == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.lr_type == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_epoch, self.hparams.scheduler_step_size)
        return [optimizer], [scheduler]

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
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            
        return {'val_loss': avg_loss, 'log': tensorboard_logs, 'progress_bar': {'val_loss' : avg_loss}}

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