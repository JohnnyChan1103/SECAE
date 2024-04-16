import os
import glob
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from dcae import SimpleDCAE
from noisy import get_noisy_sig
from dataloader import YesNoDataset, YesNoPairs, data_preprocessing
from trainer import Trainer, MaskedMSELoss

import torch
from torch.utils.data import DataLoader
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

CLEAN_PATH = r"D:\Develop\Dataset\clean_trainset_wav"
NOISY_PATH = r"D:\Develop\Dataset\noisy_trainset_wav"
dataset = YesNoPairs(clean_path= CLEAN_PATH,
                         noisy_path= NOISY_PATH)
torch.manual_seed(0)
trainset, validset = torch.utils.data.random_split(dataset, [900,100] )

mel_kwargs = {'n_mels': 128, 'fmax': 5000, 'fmin': 30, 'n_fft': 512, 'hop_length': 256}
train_loader = DataLoader(trainset, batch_size=6, shuffle=True,  collate_fn= lambda x: data_preprocessing(x, mel_kwargs))
valid_loader = DataLoader(validset, batch_size=6, shuffle=False,  collate_fn= lambda x: data_preprocessing(x, mel_kwargs))

# Define model, crit, optimizer, ...
device = 'cuda:0'
model = SimpleDCAE().to(device)
crit = MaskedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,factor=.5,  patience = 2,  min_lr=1e-5, verbose  =True )
n_epochs = 40


# Define trainer
trainer = Trainer(model, crit= crit, optimizer = optimizer, scheduler = scheduler)


# Training model
train_loss = []
val_loss = []
for epoch in range(n_epochs):
    total_train_loss = trainer.train(train_loader, device)
    total_val_loss = trainer.validate(valid_loader, device)
    trainer.scheduler.step(total_val_loss)
    print(f"Epoch {epoch+1}: train_loss: {total_train_loss/len(train_loader)} val_loss : {total_val_loss/len(valid_loader)}")
    train_loss.append(total_train_loss/len(train_loader))
    val_loss.append(total_val_loss/len(valid_loader))
    torch.save(model.state_dict(), f'./model/dcae.pt')

fig, ax = plt.subplots()
x = np.linspace(0, n_epochs, 1)
ax.plot(train_loss, label='Train Loss')
ax.plot(val_loss, label='Validate Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

plt.show()
