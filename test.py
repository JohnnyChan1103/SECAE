import os
import glob
import numpy as np
import librosa

from dcae import SimpleDCAE
from noisy import get_noisy_sig
from dataloader import YesNoDataset, YesNoPairs, data_preprocessing
from trainer import Trainer, MaskedMSELoss

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CLEAN_PATH = r"D:\Develop\Dataset\clean_testset_wav\clean_testset_wav"
NOISY_PATH = r"D:\Develop\Dataset\noisy_testset_wav\noisy_testset_wav"
device = 'cpu'
model = SimpleDCAE().to(device)
model.load_state_dict(torch.load(f'./model/dcae.pt'))
validset = YesNoPairs(clean_path= CLEAN_PATH,
                         noisy_path= NOISY_PATH)

mel_kwargs = {'n_mels': 128, 'fmax': 5000, 'fmin': 30, 'n_fft': 512, 'hop_length': 256}
valid_loader = DataLoader(validset, batch_size=10, shuffle=False,  collate_fn= lambda x: data_preprocessing(x, mel_kwargs))
test_samples = next(iter(valid_loader))
test_clean, test_noisy, input_lengths, mask_ind = test_samples
print(input_lengths)
model.eval()
test_denoised = model(test_noisy)
test_noise_sep = test_noisy - test_denoised

fig, axes = plt.subplots(1,3, figsize = (20,10))
librosa.display.specshow(test_clean.reshape(-1, test_clean.shape[-1]).detach().numpy(), x_axis = 'time', ax = axes[0], fmax=mel_kwargs['fmax'], fmin = mel_kwargs['fmin'], cmap = 'magma')
librosa.display.specshow(test_noisy.reshape(-1, test_noisy.shape[-1]).detach().numpy(), x_axis = 'time', ax = axes[1], fmax=mel_kwargs['fmax'], fmin = mel_kwargs['fmin'], cmap = 'magma')
librosa.display.specshow(test_denoised.reshape(-1, test_denoised.shape[-1]).detach().numpy(), x_axis = 'time', ax = axes[2], fmax=mel_kwargs['fmax'], fmin = mel_kwargs['fmin'], cmap = 'magma')

axes[0].set_title('Clean Mel Spectrogram')
axes[1].set_title('Noisy Mel Spectrogram')
axes[2].set_title('Denoised Mel Spectrogram')

plt.show()

from utils import  show_denoising
for i in range(len(test_clean)):
    print(f"sample {i+1}")
    show_denoising(i, clean = test_clean[i], noisy = test_noisy[i], denoised=test_denoised[i], separated_noise=test_noise_sep[i],  input_lengths = [input_lengths[i]], fmax = mel_kwargs['fmax'], fmin =mel_kwargs['fmin'],  n_fft=mel_kwargs['n_fft'], hop_length=mel_kwargs['hop_length'])
    plt.show()