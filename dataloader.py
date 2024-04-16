import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import librosa
import librosa.display


## data directory
# DATA_DIR = "./waves_yesno/"
# NOISE_DIR = "./noisesB/"


## YesNo Dataset(original speech signal)
class YesNoDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.file_list = glob.glob(os.path.join(self.dir_path, "*.wav"))
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        sig, sr = librosa.load(self.file_list[idx])
        label = np.array(self.file_list[idx].split("/")[-1].split(".wav")[0].split('_'), dtype = 'int16')
        return sig, sr, label

## YesNoPairs Dataset(original signal + noisy signal)
class YesNoPairs(Dataset):
    def __init__(self, clean_path, noisy_path, sr = 16000):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        #self.clean_flist = glob.glob(self.clean_path + '*.wav')
        # 读文件夹下每个数据文件的名称
        self.clean_file_name = os.listdir(clean_path)
        self.noisy_file_name = os.listdir(noisy_path)

        self.clean_flist = []
        self.noisy_flist = []
        # 把每一个文件的路径拼接起来
        for index in range(len(self.clean_file_name)):
            self.clean_flist.append(os.path.join(clean_path, self.clean_file_name[index]))
        for index in range(len(self.noisy_file_name)):
            self.noisy_flist.append(os.path.join(noisy_path, self.noisy_file_name[index]))
        print(len(self.clean_flist))
        self.sr = sr
    
    def __len__(self):
        #return len(self.clean_flist)
        return 1000
    
    def __getitem__(self, idx):
        clean_fpath = self.clean_flist[idx]
        clean_fn = clean_fpath.replace(self.clean_path, '')
        #noisy_fpath = glob.glob(os.path.join( self.noisy_path, f"noisy_*_{clean_fn}"))[0]
        noisy_fpath = self.noisy_flist[idx]
        clean_sig , sr = librosa.load( clean_fpath, sr=self.sr)
        noisy_sig, sr = librosa.load( noisy_fpath, sr = self.sr)
        file_name = self.clean_file_name[idx]
        return clean_sig, noisy_sig, sr, file_name
        
## get mask indices
def get_mask_ind(x, input_lengths):    
    mask_ind = torch.zeros_like(x)
    for b in range(x.shape[0]):
        x_size = input_lengths[b]
        mask_ind[b, :, :, x_size:] = 1 
    return mask_ind

## batch preprocessing
def data_preprocessing(batch, mel_kwargs):
    clean_spectrograms = []
    noisy_spectrograms = []
    #labels = []
    input_lengths = []
    fname = []
    #label_lengths = []
    
    for sig, noisy, sr, fname in batch:
        M_clean = librosa.power_to_db(librosa.feature.melspectrogram(y=sig.squeeze(), sr=sr, **mel_kwargs))
        M_noisy = librosa.power_to_db(librosa.feature.melspectrogram(y=noisy.squeeze(), sr=sr, **mel_kwargs))
        M_clean = torch.Tensor( M_clean).squeeze(0).transpose(0,1) ## (1, n_mels, time) -> (n_mels, time) -> (time, n_mels)
        M_noisy = torch.Tensor( M_noisy).squeeze(0).transpose(0,1)
        
        clean_spectrograms.append(M_clean)
        noisy_spectrograms.append(M_noisy)
        
        #labels.append(torch.Tensor(label) + 1)
        input_lengths.append(M_clean.shape[0])
        #label_lengths.append(len(label))

    clean_mel_specs = nn.utils.rnn.pad_sequence(clean_spectrograms,
                                                batch_first=True).unsqueeze(1).transpose(2,3) ## (batch_size, time, n_mels) -> (batch_size, 1, time, n_mels) -> (batch_size, 1, n_mels, time)
    noisy_mel_specs = nn.utils.rnn.pad_sequence(noisy_spectrograms,
                                                batch_first=True).unsqueeze(1).transpose(2,3)
    
    #labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    mask_ind = get_mask_ind(clean_mel_specs, input_lengths) # padding indices (1 if pad else 0)
    
    return clean_mel_specs, noisy_mel_specs, input_lengths, mask_ind