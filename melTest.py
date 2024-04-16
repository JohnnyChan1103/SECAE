import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

kwargs = {'n_mels': 128, 'fmax': 5000, 'fmin': 30}
def mel2audio(m, input_lengths, scale ='dB', **mel_kwargs):
    batch_size = m.size(0)
    if not isinstance(m, np.ndarray):
        m = m.detach().numpy()
    if scale == 'dB':
        m = librosa.db_to_power(m)
    return list(map(lambda x: librosa.feature.inverse.mel_to_audio(m[x, :, :input_lengths[x]], sr=16000,**mel_kwargs), list(range(batch_size))))

data, sampling_rate = librosa.load(r'D:\Develop\Dataset\clean_trainset_wav\p226_002.wav', sr=16000)
print(sampling_rate)
Mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data.squeeze(),
                                                         sr=16000,
                                                         n_mels=128,
                                                         fmax=5000,
                                                         fmin=30))
print(Mel.shape)
Mel = torch.Tensor(Mel)
spectrograms = []
len = Mel.shape[0]
spectrograms.append(Mel)
print(spectrograms[0].size())
mel_specs = nn.utils.rnn.pad_sequence(spectrograms,
                                                batch_first=True).unsqueeze(1).transpose(2,3) ## (batch_size, time, n_mels) -> (batch_size, 1, time, n_mels) -> (batch_size, 1, n_mels, time)
fig, axes = plt.subplots(1,1, figsize = (20,10))
librosa.display.specshow(mel_specs.reshape(-1, mel_specs.shape[-1]).detach().numpy(), x_axis = 'time', ax = axes, fmax=5000, fmin = 30, cmap = 'magma')
#plt.show()

sig = mel2audio(mel_specs[0], [len], scale='dB', **kwargs)
librosa.display.waveshow(sig[0], ax=axes[0])