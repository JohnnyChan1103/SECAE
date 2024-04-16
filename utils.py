import librosa
import os
import numpy as np
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt

## mel-spectrogram to audio signal
import soundfile


def mel2audio(mel, input_lengths, scale = 'dB', **mel_kwargs):
    batch_size = mel.size(0)
    if not isinstance(mel, np.ndarray):
        mel = mel.detach().numpy()
    if scale == 'dB':
        mel = librosa.db_to_power(mel)
    return list(map(lambda x: librosa.feature.inverse.mel_to_audio(mel[x, :, :input_lengths[x]],sr=16000,**mel_kwargs), list(range(batch_size))))

## show denoisied result
def show_denoising(fidx, clean, noisy, denoised, separated_noise, input_lengths,  **inv_mel_kwargs):
    mels = [clean, noisy, denoised, separated_noise]
    types = ['clean', 'noisy', 'denoised', 'noisy - denoised']
    fig, axes = plt.subplots(1,4, figsize = (len(mels)*4,3))
    for i, mel in enumerate(mels):
        sig = mel2audio(mel, input_lengths, scale = 'dB', **inv_mel_kwargs)
        librosa.display.waveshow(sig[0], ax = axes[i])
        axes[i].set_title(types[i])
        print(types[i])
        #ipd.display(ipd.Audio(sig[0], rate=16000))
        soundfile.write(file=os.path.join(r"./denoised_wav", 'Sample '+str(fidx)+'_'+types[i]+r".wav"), data=sig[0], samplerate=16000)
    
    
