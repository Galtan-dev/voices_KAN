'''
Script contain various function for KAN neural network
 utilization and sound features extraction
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch

def data_from_txt(file_path):
    " loading data from txt file "
    audio_data = np.loadtxt(file_path)
    return audio_data

def chroma(audio_data, sample_rate,n_chroma):
    " calculation of chroma features of data\
    and mean of chroma features"
    chroma_features = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=n_chroma)
    chroma_mean = np.mean(chroma_features, axis=1)
    chroma_mean = chroma_mean.tolist()
    return chroma_mean, chroma_features

def data_to_tensor(data, tensors_list):
    " take data and list of tensors or empty\
     list and and another one tensor made from data "
    tensors_list.append(torch.tensor(data))
    return tensors_list

def chroma_visual(chroma_features, sample_rate):
    " visialization of chroma features "
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma_features, x_axis='time', y_axis='chroma',\
                                                sr=sample_rate, cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma Features')
    plt.tight_layout()
    plt.show()
