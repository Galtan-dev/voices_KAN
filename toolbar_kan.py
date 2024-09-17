'''
Script contain various function for KAN neural network
 utilization and sound features extraction
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import os

def data_from_txt(file_path):
    " loading data from txt file, it load record specified by \
     number and if its healthy or unhealthy humens "
    if os.path.isfile(file_path):
        # print(f"Nahrávám: {file_path}")
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

def mfcc(data, sample_rate, n_mffc):
    " take data and make mffc dfeatures from them,\
    on the end reshape it into [1,n] format "
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mffc).T, axis=0)
    mfcc = np.reshape(mfcc, [1, n_mffc])

    return mfcc

def dataset_prep(data_path, ratio, sample_rate, n_chroma, n_mfcc):
    " this function take path of folder where data are\
    stored and make chroma features from them for\
    every sample in folder, after that save it into the \
    list and stack it into tensor. In the end\
    tensor is split on testing and training dataset and \
    this two tensors are returning back"
    tensors_list = []
    for item in os.listdir(data_path):
        data = data_from_txt(data_path + "\\" + item)
        # calculation of chroma and mfcc features
        mfcc_features = mfcc(data, sample_rate, n_mfcc)
        chroma_mean, chroma_features = chroma(data, sample_rate, n_chroma)
        # merging lists of input features
        tensors_list = chroma_mean + mfcc_features
        # converting lists to tensors
        tensors_list = data_to_tensor(tensors_list, tensors_list)
    tensors_list = torch.stack(tensors_list)
    tensor_train, tensor_test = torch.split(tensors_list, ratio)

    return tensor_train, tensor_test
