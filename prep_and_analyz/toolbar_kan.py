'''
Script contain various function for KAN neural network
 utilization and sound features extraction
'''

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import os
from sklearn.preprocessing import StandardScaler
import pathlib as Path
from kan import ex_round
import cmath

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

    return chroma_mean
    #, chroma_features

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
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mffc).T, axis=0).tolist()

    return mfcc

def dataset_prep(data_path, sample_rate, n_chroma, n_mfcc):
    "Tato funkce vezme cestu k adresáři, kde jsou data uložena, a vytvoří chroma vlastnosti pro každou ukázku."
    tensors_list = []
    labels = []  # Seznam pro uchování štítků
    for item in os.listdir(data_path):
        data = data_from_txt(data_path + "\\" + item)
        chroma_mean = chroma(data, sample_rate, n_chroma)
        mfcc_features = mfcc(data, sample_rate, n_mfcc)
        all_features = chroma_mean + mfcc_features

        tensors_list = data_to_tensor(all_features, tensors_list)

        # Předpokládáme, že název souboru určuje štítek
        label = 1 if 'healthy' in item else 0  # Zde bys měl upravit podle skutečného názvosloví
        labels.append(label)

    tensors_list = torch.stack(tensors_list)
    labels_tensor = torch.tensor(labels)

    return tensors_list, labels_tensor

def standardize_tensor(tensor):
    " Function take tensor and standardize him"
    scaler = StandardScaler()
    # Převod tensoru na NumPy pole
    tensor_np = tensor.numpy()
    # Standardizace
    tensor_scaled_np = scaler.fit_transform(tensor_np)
    # Převod zpět na PyTorch tensor
    return torch.from_numpy(tensor_scaled_np)


def model_resolution(formula, positive_num, negative_num, ver_input, ver_target):
    " Take calculated function and compare results\
    from validation set with real values and calc F1 score"
    batch = ver_input.shape[0]
    # take number of variables
    num_variables = ver_input.shape[1]
    TP = 0
    TN = 0
    # dynamical assigning of variables
    for i in range(batch):
        logit = formula
        for j in range(num_variables):
            logit = logit.subs(f'x_{j + 1}', ver_input[i, j])
        # if logit.imag != 0:
        #     print("----")
        #     print(logit)
        #     logit = abs(logit)
        #     print(logit)
        #     print("----")
        # else:
        #     print("----")
        #     print(logit)
        #     print("----")
        logit = np.array(logit)#.astype(np.float64)
        # contiton for tru positive and true negative calc
        if logit >= 0:
            prediction = round(logit.item())
            if prediction and ver_target[i] == 1:
                TP += 1
            elif not prediction and ver_target[i] == 0:
                TN += 1
    FN = positive_num - TP
    FP = negative_num - TN
    specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN)
    F1_score = (specificity+sensitivity)/2
    # print(f"TP: {TP},FP: {FP}, TN: {TN}, FN: {FN}")

    return F1_score


def target_generator(ratio_separation):
    "function for generation of tensor of targets for dataset"
    train_tensor_target = torch.cat([torch.ones(ratio_separation[0]), torch.zeros(ratio_separation[0])], dim=0).long()
    test_tensor_target = torch.cat([torch.ones(ratio_separation[1]), torch.zeros(ratio_separation[1])], dim=0).long()

    return train_tensor_target, test_tensor_target


def dataset_generator(sampling_frequnci, n_chroma, n_mfcc, ratio):
    path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\healthy",
            "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\unhealthy"]

    # Inicializace prázdných seznamů pro tréninkové a testovací tensory
    all_tensors = []
    all_labels = []

    # Vytvoření datasetu pro zdravé a nezdravé vzorky
    for item in path:
        tensors, labels = dataset_prep(item, sampling_frequnci, n_chroma, n_mfcc)
        all_tensors.append(tensors)
        all_labels.append(labels)

    # Slučování tensorů a štítků
    all_tensors = torch.cat(all_tensors, dim=0).double()
    all_labels = torch.cat(all_labels, dim=0).long()

    # Zamíchání vzorků v datasetu, abychom měli smíšené pozitivní a negativní vzorky
    indices = torch.randperm(all_tensors.shape[0])
    all_tensors = all_tensors[indices]
    all_labels = all_labels[indices]

    # Rozdělení na tréninkovou a testovací sadu (80% trénink, 20% test)
    split_index = int(ratio * all_tensors.shape[0])
    train_tensor_in = all_tensors[:split_index]
    test_tensor_in = all_tensors[split_index:]
    train_tensor_target = all_labels[:split_index]
    test_tensor_target = all_labels[split_index:]

    # Vytvoření slovníku s daty
    input_dataset = {
        'train_input': train_tensor_in,
        'test_input': test_tensor_in,
        'train_label': train_tensor_target,
        'test_label': test_tensor_target
    }

    # Standardizace pouze vstupů (train_input a test_input)
    input_dataset['train_input'] = standardize_tensor(input_dataset['train_input'])
    input_dataset['test_input'] = standardize_tensor(input_dataset['test_input'])

    return input_dataset


def function_fitting(model):
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
    model.auto_symbolic(lib=lib)
    formula = model.symbolic_formula()[0][0]
    ex_round(formula, 4)

    return(formula)