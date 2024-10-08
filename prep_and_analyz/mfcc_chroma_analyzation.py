# import os
# import torch
# from kan import KAN, ex_round
# import matplotlib.pyplot as plt
# import toolbar_kan as tk
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from pathlib import Path
#
#
# def dataset_prep(data_path, ratio, sample_rate, n_chroma, n_mfcc):
#     " this function take path of folder where data are\
#     stored and make chroma features from them for\
#     every sample in folder, after that save it into the \
#     list and stack it into tensor. In the end\
#     tensor is split on testing and training dataset and \
#     this two tensors are returning back"
#     tensors_list = []
#     file = Path(data_path)
#     file_num = len(list(file.glob('*')))
#     ratio_separation = [round(ratio[0]*file_num), round(ratio[1]*file_num)]
#     for item in os.listdir(data_path):
#         data = tk.data_from_txt(data_path + "\\" + item)
#         chroma_mean = tk.chroma(data, sample_rate, n_chroma)
#         mfcc_features = tk.mfcc(data, sample_rate, n_mfcc)
#         all_features = chroma_mean + mfcc_features
#
#         tensors_list = tk.data_to_tensor(all_features, tensors_list)
#     tensors_list = torch.stack(tensors_list)
#     tensor_train, tensor_test = torch.split(tensors_list, ratio_separation)
#
#     return tensor_train, tensor_test, ratio_separation
#
# # Funkce pro standardizaci tensoru
# def standardize_tensor(tensor):
#     scaler = StandardScaler()
#     # Převod tensoru na NumPy pole
#     tensor_np = tensor.numpy()
#     # Standardizace
#     tensor_scaled_np = scaler.fit_transform(tensor_np)
#     # Převod zpět na PyTorch tensor
#     return torch.from_numpy(tensor_scaled_np)
#
# path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\healthy",
#         "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\unhealthy"]
#
# train_tensor = []
# test_tensor = []
#
# # i call function that make dataset for healthy and unhealthy samples_testing and separate
# # them to test and train dataset, and this dataset merge sou test contain
# # healthy and unhealthy samples_testing
# for item in path:
#     train, test, ratio_separation = dataset_prep(item, [0.8, 0.2], 50000, 10, 10)   #13,20,30
#     train_tensor.append(train)
#     test_tensor.append(test)
#
# train_tensor_in = torch.cat(train_tensor, dim=0).double()
# test_tensor_in = torch.cat(test_tensor, dim=0).double()
#
# # this make tensors of ones and zeros for train and test dataset
# train_tensor_target = torch.cat([torch.ones(ratio_separation[0]), torch.zeros(ratio_separation[0])], dim=0).long()
# test_tensor_target = torch.cat([torch.ones(ratio_separation[1]), torch.zeros(ratio_separation[1])], dim=0).long()
#
# # making dictionary of input data
# input_dataset = {
#     'train_input': train_tensor_in,
#     'test_input': test_tensor_in,
#     'train_label': train_tensor_target,
#     'test_label': test_tensor_target
# }
#
# # zamíchá vzorky v datasetu
# indices_train = torch.randperm(input_dataset['train_input'].shape[0])
# indices_test = torch.randperm(input_dataset['test_input'].shape[0])
# input_dataset['train_input'] = input_dataset['train_input'][indices_train]
# input_dataset['train_label'] = input_dataset['train_label'][indices_train]
# input_dataset['test_input'] = input_dataset['test_input'][indices_test]
# input_dataset['test_label'] = input_dataset['test_label'][indices_test]
#
# # Standardizujeme pouze vstupy (train_input a test_input)
# input_dataset['train_input'] = standardize_tensor(input_dataset['train_input'])
# input_dataset['test_input'] = standardize_tensor(input_dataset['test_input'])
#
# torch.set_default_dtype(torch.float64)
# model = KAN(width=[20, 5, 2], grid=3, k=3, seed=0)
# results = model.fit(input_dataset, opt="LBFGS", steps=50, lamb=0.002, loss_fn=torch.nn.CrossEntropyLoss())
# # model.plot(beta=10)
# # plt.show()
#
# print("training completed")
#
# lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
# model.auto_symbolic(lib=lib)
# formula = model.symbolic_formula()[0][0]
# print("Functions fitting completed")
# ex_round(formula, 4)
# print("ex round completed")
#
# # grids = [3, 5, 10, 20, 50]
# #
# # train_rmse = []
# # test_rmse = []
# #
# # for i in range(len(grids)):
# #     model = model.refine(grids[i])
# #     results = model.fit(input_dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
# #     train_rmse.append(results['train_loss'][-1].item())
# #     test_rmse.append(results['test_loss'][-1].item())
# #
# # n_params = np.array(grids) * (2*5+5*20)
# # plt.plot(n_params, train_rmse, marker="o")
# # plt.plot(n_params, test_rmse, marker="o")
# # plt.plot(n_params, 300*n_params**(-2.), color="black", ls="--")
# # plt.legend(['train', 'test', r'$N^{-4}$'], loc="lower left")
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.show()
# # print(train_rmse)
# # print(test_rmse)
# #
#
# # výkonnost modelu
# def model_resolution(formula, ver_input, ver_target):
#     batch = ver_input.shape[0]
#     num_variables = ver_input.shape[1]  # počet proměnných (sloupců) v X
#     TP = 0  # True Positives
#     TN = 0  # True Negatives
#     for i in range(batch):
#         logit = formula
#         # Dynamické dosazování hodnot z ver_input pro každou proměnnou
#         for j in range(num_variables):
#             logit = logit.subs(f'x_{j + 1}', ver_input[i, j])
#
#         logit = np.array(logit).astype(np.float64)
#
#         # Predikce: jestli je logit > 0, považujeme to za pozitivní předpověď (predikce 1)
#         if logit >= -0.44:
#             prediction = round(logit.item())
#
#             # Kontrola predikce proti skutečnému cíli (ver_target[i])
#             if prediction and ver_target[i] == 1:
#                 TP += 1  # Skutečně pozitivní
#             elif not prediction and ver_target[i] == 0:
#                 TN += 1  # Skutečně negativní
#
#     FP = 661-TP
#     FN = 1324-TN
#
#     specificity = TP/(TP+FN)
#     senzitivity = TN/(TN+FP)
#
#     f1 = (specificity+senzitivity)/2
#     print(f"TP: {TP},FP: {FP}, TN: {TN}, FN: {FN}")
#
#     return f1
#
#
# print('F1 score of train:', model_resolution(formula, input_dataset['train_input'], input_dataset['train_label']))
# print('F1 score of test:', model_resolution(formula, input_dataset['test_input'], input_dataset['test_label']))


import os
import torch
from kan import KAN, ex_round
import matplotlib.pyplot as plt
import toolbar_kan as tk
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def dataset_prep(data_path, sample_rate, n_chroma, n_mfcc):
    "Tato funkce vezme cestu k adresáři, kde jsou data uložena, a vytvoří chroma vlastnosti pro každou ukázku."
    tensors_list = []
    labels = []  # Seznam pro uchování štítků
    for item in os.listdir(data_path):
        data = tk.data_from_txt(data_path + "\\" + item)
        chroma_mean = tk.chroma(data, sample_rate, n_chroma)
        mfcc_features = tk.mfcc(data, sample_rate, n_mfcc)
        all_features = chroma_mean + mfcc_features

        tensors_list = tk.data_to_tensor(all_features, tensors_list)

        # Předpokládáme, že název souboru určuje štítek
        label = 1 if 'healthy' in item else 0  # Zde bys měl upravit podle skutečného názvosloví
        labels.append(label)

    tensors_list = torch.stack(tensors_list)
    labels_tensor = torch.tensor(labels)

    return tensors_list, labels_tensor


# Funkce pro standardizaci tensoru
def standardize_tensor(tensor):
    scaler = StandardScaler()
    tensor_np = tensor.numpy()
    tensor_scaled_np = scaler.fit_transform(tensor_np)
    return torch.from_numpy(tensor_scaled_np)


path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\healthy",
        "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\unhealthy"]

# Inicializace prázdných seznamů pro tréninkové a testovací tensory
all_tensors = []
all_labels = []

# Vytvoření datasetu pro zdravé a nezdravé vzorky
for item in path:
    tensors, labels = dataset_prep(item, 50000, 12, 30)
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
split_index = int(0.8 * all_tensors.shape[0])
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

# Nastavení výchozího typu pro torch
torch.set_default_dtype(torch.float64)

# Trénink modelu
model = KAN(width=[42, 20, 2], grid=35, k=3, seed=0)
results = model.fit(input_dataset, opt="LBFGS", steps=50, lamb=0.002, loss_fn=torch.nn.CrossEntropyLoss())

print("training completed")

# Symbolické funkce
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
model.auto_symbolic(lib=lib)
formula = model.symbolic_formula()[0][0]
print("Functions fitting completed")
ex_round(formula, 4)
print("ex round completed")


# Výkonnost modelu
def model_resolution(formula, ver_input, ver_target):
    batch = ver_input.shape[0]
    num_variables = ver_input.shape[1]  # počet proměnných (sloupců) v X
    TP = 0  # True Positives
    TN = 0  # True Negatives
    for i in range(batch):
        logit = formula
        # Dynamické dosazování hodnot z ver_input pro každou proměnnou
        for j in range(num_variables):
            logit = logit.subs(f'x_{j + 1}', ver_input[i, j])

        logit = np.array(logit).astype(np.float64)
        # print(logit)
        # Predikce: jestli je logit > 0, považujeme to za pozitivní předpověď (predikce 1)
        if logit >= -0.44:
            prediction = round(logit.item())

            # Kontrola predikce proti skutečnému cíli (ver_target[i])
            if prediction and ver_target[i] == 1:
                TP += 1  # Skutečně pozitivní
            elif not prediction and ver_target[i] == 0:
                TN += 1  # Skutečně negativní

    FP = 661 - TP
    FN = 1324 - TN

    specificity = TP / (TP + FN)
    senzitivity = TN / (TN + FP)

    f1 = (specificity + senzitivity) / 2
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    return f1


# Výpočet F1 skóre
print('F1 score of train:', model_resolution(formula, input_dataset['train_input'], input_dataset['train_label']))
print('F1 score of test:', model_resolution(formula, input_dataset['test_input'], input_dataset['test_label']))
