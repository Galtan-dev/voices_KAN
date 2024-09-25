import os
import torch
from kan import KAN, ex_round
import matplotlib.pyplot as plt
import toolbar_kan as tk
import numpy as np
from sklearn.preprocessing import StandardScaler


def dataset_prep(data_path, ratio, sample_rate, n_chroma, n_mfcc):
    " this function take path of folder where data are\
    stored and make chroma features from them for\
    every sample in folder, after that save it into the \
    list and stack it into tensor. In the end\
    tensor is split on testing and training dataset and \
    this two tensors are returning back"
    tensors_list = []
    # scaler = StandardScaler()
    for item in os.listdir(data_path):
        data = tk.data_from_txt(data_path + "\\" + item)
        chroma_mean = tk.chroma(data, sample_rate, n_chroma)
        mfcc_features = tk.mfcc(data, sample_rate, n_mfcc)
        all_features = chroma_mean + mfcc_features
        # all_features = scaler.fit_transform(all_features)
        tensors_list = tk.data_to_tensor(all_features, tensors_list)
    tensors_list = torch.stack(tensors_list)
    tensor_train, tensor_test = torch.split(tensors_list, ratio)

    return tensor_train, tensor_test

# Funkce pro standardizaci tensoru
def standardize_tensor(tensor):
    scaler = StandardScaler()
    # Převod tensoru na NumPy pole
    tensor_np = tensor.numpy()
    # Standardizace
    tensor_scaled_np = scaler.fit_transform(tensor_np)
    # Převod zpět na PyTorch tensor
    return torch.from_numpy(tensor_scaled_np)

path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\healthy",
        "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\unhealthy"]

train_tensor = []
test_tensor = []

# i call function that make dataset for healthy and unhealthy samples_testing and separate
# them to test and train dataset, and this dataset merge sou test contain
# healthy and unhealthy samples_testing
for item in path:
    train, test = dataset_prep(item, [80, 20], 50000, 10, 10)   #13,20,30
    train_tensor.append(train)
    test_tensor.append(test)

train_tensor_in = torch.cat(train_tensor, dim=0).double()
test_tensor_in = torch.cat(test_tensor, dim=0).double()

# this make tensors of ones and zeros for train and test dataset
train_tensor_target = torch.cat([torch.ones(80), torch.zeros(80)], dim=0).long()
test_tensor_target = torch.cat([torch.ones(20), torch.zeros(20)], dim=0).long()

# making dictionary of input data
input_dataset = {
    'train_input': train_tensor_in,
    'test_input': test_tensor_in,
    'train_label': train_tensor_target,
    'test_label': test_tensor_target
}

# Standardizujeme pouze vstupy (train_input a test_input)
input_dataset['train_input'] = standardize_tensor(input_dataset['train_input'])
input_dataset['test_input'] = standardize_tensor(input_dataset['test_input'])

torch.set_default_dtype(torch.float64)
model = KAN(width=[20, 5, 2], grid=3, k=3, seed=0)
results = model.fit(input_dataset, opt="LBFGS", steps=50, lamb=0.002, loss_fn=torch.nn.CrossEntropyLoss())
model.plot(beta=10)
plt.show()

lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','tan','abs']
model.auto_symbolic(lib=lib)
formula = model.symbolic_formula()[0][0]
print(ex_round(formula, 4))

# grids = [3, 5, 10, 20, 50]
#
# train_rmse = []
# test_rmse = []
#
# for i in range(len(grids)):
#     model = model.refine(grids[i])
#     results = model.fit(input_dataset, opt="LBFGS", steps=50, stop_grid_update_step=30)
#     train_rmse.append(results['train_loss'][-1].item())
#     test_rmse.append(results['test_loss'][-1].item())
#
# n_params = np.array(grids) * (2*5+5*20)
# plt.plot(n_params, train_rmse, marker="o")
# plt.plot(n_params, test_rmse, marker="o")
# plt.plot(n_params, 300*n_params**(-2.), color="black", ls="--")
# plt.legend(['train', 'test', r'$N^{-4}$'], loc="lower left")
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
# print(train_rmse)
# print(test_rmse)
#



