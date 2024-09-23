import os
import torch
from kan import KAN
import matplotlib.pyplot as plt
import toolbar_kan as tk

def dataset_prep(data_path, ratio, sample_rate, n_chroma, n_mfcc):
    " this function take path of folder where data are\
    stored and make chroma features from them for\
    every sample in folder, after that save it into the \
    list and stack it into tensor. In the end\
    tensor is split on testing and training dataset and \
    this two tensors are returning back"
    tensors_list = []
    for item in os.listdir(data_path):
        data = tk.data_from_txt(data_path + "\\" + item)
        chroma_mean = tk.chroma(data, sample_rate, n_chroma)
        mfcc_features = tk.mfcc(data, sample_rate, n_mfcc)
        all_features = chroma_mean + mfcc_features
        tensors_list = tk.data_to_tensor(all_features, tensors_list)
    tensors_list = torch.stack(tensors_list)
    tensor_train, tensor_test = torch.split(tensors_list, ratio)

    return tensor_train, tensor_test

path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\healthy",
        "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\unhealthy"]

train_tensor = []
test_tensor = []

# i call function that make dataset for healthy and unhealthy samples and separate
# them to test and train dataset, and this dataset merge sou test contain
# healthy and unhealthy samples
for item in path:
    train, test = dataset_prep(item, [80, 20], 50000, 4, 4)
    train_tensor.append(train)
    test_tensor.append(test)

train_tensor_in = torch.cat(train_tensor, dim=0).double()
test_tensor_in = torch.cat(test_tensor, dim=0).double()

# this make tensors of ones and zeros for train and test dataset
train_tensor_target = torch.cat([torch.ones(80, 1), torch.zeros(80, 1)], dim=0).double()
test_tensor_target = torch.cat([torch.ones(20, 1), torch.zeros(20, 1)], dim=0).double()

# making dictionary of input data
input_dataset = {
    'train_input': train_tensor_in,
    'test_input': test_tensor_in,
    'train_label': train_tensor_target,
    'test_label': test_tensor_target
}

print("Dataset completed")


torch.set_default_dtype(torch.float64)
model = KAN(width=[2, 5, 5, 4], grid=3, seed=42)

# plot kan
model(input_dataset["train_input"])
model.plot()
plt.show()

print("model_defined")

# model training
model.fit(input_dataset, opt="LBFGS", steps=50, lamb=0.001)
model.plot()
plt.show()

print("ende")
