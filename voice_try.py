import torch
from kan import KAN
from kan.utils import create_dataset
import matplotlib.pyplot as plt
import numpy as np


# Funkce pro načtení dat z textového souboru
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            data.append(values)
    return torch.tensor(data)

file_path = 'C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\healthy\\svdadult0001_healthy_50000.txt'
y = load_data(file_path)
x = np.arange(len(y))
print(y.shape)

# model creation
torch.set_default_dtype(torch.float64)
model = KAN(width=[2, 5, 1], grid=3, seed=42)

# plot kan
model(dataset)
model.plot()
plt.show()

# model training
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model.plot()
plt.show()