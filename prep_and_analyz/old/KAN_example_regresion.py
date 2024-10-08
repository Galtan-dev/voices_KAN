# example script
import torch
from kan import KAN
from kan.utils import create_dataset
import matplotlib.pyplot as plt

# model creation
torch.set_default_dtype(torch.float64)
model = KAN(width=[2, 5, 1], grid=3, seed=42)

# dataset creation (function f(x,y)=exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:, [0]]) + x[:, [1]]**2)
dataset = create_dataset(f, n_var=2)
print(dataset["train_input"].shape)
print(dataset["train_label"].shape)
# print(dataset.shape)
print(dataset.keys())
print(dataset.items())


# plot kan
model(dataset["train_input"])
model.plot()
plt.show()

# model training
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
model.plot()
plt.show()


#
# import torch
#
# # Vytvoření tensoru typu Float (defaultní typ)
# tensor_float = torch.randn(3, 3)
#
# # Převod na typ Double
# tensor_double = tensor_float.double()
#
# print(tensor_float)  # torch.float32
# print(tensor_double)  # torch.float64
