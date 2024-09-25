import toolbar_kan as tbk
import torch
import matplotlib.pyplot as plt
from kan import KAN


path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\healthy",
        "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing\\unhealthy"]

train_tensor = []
test_tensor = []

for item in path:
    tensor_train, tensor_test = tbk.dataset_prep(item, [80, 20], 50000, 4, 4)
    train_tensor.append(train_tensor)
    test_tensor.append(test_tensor)

train_tensor_in = torch.cat(torch.tensor(train_tensor), dim=0).double()
test_tensor_in = torch.cat(torch.tensor(test_tensor), dim=0).double()

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
