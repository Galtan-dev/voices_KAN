import toolbar_kan as tbk
import torch
from kan import KAN, LBFGS

# generation of dataset
input_dataset = tbk.dataset_generator(50000, 13, 20, 0.8)

# Nastavení výchozího typu pro torch
torch.set_default_dtype(torch.float64)

# KAN model training
model = KAN(width=[33, 20, 10, 2], grid=50, k=3, seed=0)

results = model.fit(
    input_dataset,
    opt="LBFGS",
    steps=200,
    lamb=0.002,
    loss_fn=torch.nn.CrossEntropyLoss())

# fiting function on KAN model
formula = tbk.function_fitting(model)

# calculation of F1 score from given function a data
positive_num = 661
negative_num = 1324
# print('F1 score of train:', tbk.model_resolution(formula, positive_num, negative_num, input_dataset['train_input'], input_dataset['train_label']))
print('F1 score of test:', tbk.model_resolution(formula, positive_num, negative_num, input_dataset['test_input'], input_dataset['test_label']))
