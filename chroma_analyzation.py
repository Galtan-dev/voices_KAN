import toolbar_kan as tk
import torch
import os

tensors_list = []
data_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\healthy"

for item in os.listdir(data_path):
    print(data_path + "\\" + item)
    data = tk.data_from_txt(data_path + "\\" + item)
    chroma_mean, chroma_features = tk.chroma(data, 50000, 4)
    tensors_list = tk.data_to_tensor(chroma_mean, tensors_list)

tensors_list = torch.stack(tensors_list)

print(tensors_list)
