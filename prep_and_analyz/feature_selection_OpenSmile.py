import opensmile
import toolbar_kan as tbk
from sklearn.decomposition import PCA
import os
import toolbar_kan as tbk
import torch
from kan import KAN, LBFGS
from sklearn.preprocessing import StandardScaler



smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def dataset_prep_smile(data_path, sample_rate):
    "Tato funkce vezme cestu k adresáři, kde jsou data uložena, a vytvoří chroma vlastnosti pro každou ukázku."
    tensors_list = []
    labels = []  # Seznam pro uchování štítků
    for item in os.listdir(data_path):
        data = tbk.data_from_txt(data_path + "\\" + item)
        feaures = smile.process_signal(data, sample_rate)
        feaures = feaures.values.flatten().tolist()
        tensors_list = tbk.data_to_tensor(feaures, tensors_list)
        # Předpokládáme, že název souboru určuje štítek
        label = 0 if 'unhealthy' in item else 1  # Zde bys měl upravit podle skutečného názvosloví
        labels.append(label)

    tensors_list = torch.stack(tensors_list)
    labels_tensor = torch.tensor(labels)

    return tensors_list, labels_tensor

def standardize_tensor(tensor):
    " Function take tensor and standardize him"
    scaler = StandardScaler()

    # Standardizace
    tensor_scaled_np = scaler.fit_transform(tensor)
    # Převod zpět na PyTorch tensor
    return torch.from_numpy(tensor_scaled_np)

ratio = 0.8
path = ["C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\healthy",
        "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\unhealthy"]

# Inicializace prázdných seznamů pro tréninkové a testovací tensory
all_tensors = []
all_labels = []

# Vytvoření datasetu pro zdravé a nezdravé vzorky
for item in path:
    tensors, labels = dataset_prep_smile(item, 50000)
    all_tensors.append(tensors)
    all_labels.append(labels)


# Slučování tensorů a štítků
all_tensors = torch.cat(all_tensors, dim=0).double()
all_labels = torch.cat(all_labels, dim=0).long()

# Zamíchání vzorků v datasetu, abychom měli smíšené pozitivní a negativní vzorky
indices = torch.randperm(all_tensors.shape[0])
all_tensors = all_tensors[indices]
all_labels = all_labels[indices]

all_tensors = tbk.standardize_tensor(all_tensors)
X = all_tensors.numpy()
y = all_labels.numpy()

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)

all_tensors = X_reduced
all_labels = y

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

print("complete")

# Nastavení výchozího typu pro torch
torch.set_default_dtype(torch.float64)

# KAN model training
model = KAN(width=[5, 4, 3, 2], grid=50, k=3, seed=0)

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