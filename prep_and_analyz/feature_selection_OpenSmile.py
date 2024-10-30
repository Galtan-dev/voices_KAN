import opensmile
import toolbar_kan as tbk
from sklearn.decomposition import PCA
import os
import toolbar_kan as tbk
import torch
from kan import KAN, LBFGS
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from statistics import mean

def test_tp():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
    labels = input_dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp

def test_tn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
    labels = input_dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn

def test_fp():
    """
    Specificity for the test. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
    labels = input_dataset["test_label"]
    # Calculate TP, TN, FP, FN

    fp = ((predictions == 1) & (labels == 0)).sum().float()

    return fp

def test_fn():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
    labels = input_dataset["test_label"]
    # Calculate TP, TN, FP, FN
    fn = ((predictions == 0) & (labels == 1)).sum().float()

    # Calculate recall
    return fn

def test_uar():
    """
    Recall for the test set. That is how the PyKAN needs the metric functions.
    """
    predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
    labels = input_dataset["test_label"]
    # Calculate TP, TN, FP, FN
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()

    # Calculate recall
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar

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

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

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

ksb = 15

# PCA attempt
# pca = PCA(n_components=ksb)
# X_reduced = pca.fit_transform(X)
# all_tensors = X_reduced
# all_labels = torch.tensor(y)


# # SelectBest attempt
selector = SelectKBest(score_func=mutual_info_classif, k=ksb)
X_selected = selector.fit_transform(X, y)
all_tensors = X_selected
all_labels = torch.tensor(y)

# Rozdělení na tréninkovou a testovací sadu (80% trénink, 20% test)
split_index = int(ratio * all_tensors.shape[0])
train_tensor_in = all_tensors[:split_index]
test_tensor_in = all_tensors[split_index:]
train_tensor_target = all_labels[:split_index]
test_tensor_target = all_labels[split_index:]

ones_count = torch.sum(test_tensor_target == 1).item()
zeros_count = torch.sum(test_tensor_target == 0).item()
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

width = [15, 12, 8, 6, 4, 2]
grid = 4
k = 2
steps = 200
lamb = 0.001

for i in range(1, 5):
    # KAN model training
    grid = i
    model = KAN(width=width, grid=grid, k=k, seed=0, auto_save=False, save_act=True)

    results = model.fit(
        input_dataset,
        opt="LBFGS",
        steps=steps,
        lamb=lamb,
        metrics=(test_fn, test_fp, test_tn, test_tp, test_uar),
        loss_fn=torch.nn.CrossEntropyLoss())

    res = (mean(results["test_uar"]))
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Cesta k souboru, který chcete vytvořit nebo upravit
    file_path = os.path.join(current_dir, "log_box.txt")

    # Zapisování do souboru (append mode)
    with open(file_path, "a") as file:  # "a" znamená append (přidat na konec souboru)
        file.write(f"F1:{res},width:{width},grid:{grid},k:{k},ksb:{ksb},steps:{steps},lamb:{lamb}\n")
with open(file_path, "a") as file:  # "a" znamená append (přidat na konec souboru)
    file.write("\n")
# fiting function on KAN model
# formula = tbk.function_fitting(model)

# calculation of F1 score from given function a data
# positive_num = 661
# negative_num = 1324
# positive_num = ones_count
# negative_num = zeros_count
# # print('F1 score of train:', tbk.model_resolution(formula, positive_num, negative_num, input_dataset['train_input'], input_dataset['train_label']))
# print('F1 score of test:', tbk.model_resolution(formula, positive_num, negative_num, input_dataset['test_input'], input_dataset['test_label']))