import os
import torch
import pandas as pd
import random
import opensmile
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from kan import KAN
from statistics import mean
import numpy as np

# Funkce pro načtení dat z textového souboru
def data_from_txt(filepath):
    with open(filepath, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

# Konverze dat do tensoru
def data_to_tensor(data, tensors_list):
    tensor = torch.tensor(data, dtype=torch.float32)
    tensors_list.append(tensor)
    return tensors_list

# Standardizace tensoru
def standardize_tensor(tensor):
    scaler = StandardScaler()
    tensor_np = tensor.numpy()
    tensor_scaled_np = scaler.fit_transform(tensor_np)
    return torch.from_numpy(tensor_scaled_np)

# Výběr duplicity a vytvoření dvou seznamů
def split_data_by_duplicates(csv_file):
    data = pd.read_csv(csv_file, header=None)
    duplicates = data[data.duplicated(subset=[4], keep=False)]
    unique = data.drop(duplicates.index)

    duplicate_list = duplicates[0].tolist()
    unique_list = unique[0].tolist()

    num_duplicates = len(duplicates[4])
    num_to_select = int(len(data) * 0.8 - num_duplicates)

    # Vybereme náhodných 80% z unikátních ID pro trénovací sadu
    selected_unique_ids = random.sample(unique[4].unique().tolist(), num_to_select)

    # Rozdělíme unikátní ID na trénovací a testovací
    train_unique_ids = selected_unique_ids
    test_unique_ids = [id for id in unique[4].unique().tolist() if id not in selected_unique_ids]

    return duplicates[4].unique().tolist(), train_unique_ids, test_unique_ids

# Připrava datasetu pomocí opensmile
def dataset_prep_smile(file_list, data_path, sample_rate):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    tensors_list = []
    labels = []

    for item_id in file_list:
        for condition in ["healthy", "unhealthy"]:
            filename = f"svdadult{str(item_id).zfill(4)}_{condition}_50000.txt"
            filepath = os.path.join(data_path, condition, filename)
            if os.path.exists(filepath):
                data = data_from_txt(filepath)
                features = smile.process_signal(data, sample_rate).values.flatten().tolist()
                tensors_list = data_to_tensor(features, tensors_list)
                label = 1 if condition == "healthy" else 0
                labels.append(label)

    tensors = torch.stack(tensors_list)
    labels_tensor = torch.tensor(labels)
    return tensors, labels_tensor

# Kompletace datasetu
def complete_dataset_prep(csv_file, data_path, ksb=100):
    duplicate_ids, train_unique_ids, test_unique_ids = split_data_by_duplicates(csv_file)

    # Připravíme tréninkovou sadu (duplicitní a část neduplicitních záznamů)
    train_tensors, train_labels = dataset_prep_smile(duplicate_ids + train_unique_ids, data_path, sample_rate=50000)

    # Připravíme testovací sadu (pouze zbývající neduplicitní záznamy)
    test_tensors, test_labels = dataset_prep_smile(test_unique_ids, data_path, sample_rate=50000)

    # Standardizace celého datasetu
    train_tensors = standardize_tensor(train_tensors)
    test_tensors = standardize_tensor(test_tensors)

    # Feature selection
    X_train = train_tensors.numpy()
    y_train = train_labels.numpy()
    X_test = test_tensors.numpy()
    y_test = test_labels.numpy()

    selector = SelectKBest(score_func=mutual_info_classif, k=ksb)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    train_tensors = torch.tensor(X_train_selected)
    train_labels = torch.tensor(y_train)
    test_tensors = torch.tensor(X_test_selected)
    test_labels = torch.tensor(y_test)

    # Vytvoření slovníku s daty
    input_dataset = {
        'train_input': train_tensors.double(),
        'train_label': train_labels.long(),
        'test_input': test_tensors.double(),
        'test_label': test_labels.long()
    }

    return input_dataset

def auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb):
    res = (mean(results["test_uar"]))
    max_val = (max(results["test_uar"]))
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Cesta k souboru, který chcete vytvořit nebo upravit
    file_path = os.path.join(current_dir, "log_box.txt")

    # Zapisování do souboru (append mode)
    with open(file_path, "a") as file:
        file.write(
            f"{res},{max_val},{kan_arch},{grid},{k},{ksb},{steps},{lamb}\n")

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

def kan_arch_gen(input_size):
    "Kan architecture generator for multiple analyzation"
    # define KAN architecture
    steps = list(np.linspace(0, 2, 11))
    kan_archs = []
    for first in steps:
        first_layer = input_size * 2 - int(first * input_size)
        if first_layer > 0:
            kan_archs.append([input_size, first_layer, 2])
        for second in steps:
            second_layer = input_size * 2 - int(second * input_size)
            if first_layer >= second_layer > 0:
                kan_archs.append([input_size, first_layer, second_layer, 2])

    return kan_archs

# Přivolání hlavní funkce
csv_file = "/disk2/seinerj/file_information.csv"
data_path = "/disk2/seinerj/samples_analyzing"
# csv_file = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\sources\\file_information.csv"
# data_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing"

steps = 400
lamb = 0.001
ratio = 0.8
grids = [1, 2, 3, 4, 5, 6, 7, 8]
k_set = [1, 2, 3]
ksb_set = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# steps = 200
# lamb = 0.001
# ratio = 0.8
# input_size = 5
# grids = [1, 2]
# k_set = [1, 2]
# ksb_set = [3, 4]
# # kan_arch = tbk.kan_arch_gen(input_size)
# kan_arch = [[2, 4, 2], []
torch.set_default_dtype(torch.float64)

for ksb in ksb_set:
    input_dataset = complete_dataset_prep(csv_file, data_path, ksb=ksb)
    print(f"Dataset with ksb:{ksb} prepared")
    kan_arch = kan_arch_gen(ksb)
    for k in k_set:
        for arch in kan_arch:
            for grid in grids:
                print(f"Experiment: {arch},{grid},{k},{ksb},{steps},{lamb}\n")
                model = KAN(width=arch, grid=grid, k=k, seed=0, auto_save=False, save_act=True)
                print("model created")
                results = model.fit(
                    input_dataset,
                    opt="LBFGS",
                    steps=steps,
                    lamb=lamb,
                    metrics=(test_fn, test_fp, test_tn, test_tp, test_uar),
                    loss_fn=torch.nn.CrossEntropyLoss())
                auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb)
