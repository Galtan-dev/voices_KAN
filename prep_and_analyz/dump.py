import os

import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
import opensmile
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from kan import KAN
from statistics import mean
import numpy as np

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Kontrola dostupnosti GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.version.cuda)  # Zjistí verzi CUDA kompatibilní s PyTorch
print(torch.cuda.is_available())  # Mělo by vrátit True, pokud je GPU dostupné
# Funkce pro načtení dat z textového souboru
def data_from_txt(filepath):
    with open(filepath, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

# Konverze dat do tensoru
def data_to_tensor(data, tensors_list):
    tensor = torch.tensor(data, dtype=torch.float32).to(device)
    tensors_list.append(tensor)
    return tensors_list

# Standardizace tensoru
def standardize_train_test(train_tensor, test_tensor):
    scaler = StandardScaler()
    train_np = train_tensor.cpu().numpy()
    test_np = test_tensor.cpu().numpy()

    # Fit pouze na trénovací data a transformovat obě sady
    train_scaled = scaler.fit_transform(train_np)
    test_scaled = scaler.transform(test_np)

    return torch.tensor(train_scaled).to(device), torch.tensor(test_scaled).to(device)

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
def dataset_prep_smile(file_list, data_path, sample_rate, exclude_augmentations=False):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals)
    tensors_list = []
    labels = []
    augmentations = ["_noise", "_shifted", "_speed"]

    for item_id in file_list:
        for condition in ["healthy", "unhealthy"]:
            filename = f"svdadult{str(item_id).zfill(4)}_{condition}_50000.txt"

            # Kontrola, jestli soubor není augmentovaný (pokud je exclude_augmentations nastaveno na True)
            if exclude_augmentations and any(aug in filename for aug in augmentations):
                continue

            filepath = os.path.join(data_path, condition, filename)
            if os.path.exists(filepath):
                data = data_from_txt(filepath)
                features = smile.process_signal(data, sample_rate).values.flatten().tolist()
                tensors_list = data_to_tensor(features, tensors_list)
                label = 1 if condition == "healthy" else 0
                labels.append(label)

    tensors = torch.stack(tensors_list).to(device)
    labels_tensor = torch.tensor(labels).to(device)
    return tensors, labels_tensor

# Kompletace datasetu
def complete_dataset_prep(csv_file, data_path, ksb=100):
    duplicate_ids, train_unique_ids, test_unique_ids = split_data_by_duplicates(csv_file)

    # Připravíme tréninkovou sadu (duplicitní a část neduplicitních záznamů)
    train_tensors, train_labels = dataset_prep_smile(duplicate_ids + train_unique_ids, data_path, sample_rate=50000)

    # Připravíme testovací sadu (pouze zbývající neduplicitní záznamy, bez augmentovaných nahrávek)
    test_tensors, test_labels = dataset_prep_smile(test_unique_ids, data_path, sample_rate=50000,
                                                   exclude_augmentations=True)

    # Standardizace celého datasetu
    train_tensors, test_tensors = standardize_train_test(train_tensors, test_tensors)

    # Feature selection
    X_train = train_tensors.cpu().numpy()  # Přesun na CPU pro numpy operace
    y_train = train_labels.cpu().numpy()
    X_test = test_tensors.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    selector = SelectKBest(score_func=mutual_info_classif, k=ksb)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    train_tensors = torch.tensor(X_train_selected).to(device)
    train_labels = torch.tensor(y_train).to(device)
    test_tensors = torch.tensor(X_test_selected).to(device)
    test_labels = torch.tensor(y_test).to(device)

    # Vytvoření slovníku s daty
    input_dataset = {
        'train_input': train_tensors.float().to(device),
        'train_label': train_labels.long().to(device),
        'test_input': test_tensors.float().to(device),
        'test_label': test_labels.long().to(device)
    }

    return input_dataset

def auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb):
    # Výpočet průměru z posledních 15 F1 skóre
    if len(results["test_uar"]) >= 15:
        avg_last_15_f1 = mean(results["test_uar"][-15:])
    else:
        avg_last_15_f1 = mean(results["test_uar"])  # Použijeme všechny hodnoty, pokud je jich méně než 15

    # Poslední hodnota F1 skóre
    last_f1 = results["test_uar"][-1]

    # Průměr a maximální hodnota UAR
    avg_uar = mean(results["test_uar"])
    max_val = max(results["test_uar"])

    # Cesta k souboru, který chcete vytvořit nebo upravit
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "log_box.txt")

    # Zapisování do souboru (append mode)
    with open(file_path, "a") as file:
        file.write(
            f"{avg_uar},{max_val},{avg_last_15_f1},{last_f1},{kan_arch},{grid},{k},{ksb},{steps},{lamb}\n"
        )

def test_tp():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp

def test_tn():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn

def test_fp():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    return fp

def test_fn():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    return fn

def test_uar():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar

def kan_arch_gen(input_size):
    steps = list(np.linspace(0, 2, 11))
    kan_archs = []

    # Procházení všech kombinací vrstev
    for first in steps:
        first_layer = input_size * 2 - int(first * input_size)
        if first_layer > 0:
            # Přidáme pouze jednu vrstvu s `Softmax` jako poslední
            kan_archs.append([[input_size, 0], [first_layer, 0], [2, 3]])

        for second in steps:
            second_layer = input_size * 2 - int(second * input_size)
            if first_layer >= second_layer > 0:
                # Přidáme architekturu s více vrstvami, kde poslední má `Softmax`
                kan_archs.append([[input_size, 0], [first_layer, 0], [second_layer, 0], [2, 3]])

    return kan_archs

# Přivolání hlavní funkce
# csv_file = "/disk2/seinerj/voices_kan/file_information.csv"
# data_path = "/disk2/seinerj/voices_kan/samples_analyzing"
csv_file = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\sources\\file_information.csv"
# data_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_testing"
data_path = "D:\\Kan_voices\\samples_analyzing"


# steps = 200
# lamb = 0.001
# ratio = 0.8
# grids = [2, 4, 6, 8]
# k_set = [1, 2, 3, 4]
# ksb_set = [3, 6, 9, 12]

steps = 150
lamb = 0.001
ratio = 0.8
grids = [2]
k_set = [4]
ksb_set = [6]
kan_arch = [[[6, 0], [8, 0], [2, 3]]]

for ksb in ksb_set:
    input_dataset = complete_dataset_prep(csv_file, data_path, ksb=ksb)
    input_dataset['train_input'].to(device)
    input_dataset['train_label'].to(device)
    input_dataset['test_input'].to(device)
    input_dataset['test_label'].to(device)
    print(f"Dataset with ksb:{ksb} prepared")
    # kan_arch = kan_arch_gen(ksb)      # zapnout při více hodnotách
    for k in k_set:
        for arch in kan_arch:
            for grid in grids:
                print(f"Experiment: {arch},{grid},{k},{ksb},{steps},{lamb}\n")
                model = KAN(width=arch, grid=grid, k=k, device=device, seed=seed, auto_save=True, save_act=True)
                results = model.fit(
                    input_dataset,
                    opt="LBFGS",
                    steps=steps,
                    lamb=lamb,
                    metrics=(test_fn, test_fp, test_tn, test_tp, test_uar),
                    loss_fn=torch.nn.CrossEntropyLoss()
                    )
                # model.prune(edge_th=0.01)
                # model.plot(beta=10)
                # plt.show()
                auto_res_log(results, arch, grid, k, ksb, steps, lamb)
