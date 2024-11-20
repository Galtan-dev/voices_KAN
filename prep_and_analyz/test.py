import toolbar_kan as tbk
import torch
from kan import KAN
import os
from statistics import mean

# def auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb):
#     res = (mean(results["test_uar"]))
#     max_val = (max(results["test_uar"]))
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # Cesta k souboru, který chcete vytvořit nebo upravit
#     file_path = os.path.join(current_dir, "log_box.txt")
#
#     # Zapisování do souboru (append mode)
#     with open(file_path, "a") as file:
#         file.write(
#             f"{res},{max_val},{kan_arch},{grid},{k},{ksb},{steps},{lamb}\n")
#
# def test_tp():
#     """
#     Recall for the test set. That is how the PyKAN needs the metric functions.
#     """
#     predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
#     labels = input_dataset["test_label"]
#     # Calculate TP, TN, FP, FN
#     tp = ((predictions == 1) & (labels == 1)).sum().float()
#     return tp
#
# def test_tn():
#     """
#     Recall for the test set. That is how the PyKAN needs the metric functions.
#     """
#     predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
#     labels = input_dataset["test_label"]
#     # Calculate TP, TN, FP, FN
#     tn = ((predictions == 0) & (labels == 0)).sum().float()
#     return tn
#
# def test_fp():
#     """
#     Specificity for the test. That is how the PyKAN needs the metric functions.
#     """
#     predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
#     labels = input_dataset["test_label"]
#     # Calculate TP, TN, FP, FN
#
#     fp = ((predictions == 1) & (labels == 0)).sum().float()
#
#     return fp
#
# def test_fn():
#     """
#     Recall for the test set. That is how the PyKAN needs the metric functions.
#     """
#     predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
#     labels = input_dataset["test_label"]
#     # Calculate TP, TN, FP, FN
#     fn = ((predictions == 0) & (labels == 1)).sum().float()
#
#     # Calculate recall
#     return fn
#
# def test_uar():
#     """
#     Recall for the test set. That is how the PyKAN needs the metric functions.
#     """
#     predictions = torch.argmax(model(input_dataset["test_input"]), dim=1)
#     labels = input_dataset["test_label"]
#     # Calculate TP, TN, FP, FN
#     tn = ((predictions == 0) & (labels == 0)).sum().float()
#     tp = ((predictions == 1) & (labels == 1)).sum().float()
#     fn = ((predictions == 0) & (labels == 1)).sum().float()
#     fp = ((predictions == 1) & (labels == 0)).sum().float()
#
#     # Calculate recall
#     recall = tp / (tp + fn)
#     specificity = tn / (tn + fp)
#     uar = 0.5 * (recall + specificity)
#     return uar
#
# steps = 400
# lamb = 0.001
# ratio = 0.8
# input_size = 10
# grids = [1, 2, 3, 4, 5, 6, 7, 8]
# k_set = [1, 2, 3]
# ksb_set = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# kan_arch = tbk.kan_arch_gen(input_size)
#
# # steps = 200
# # lamb = 0.001
# # ratio = 0.8
# # input_size = 5
# # grids = [1, 2]
# # k_set = [1, 2]
# # ksb_set = [3, 4]
# # # kan_arch = tbk.kan_arch_gen(input_size)
# # kan_arch = [[2, 4, 2]]
# # torch.set_default_dtype(torch.float64)
#
# for ksb in ksb_set:
#     input_dataset = tbk.complete_dataset_prep(ratio, ksb)
#     for k in k_set:
#         for arch in kan_arch:
#             for grid in grids:
#                 model = KAN(width=arch, grid=grid, k=k, seed=0, auto_save=False, save_act=True)
#                 results = model.fit(
#                     input_dataset,
#                     opt="LBFGS",
#                     steps=steps,
#                     lamb=lamb,
#                     metrics=(test_fn, test_fp, test_tn, test_tp, test_uar),
#                     loss_fn=torch.nn.CrossEntropyLoss())
#                 auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb)

# import numpy as np
#
# def kan_arch_gen(input_size):
#     steps = list(np.linspace(0, 2, 11))
#     kan_archs = []
#     for first in steps:
#         first_layer = input_size * 2 - int(first * input_size)
#         if first_layer > 0:
#             kan_archs.append([input_size, first_layer, 2])
#         for second in steps:
#             second_layer = input_size * 2 - int(second * input_size)
#             if first_layer >= second_layer > 0:
#                 kan_archs.append([input_size, first_layer, second_layer, 2])
#
#     return kan_archs
#
# # Počet validních architektur pro každý ksb
# ksb_set = [3, 6, 9, 12]
# arch_counts = {}
#
# for ksb in ksb_set:
#     archs = kan_arch_gen(ksb)
#     arch_counts[ksb] = len(archs)
#
# print(arch_counts)

kan_arch = [[[6, 0], [4, 0], [2, 0]]]

for arch in kan_arch:
    print(arch)