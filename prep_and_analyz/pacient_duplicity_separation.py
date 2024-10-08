import csv
from collections import Counter
from collections import Counter
import os
import shutil

with open('C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\sources\\file_information.csv', mode ='r')as file:
    samples_info = csv.reader(file)
    numbers = []
    for lines in samples_info:
        numbers.append(lines[4])
    count = Counter(numbers)
    only_once = [num for num, freq in count.items() if freq == 1]
    multiple_times = [num for num, freq in count.items() if freq > 1]

    print("Čísla, která se vyskytují pouze jednou:", only_once)
    print("Čísla, která se vyskytují vícekrát:", multiple_times)


# data_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing"
# for item in os.listdir(data_path):