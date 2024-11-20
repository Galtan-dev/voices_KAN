import numpy as np
import os


# Funkce pro načtení zvukových dat z textového souboru
def load_audio_from_txt(file_path):
    return np.loadtxt(file_path)


# Funkce pro přidání šumu
def add_noise(data, noise_factor=0.005):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise


# Funkce pro změnu rychlosti
def change_speed(data, speed_factor=1.2):
    indices = np.round(np.arange(0, len(data), speed_factor)).astype(int)
    indices = indices[indices < len(data)]
    return data[indices]


# Funkce pro časové posunutí
def shift_time(data, shift_max=500):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift)


# Cesta k adresáři s nahrávkami
input_dir = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\healthy"
output_dir = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples_analyzing\\augmented"
os.makedirs(output_dir, exist_ok=True)

# Pro každý soubor v adresáři provést augmentaci
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)

        # Načtení originálního zvukového signálu
        data = load_audio_from_txt(file_path)

        # Vytvoření augmentovaných verzí
        augmented_data = {
            "noise": add_noise(data),
            "speed": change_speed(data),
            "shifted": shift_time(data)
        }

        # Uložení augmentovaných dat
        for aug_type, aug_data in augmented_data.items():
            aug_filename = f"{os.path.splitext(filename)[0]}_{aug_type}.txt"
            aug_file_path = os.path.join(output_dir, aug_filename)
            np.savetxt(aug_file_path, aug_data)
            print(f"Uloženo: {aug_file_path}")
