import numpy as np
import librosa
import matplotlib.pyplot as plt

# Funkce pro načtení zvukového signálu z textového souboru
def load_audio_from_txt(file_path):
    # Načtení číselných hodnot ze souboru
    audio_data = np.loadtxt(file_path)
    return audio_data

# Funkce pro extrakci Chroma features
def extract_chroma_features(audio_data, sample_rate=50000):
    # Výpočet chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_chroma=12)
    return chroma

# Cesta k textovému souboru
file_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\unhelathy\\svdadult0160_unhealthy_50000.txt"
# Načtení zvukového signálu
audio_data = load_audio_from_txt(file_path)

# Extrakce Chroma features
sample_rate = 50000  # Předpokládáme standardní vzorkovací frekvenci
chroma_features = extract_chroma_features(audio_data, sample_rate)

# Průměr přes časové úseky (můžete použít podle potřeby)
chroma_mean = np.mean(chroma_features, axis=1)

# Zobrazení Chroma features
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_features, x_axis='time', y_axis='chroma', sr=sample_rate, cmap='coolwarm')
plt.colorbar()
plt.title('Chroma Features')
plt.tight_layout()
plt.show()

print(chroma_mean)
print(type(chroma_mean))
print(chroma_mean.tolist())

# organization test
print(len(audio_data))