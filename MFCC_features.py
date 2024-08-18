import numpy as np
import librosa
import matplotlib.pyplot as plt


def load_audio_from_txt(file_path):
    """Načte zvuková data z textového souboru."""
    audio_data = np.loadtxt(file_path)
    return audio_data


def extract_mfcc_features(audio_data, sample_rate=8000, n_mfcc=13, segment_length=2):
    """Extrakce MFCC ze zvukového signálu."""
    mfcc_features = []
    num_samples_per_segment = segment_length * sample_rate
    num_segments = len(audio_data) // num_samples_per_segment

    for i in range(num_segments):
        start_sample = i * num_samples_per_segment
        end_sample = start_sample + num_samples_per_segment
        segment = audio_data[start_sample:end_sample]

        if len(segment) == num_samples_per_segment:
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_features.append(mfcc_mean)

    return np.array(mfcc_features)


def plot_mfcc(mfcc_features):
    """Vykreslí MFCC."""
    if mfcc_features.size == 0:
        print("No MFCC features extracted.")
        return

    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc_features.T, aspect='auto', origin='lower')
    plt.title('MFCC')
    plt.xlabel('Segment')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar()
    plt.show()


def main(file_path):
    """Hlavní funkce pro načtení souboru, extrakci MFCC a jejich vykreslení."""
    # Načtení zvukových dat z textového souboru
    audio_data = load_audio_from_txt(file_path)

    # Kontrola načtení dat
    if audio_data.size == 0:
        print("Audio data could not be loaded or is empty.")
        return

    print(f"Loaded audio data with shape: {audio_data.shape}")

    # Extrakce MFCC
    mfcc_features = extract_mfcc_features(audio_data)

    # Kontrola extrahovaných MFCC
    if mfcc_features.size == 0:
        print("No MFCC features extracted.")
        return

    print(f"Extracted MFCC features with shape: {mfcc_features.shape}")

    # Vytisknutí MFCC hodnot
    print("MFCC Features (mean values for each segment):")
    print(mfcc_features)

    # Vykreslení MFCC
    plot_mfcc(mfcc_features)


# Cesta k vašemu textovému souboru
file_path = "C:\\Users\\jakub\\PycharmProjects\\KAN_voices\\samples\\unhelathy\\svdadult0160_unhealthy_50000.txt"

# Volání hlavní funkce
main(file_path)
