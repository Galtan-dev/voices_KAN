import librosa
import librosa.display
import IPython.display as ipd
import os
import numpy as np
import toolbar_kan as tbk

path = '/samples_testing\\healthy\\svdadult0001_healthy_50000.txt'

data = tbk.data_from_txt(path)
sample_rate = 8000

mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=10).T, axis=0)
mfcc = np.reshape(mfcc, [1, 10])
print(mfcc)
