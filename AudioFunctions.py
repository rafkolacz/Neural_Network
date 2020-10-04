import librosa
import numpy as np
import pandas as pd
import sklearn


# This function extracts features from music file
def audioAnalysis(path):
    y, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0)
    y, sr = librosa.load(path)

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    tempo = librosa.feature.tempogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonal = librosa.feature.tonnetz(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    #bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    data = [np.mean(zero_crossing), np.mean(cent), np.mean(rolloff), np.mean(chroma), np.mean(rms), np.mean(spec_bw),
            np.mean(tempo), np.mean(contrast), np.mean(tonal), np.mean(flatness)]
    for e in mfcc:
        data.append(np.mean(e))
    return data


def labeling(output):
    genres = []
    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for song in output:
        genres.append(labels[np.argmax(song)])
    return genres


# do rownego podzialu, tak zeby na kazdy gatunek przypadalo 80/20 utworow testowych/treningowcyh
def divide(data):
    for i in range(10):
        arr = data.loc[(100*i):(100*i)+99]
        XYtrain, XYtest= sklearn.model_selection.train_test_split(arr, test_size=0.2, random_state=4)
        if i == 0:
            train = pd.DataFrame(XYtrain)
            test = pd.DataFrame(XYtest)
        else:
            test = pd.concat([test, XYtest])
            train = pd.concat([train, XYtrain])
    return test, train
