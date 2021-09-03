import librosa
from librosa.core.spectrum import db_to_amplitude
import numpy as np
import sklearn
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings("ignore")
    
#helper Function
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def Extract_Mfcc(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        mfccs = librosa.feature.mfcc(x, sr=sr,n_mfcc=40)
        features.append(normalize(np.mean(mfccs.T, axis = 0)))
    return features,DataFrame['Label'].to_list()


def Extract_Spectral_Centroids(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
        features.append(spectral_centroids)
    features_New = np.array(list(itertools.zip_longest(*features, fillvalue=0))).T
    return features_New,DataFrame['Label'].to_list()

def Extract_Spectral_Rolloff(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
        features.append(spectral_rolloff)
    features_New = np.array(list(itertools.zip_longest(*features, fillvalue=0))).T
    return features_New,DataFrame['Label'].to_list()

# helper Function
def zero_crossings_helper(val):
    if val==True:
        return 1
    return 0

def Extract_Zero_Crossings(DataFrame):
    features = []
    n0 = 9000
    n1 = 9100
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
        features.append(np.array([zero_crossings_helper(i) for i in zero_crossings]))
    features_New = np.array(list(itertools.zip_longest(*features, fillvalue=0))).T
    return features_New,DataFrame['Label'].to_list()

def Extract_Spectral_Bandwidth(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
        features.append(spectral_bandwidth)
    features_New = np.array(list(itertools.zip_longest(*features, fillvalue=0))).T
    return features_New,DataFrame['Label'].to_list()

def Extract_Chromagram(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        chromagram = librosa.feature.chroma_stft(x, sr=sr)
        features.append(normalize(np.mean(chromagram.T, axis = 0)))
    return features,DataFrame['Label'].to_list()

def Extract_Stft(DataFrame):
    features = []
    for audio_data in tqdm(DataFrame['File_List'].to_list()):
        x , sr = librosa.load(audio_data)
        stft = np.abs(librosa.stft(x))
        features.append(normalize(np.mean(stft.T, axis = 0)))
    return features,DataFrame['Label'].to_list()


