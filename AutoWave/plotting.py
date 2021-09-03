import pandas as pd
from datetime import datetime
from os import scandir
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import librosa

import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import minmax_scale


class plotOneFile():
    '''
    This class is used to plot for single file 
    '''
    def time_freq_domain(filename): 
        '''
        Plot time and frequency domain
        ''' 
        rate, data = wav.read(filename)
        fft_out = fft(data)
        duration = len(data)/rate
        time = np.arange(0,duration,1/rate) #time vector
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].plot(data, np.abs(fft_out))
        axes[1].plot(time,data)
        axes[0].set_xlabel('Frequency Domain')
        axes[1].set_xlabel('Time Domain')
        fig.suptitle('Frequency and Time Domain'+' '+os.path.basename(filename))
        fig.tight_layout()
        
    def freq_domain(filename):
        '''
        Plot frequency domain
        '''
        rate, data = wav.read(filename)
        fft_out = fft(data)
        plt.figure(figsize=(10, 5))
        plt.plot(data, np.abs(fft_out))
        plt.title('Frequency Domain'+' '+os.path.basename(filename))
    
    def time_domain(filename):
        '''
        Plot in time domain
        '''
        rate, data = wav.read(filename)
        duration = len(data)/rate
        time = np.arange(0,duration,1/rate)
        plt.figure(figsize=(10, 5))
        plt.plot(time,data)
        plt.title('Time Domain'+' '+os.path.basename(filename))
        
    def waveplot(filename):
        '''
        Plot waveplot
        '''
        data , sr = librosa.load(filename)
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(data, sr=sr)
        plt.title('Waveplot')
        
    def spectrogram(filename):
        '''
        Plot Spectrogram
        '''
        data , sr = librosa.load(filename)
        X = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.title('Spectrogram')
        
    def spectral_centroid(filename):
        '''
        Plot Spectral centroid
        '''
        data, sr = librosa.load(filename)
        spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
        
        plt.figure(figsize=(12, 4))
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(data, sr=sr, alpha=0.4)
        plt.plot(t, minmax_scale(spectral_centroids,axis=0),color='r')
        plt.title('Spetral Centroid')
        
    def spectral_rolloff(filename):
        '''
        Plot Spectral rolloff
        '''
        data, sr = librosa.load(filename)
        spectral_rolloff = librosa.feature.spectral_rolloff(data+0.01, sr=sr)[0]
        plt.figure(figsize=(12, 4))
        spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        librosa.display.waveplot(data, sr=sr, alpha=0.4)
        plt.plot(t, minmax_scale(spectral_rolloff,axis=0), color='r')
        plt.title('Spectral Rolloff')
    
    def spectral_brandwidth(filename):
        '''
        Plot Spectral brandwidth
        '''
        data, sr = librosa.load(filename)
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr)[0]
        spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr, p=3)[0]
        spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr, p=4)[0]
        plt.figure(figsize=(15, 9))
        librosa.display.waveplot(data, sr=sr, alpha=0.4)
        spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames)
        plt.plot(t, minmax_scale(spectral_bandwidth_2,axis=0), color='r')
        plt.plot(t, minmax_scale(spectral_bandwidth_3,axis=0), color='g')
        plt.plot(t, minmax_scale(spectral_bandwidth_4,axis=0), color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))
        plt.title('Spectral Brandwidth')

    
class plotMultipleFile():
    '''
    This class is used for plotting of multiple files by default is 10
    '''
                
    def time_freq_domain(dataframe,number_of_plots=10):
        '''
        Plot for time frequency domain
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            rate, data = wav.read(files)
            fft_out = fft(data)
            duration = len(data)/rate
            time = np.arange(0,duration,1/rate) #time vector
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            axes[0].plot(data, np.abs(fft_out))
            axes[1].plot(time,data)
            axes[0].set_xlabel('Frequency Domain')
            axes[1].set_xlabel('Time Domain')
            fig.suptitle('Frequency and Time Domain'+' '+os.path.basename(files))
            fig.tight_layout()
    
    
    def time_domain(dataframe,number_of_plots=10):
        '''
        Plot for time domain
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            rate, data = wav.read(files)
            duration = len(data)/rate
            time = np.arange(0,duration,1/rate)
            plt.figure(figsize=(10, 5))
            plt.plot(time,data)
            plt.title('Time Domain'+' '+os.path.basename(files))
                
    def freq_domain(dataframe,number_of_plots=10):
        '''
        Plot for frequency domain
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            rate, data = wav.read(files)
            fft_out = fft(data)
            plt.figure(figsize=(10, 5))
            plt.plot(data, np.abs(fft_out))
            plt.title('Frequency Domain'+' '+os.path.basename(files))
                
    def waveplot(dataframe,number_of_plots=10):
        '''
        plot waveplot
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            data , sr = librosa.load(files)
            plt.figure(figsize=(14, 5))
            librosa.display.waveplot(data, sr=sr)
            plt.title('Waveplot'+' '+os.path.basename(files))
        
    def spectrogram(dataframe,number_of_plots=10):
        '''
        Plot spectrogram
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            data , sr = librosa.load(files)
            X = librosa.stft(data)
            Xdb = librosa.amplitude_to_db(abs(X))
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar()
            plt.title('Spectrogram'+' '+os.path.basename(files))
        
    def spectral_centroid(dataframe,number_of_plots=10):
        '''
        Plot Spectral Centroid
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            data , sr = librosa.load(files)
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            spectral_centroids.shape
            plt.figure(figsize=(12, 4))
            frames = range(len(spectral_centroids))
            t = librosa.frames_to_time(frames)
            librosa.display.waveplot(data, sr=sr, alpha=0.4)
            plt.plot(t, minmax_scale(spectral_centroids,axis=0),color='r')
            plt.title('Spectral Centroid'+' '+os.path.basename(files))
        
    def spectral_rolloff(dataframe,number_of_plots=10):
        '''
        Plot spectral rolloff
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            data , sr = librosa.load(files)
            spectral_rolloff = librosa.feature.spectral_rolloff(data+0.01, sr=sr)[0]
            plt.figure(figsize=(12, 4))
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            frames = range(len(spectral_centroids))
            t = librosa.frames_to_time(frames)
            librosa.display.waveplot(data, sr=sr, alpha=0.4)
            plt.plot(t, minmax_scale(spectral_rolloff,axis=0), color='r')
            plt.title('Spectral Rolloff'+' '+os.path.basename(files))
    
    def spectral_brandwidth(dataframe,number_of_plots=10):
        '''
        Plot Spectral brandwidth
        '''
        for files in dataframe['File_List'][0:number_of_plots]:
            data , sr = librosa.load(files)
            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr)[0]
            spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr, p=3)[0]
            spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(data+0.01, sr=sr, p=4)[0]
            plt.figure(figsize=(15, 9))
            librosa.display.waveplot(data, sr=sr, alpha=0.4)
            spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]
            frames = range(len(spectral_centroids))
            t = librosa.frames_to_time(frames)
            plt.plot(t, minmax_scale(spectral_bandwidth_2,axis=0), color='r')
            plt.plot(t, minmax_scale(spectral_bandwidth_3,axis=0), color='g')
            plt.plot(t, minmax_scale(spectral_bandwidth_4,axis=0), color='y')
            plt.legend(('p = 2', 'p = 3', 'p = 4'))
            plt.title('Spectral Brandwidth'+' '+os.path.basename(files))

    
    
    
    
    
    
    
    
    
    