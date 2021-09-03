# AutoWave - Automatic Audio Classification Library
<p align="center"><img src="https://github.com/TechyNilesh/AutoWave/blob/main/logo/autowave_logo.png?raw=true" alt="Brain+Machine"></p>

**AutoWave** is an complete audio automatic classification library with other features plottings,audio agumentaion, data loading etc.

![Generic badge](https://img.shields.io/badge/AutoWave-v1-orange.svg) ![Generic badge](https://img.shields.io/badge/Artificial_Intelligence-Advance-green.svg) ![Generic badge](https://img.shields.io/badge/Python-v3-blue.svg) ![Generic badge](https://img.shields.io/badge/pip-v3-red.svg) [![Downloads](https://static.pepy.tech/personalized-badge/autowave?period=total&units=none&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/autowave)

<h2><img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence9-512.png" alt="Brain+Machine" height="38" width="38"> Creators </h2>

#### [Nilesh Verma](https://nileshverma.com "Nilesh Verma")
#### [Satyajit Pattnaik](https://github.com/pik1989 "Satyajit Pattnaik")
#### [Kalash Jindal](https://github.com/erickeagle "Kalash Jindal")

## Features
- Training the model fastly 
- Augumentation can be done easily on one or multiple file and while training model we can use augumentation
- Plotting of single or multiples files can be done by just running one function
- Multiple models can be trained simultaneously

## Installation

This library is compatible with both *windows* and *Linux system* you can just use **PIP command** to install this library on your system:

```shell
pip install AutoWave
```

## How To Use?

We have provided the **Demo** folder under the *GitHub repository*, you can find the example in both **.py** and **.ipynb**  file. Following are the ideal flow of the code:

### 1. Importing the Important Classes

There are three important classes you need to load **AudioPlayer** - for playing the audio , **audioConversion** - for converting the single audio file, **read_file_properties** - for getting the info of any wave file ,**augumentOneFile** - for augumenting any single file ,**augumentFolder** - for augumenting the whole folder,**plotOneFile** - for plotting one file,**plotMultipleFile** -n for plotting multiple files ,**gen_data_from_folder** - for generating the data for classification,**Auto_Audio_Classification** - for training the model

```python
# Importing the proper classes
from AutoWave.audio_player import AudioPlayer
from AutoWave.audio_conversion import audioConversion
from AutoWave.WaveInfo import read_file_properties
from AutoWave.augumentor import augumentOneFile,augumentFolder
from AutoWave.plotting import plotOneFile,plotMultipleFile
from AutoWave.DataLoad import gen_data_from_folder
from AutoWave.Auto_Audio_Classification import Auto_Audio_Classification
```


### 2. For playing the audio

For playing the audio you can use **AudioPlayer** which takes the path of the file as input

```python
AudioPlayer('Test_Data/car_horn/107090-1-1-0.wav')
```


### 3. For converting the audio format

For converting the format of single file using **audioConversion** which take input the filename, input format and output format. It return the converted file in the same path

```python
audioConversion('test.mp3','mp3','wav')
```
<img src="https://github.com/Autowave/Autowave/blob/main/img/img1.png?raw=true" />

### 4. For Augumenting the audio file

For augumenting the single file **augumentOneFile** which take audio file path, output path, by default it augument one file 10 time by shifing, adding noise, changing pitch and streching. We can Off any one this functionality by making it False.

```python
augumentOneFile('test.wav','augumented_data',aug_times=10,noise=True,shift=True,stretch=True,pitch=True)
```
<img src="https://github.com/Autowave/Autowave/blob/main/img/img2.png?raw=true" />
For folder **augementFolder** which takes a dataframe generated using the **gen_data_from_folder**, output path,  by default it augument one file 10 time by shifing, adding noise, changing pitch and streching.  We can Off any one this functionality by making it False.


### 5. Reading the Info of the Wave file

For Reading the Wave file **read_file_properties** take the path of audio file and returns the filename, number of channels, sample rate and bit depth.


```python
read_file_properties('test.wav')
```


### 6. ForPloting the audio file

For augumenting the single file **plotOneFile** which has different functions for ploting like **time_fre_domain** for plotting in time and frequency domain, **fre_doman** for ploting in frequency domain, **time_domain** for ploting in time doamin, **waveplot** for ploting the waveplot, **spectrogram** for ploting the spectrogram, **spectral_centroid** for ploting the spectral centroid, **spectraal_rolloff** for ploting the spectral rolloff, **spectral_brandwidth** for ploting the spectral brandwidth.


For folder **plotMultipleFile** which takes a dataframe generated using the **gen_data_from_folder**, which has different functions for ploting like **time_fre_domain** for plotting in time and frequency domain, **fre_doman** for ploting in frequency domain, **time_domain** for ploting in time doamin, **waveplot** for ploting the waveplot, **spectrogram** for ploting the spectrogram, **spectral_centroid** for ploting the spectral centroid, **spectraal_rolloff** for ploting the spectral rolloff, **spectral_brandwidth** for ploting the spectral brandwidth.


```python
plotOneFile.time_freq_domain('Test.wav')
plotMultipleFile.time_freq_domain(data)
```
<img src="https://github.com/Autowave/Autowave/blob/main/img/img3.png?raw=true" />

### 7. Loading the data

For loading the data from the folder **gen_data_from_folder** which takes the input the folder containing the different classes of the audio file in different folders in a sinlge folder and it returns the dataframe tha path of the each and every file with there label.

```python
dataset_dir = 'Test_Data/'
data = gen_data_from_folder(dataset_dir,get_dataframe=True,label_folder=True)
```
<img src="https://github.com/Autowave/Autowave/blob/main/img/img4.png?raw=true" />

### 8. For Training the model

For training the model we use **Auto_Audio_Classification** which takes the input the size of the test data, we can augument the data to by making aug_data = True, if want the trainned model then we can make get_prediction_model which will return the best model with higher accuracy, by making return _dataframe = True it will return the results in the dataframe format.


```python
model = Auto_Audio_Classification(test_size=0.2,label_encoding=True,result_dataframe=False,aug_data=True)
model.fit(data)
```

<img src="https://github.com/Autowave/Autowave/blob/main/img/img5.png?raw=true" />
<img src="https://github.com/Autowave/Autowave/blob/main/img/img6.png?raw=true" />
<img src="https://github.com/Autowave/Autowave/blob/main/img/img7.png?raw=true" />



```python
audio_file = 'Test_Data/class_2/test1.wav'
model.predict(audio_file)
```
<img src="https://github.com/Autowave/Autowave/blob/main/img/img8.jpeg?raw=true" />











