import random
import soundfile as sf
import librosa
import numpy as np
import glob, os
import tqdm
import warnings
warnings.filterwarnings("ignore")

from .DataLoad import gen_data_from_folder

def augumentOneFile(audioname,output_path,aug_times=10,noise=True,shift=True,stretch=True,pitch=True):
    '''
    This function is used to augument the single audio file.
    audioname: path of the input file
    output_path: path of the output folder
    aug_times: total number of files required for each wav file in folder
    noise: if you want to add noise then True otherwise False
    shift: if you want to add shift then True otherwise False
    stretch: if you want to add strech then True otherwise False
    pitch: if you want to add pitch then True otherwise False
    '''
    print("Converting"+" "+audioname)
    formats_to_convert = ['.wav']  
    if audioname.endswith(tuple(formats_to_convert)):
        data,sr = librosa.core.load(audioname)
        for i in range(1,aug_times+1):
            #noise
            if noise==True and random.randint(0,5)>=3:
                wn = np.random.randn(len(data))
                data = data + 0.005*wn

            # Shifting the sound
            if shift==True and random.randint(0,5)>=3:
                data = np.roll(data, random.randint(100,200))

            #stretch
            if stretch==True and random.randint(0,5)>=3:
                data =librosa.effects.time_stretch(data, 1.2)

            #pitch
            if pitch==True and random.randint(0,5)>=3:
                data = librosa.effects.pitch_shift(data, sr, n_steps=random.randint(1,10),bins_per_octave=random.randint(10,200))

            sf.write(os.path.join(output_path,('{}'+audioname).format(i)), data, sr)
        print("Augmentation Completed")
    else:
        print('Please use .wav file')


    
def augumentFolder(data_column,output_path,aug_times=10,noise=True,shift=True,stretch=True,pitch=True):
    '''
    This function is used to augument the audio files present in any folder, means total number of files required for each wav file in folder
    input_path: path of the input folder
    output_path: path of the output folder
    aug_times: total number of files required for each wav file in folder
    noise: if you want to add noise then True otherwise False
    shift: if you want to add shift then True otherwise False
    stretch: if you want to add strech then True otherwise False
    pitch: if you want to add pitch then True otherwise False
    '''
    x=data_column.to_list()

    for audioname in tqdm.tqdm(x):
        #print(audioname)
        formats_to_convert = ['.wav']  
        if audioname.endswith(tuple(formats_to_convert)):
            data,sr = librosa.core.load(audioname)
            for i in range(1,aug_times+1):
                #noise
                if noise==True and random.randint(0,5)>=3:
                    wn = np.random.randn(len(data))
                    data = data + 0.005*wn

                # Shifting the sound
                if shift==True and random.randint(0,5)>=3:
                    data = np.roll(data, random.randint(100,200))

                #stretch
                if stretch==True and random.randint(0,5)>=3:
                    data =librosa.effects.time_stretch(data, 1.2)

                #pitch
                if pitch==True and random.randint(0,5)>=3:
                    data = librosa.effects.pitch_shift(data, sr, n_steps=random.randint(1,10),bins_per_octave=random.randint(10,200))
                
                directory=audioname.split('/')[-2]
                #print(directory)
                audio_name=audioname.split('/')[-1]
                #print(audio_name)

                if output_path not in os.listdir():
                    os.mkdir(output_path)

                if directory not in os.listdir(output_path):
                    os.mkdir(os.path.join(output_path,directory))

                sf.write(os.path.join(os.path.join(output_path,directory),('{}'+audio_name).format(i)), data, sr)
            
        else:
            print('Please use .wav file')
    print("Augmentation Completed")
    return gen_data_from_folder(output_path,get_dataframe=True,label_folder=True)
 
        
    
