from AutoWave.DataLoad import gen_data_from_folder
from AutoWave.Auto_Audio_Classification import Auto_Audio_Classification,select_FE

dataset_dir = 'Your Audio Folder'
data = gen_data_from_folder(dataset_dir,get_dataframe=True,label_folder=True)

model = Auto_Audio_Classification(test_size=0.2,label_encoding=True,result_dataframe=False,aug_data=False)

model.fit(data)

audio_file = 'Your_audio_file_path'
model.predict(audio_file)