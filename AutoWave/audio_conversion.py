from pydub import AudioSegment
import os
import glob

def audioConversion(audioname,input_format,output_format):
    '''
    This function is used to convert the format of single audio file into another.
    '''
    formats_to_convert = ['.'+input_format]  
    if audioname.endswith(tuple(formats_to_convert)):
        (path, file_extension) = os.path.splitext(audioname)
        file_extension_final = file_extension.replace('.', '')
        try:
            track = AudioSegment.from_file(audioname, file_extension_final) 
            wav_filename = audioname.replace(file_extension_final, output_format)
            print('CONVERTING: ' + str(audioname)) 
            file_handle = track.export(wav_filename, format=output_format)

            audioname=wav_filename
            print('CONVERTED: ' + str(audioname)) 
        except:
            print("ERROR CONVERTING " + str(audioname))

def audioConversionFolder(audioname,input_path,input_format,output_format):
    '''
    This function is used to convert the format of  audio file into another in a folder.
    '''
    os.chdir( input_path)
    for file in glob.glob("*.wav"):
        print(file)
        audioConversion(audioname,input_format,output_format)
    print("Augmentation Completed")
        