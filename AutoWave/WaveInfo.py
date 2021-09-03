import struct
import pandas as pd
import tqdm

def read_file_properties(filename):

        wave_file = open(filename,"rb")
        
        riff = wave_file.read(12)
        fmt = wave_file.read(36)
        
        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (filename.split("/")[-1],num_channels, sample_rate, bit_depth)
def read_file_properties_bulk(file_list):
    audiodata = []
    for file in tqdm.tqdm(file_list):
        audiodata.append(read_file_properties(file))
    audiodf = pd.DataFrame(audiodata, columns=['file_name','num_channels','sample_rate','bit_depth'])
    return audiodf