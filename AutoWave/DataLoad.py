import pandas as pd
import os
import tqdm

def gen_data_from_folder(folder_name,get_dataframe=False,get_label=True,label_folder=False):
    data = []
    label = []
    if label_folder ==True:
        data_label = os.listdir(folder_name)
        for i in tqdm.tqdm(data_label):
            for j in os.listdir(os.path.join(folder_name,i)):
                data.append(folder_name+"/"+i+"/"+j)
                label.append(i)
        if get_dataframe==True:
            return pd.DataFrame({'File_List':data,'Label':label})
        elif get_label == True:
            return data,label
        else:
            return data
    else:
        for i in tqdm.tqdm(os.listdir(folder_name)):
            data.append(folder_name+i)
        return data
def gen_data_from_csv(csv_file,file_Column_name:str,label_column_name,get_dataframe=True):
    data = pd.read_csv(csv_file)
    if get_dataframe==False:
        return data[file_Column_name].to_list(),data[label_column_name].to_list()
    return pd.DataFrame({'File_List':data[file_Column_name],'Label':data[label_column_name]})