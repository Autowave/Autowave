import sklearn
import xgboost
import lightgbm
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn import preprocessing
import time
from sklearn.metrics import recall_score,precision_score,roc_auc_score,f1_score,accuracy_score
import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
import pprint
from .augumentor import augumentFolder
from .ExtractFeature import Extract_Mfcc,Extract_Spectral_Centroids,Extract_Spectral_Rolloff,Extract_Zero_Crossings,Extract_Spectral_Bandwidth,Extract_Chromagram,Extract_Stft


import warnings
warnings.filterwarnings("ignore")

def label_encode(label_data:list):
    label_encodder = preprocessing.LabelEncoder()
    label_encodder.fit(label_data)
    label_data_encode = label_encodder.transform(label_data)
    return label_data_encode, label_encodder

def select_FE(DataFrame,type:str):
    if type == 'mfcc':
        X,Y = Extract_Mfcc(DataFrame)
        return X,Y
    elif type == 'spectral centroid':
        X,Y = Extract_Spectral_Centroids(DataFrame)
        return X,Y
    elif type == 'spectral rolloff':
        X,Y = Extract_Spectral_Rolloff(DataFrame)
        return X,Y
    elif type == 'zero crossings':
        X,Y = Extract_Zero_Crossings(DataFrame)
        return X,Y
    elif type == 'spectral bandwidth':
        X,Y = Extract_Spectral_Bandwidth(DataFrame)
        return X,Y
    elif type == 'chromagram':
        X,Y = Extract_Chromagram(DataFrame)
        return X,Y
    elif type == 'stft':
        X,Y = Extract_Stft(DataFrame)
        return X,Y
    else:
        print("ERROR: Please select proper feature type.")


CLASSIFIERS = [est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]

removed_classifiers = [
    ("DummyClassifier", sklearn.dummy.DummyClassifier),
    ("ClassifierChain", sklearn.multioutput.ClassifierChain),
    ("ComplementNB", sklearn.naive_bayes.ComplementNB),
    (
        "GradientBoostingClassifier",
        sklearn.ensemble.GradientBoostingClassifier,
    ),
    (
        "GaussianProcessClassifier",
        sklearn.gaussian_process.GaussianProcessClassifier,
    ),
    (
        "HistGradientBoostingClassifier",
        sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier,
    ),
    ("MLPClassifier", sklearn.neural_network.MLPClassifier),
    ("LogisticRegressionCV", sklearn.linear_model.LogisticRegressionCV),
    ("MultiOutputClassifier", sklearn.multioutput.MultiOutputClassifier),
    ("MultinomialNB", sklearn.naive_bayes.MultinomialNB),
    ("OneVsOneClassifier", sklearn.multiclass.OneVsOneClassifier),
    ("OneVsRestClassifier", sklearn.multiclass.OneVsRestClassifier),
    ("OutputCodeClassifier", sklearn.multiclass.OutputCodeClassifier),
    (
        "RadiusNeighborsClassifier",
        sklearn.neighbors.RadiusNeighborsClassifier,
    ),
    ("VotingClassifier", sklearn.ensemble.VotingClassifier),
    ("StackingClassifier",sklearn.ensemble._stacking.StackingClassifier)
]

for i in removed_classifiers:
    CLASSIFIERS.pop(CLASSIFIERS.index(i))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))

CLASSIFIERS = dict(CLASSIFIERS)

class Auto_Audio_Classification:
    def __init__(self,test_size=0.20,get_prediction_model=False,label_encoding=False,result_dataframe=False,prediction=True,aug_data=False,feature='mfcc'):
        self.test_size=test_size
        self.get_prediction_model = get_prediction_model
        self.label_encoding = label_encoding
        self.result_dataframe = result_dataframe
        self.label_encoder =  None
        self.infrence_model = None
        self.prediction = prediction
        self.aug_data = aug_data
        self.feature = feature
        
    def fit(self,dataframe):
        
        self.dataframe = dataframe
        
        if self.aug_data == True:
            self.dataframe = augumentFolder(self.dataframe['File_List'],'augumented_data')
        
        self.X,self.Y = select_FE(self.dataframe,self.feature)          
        
        if self.label_encoding==True:
            self.Y,self.label_encoder=label_encode(self.Y)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle = True)

        dic =  {"Model":[],"Acuracy":[],"Sensitivity":[],"Precision":[],"F-Score":[],"ROC_AUC":[],'Time':[]}
        for name, model_new in tqdm.tqdm(CLASSIFIERS.items()):
            start = time.time()
            model = model_new()
            try:
                model.fit(np.array(X_train), np.array(y_train))
                predict_test = model.predict(np.array(X_test))
            except Exception as exception:
                print("Invalid Classifier(s) : "+name)
                print(f"Error Code: {exception}")
                continue
            Accuracy = accuracy_score(y_test,predict_test,normalize=True)
            Sensitivity = recall_score(y_test,predict_test,average="weighted")
            #ROC_AUC = roc_auc_score(y_test,predict_test, average="weighted",multi_class='ovr')
            try:
                roc_auc = roc_auc_score(y_test,predict_test,multi_class='ovo')
                dic["ROC_AUC"].append(roc_auc)
            except Exception as exception:
                roc_auc = None
                #print("ROC AUC couldn't be calculated for " + name)
                #print(exception)
                dic["ROC_AUC"].append(roc_auc)
            Precision =  precision_score(y_test,predict_test,average="weighted")
            FScore =  f1_score(y_test,predict_test,average="weighted")
            dic["Model"].append(name)
            dic["Acuracy"].append(Accuracy)
            dic["Sensitivity"].append(Sensitivity)
            dic["Precision"].append(Precision)
            dic["F-Score"].append(FScore)
            dic["Time"].append(time.time() - start)
        final_data = pd.DataFrame(dic)
        final_data = final_data.sort_values(by='Acuracy',ascending=False).reset_index(drop=True)
        
        if all(x is None for x in dic["ROC_AUC"]):
            final_data = final_data.sort_values(by='Acuracy',ascending=False).reset_index(drop=True).drop(['ROC_AUC'],axis=1)
        
        pprint.pprint("==================================================================")
        pprint.pprint("Best Model Sorted By Accurcy")
        pprint.pprint("==================================================================")
        pprint.pprint(final_data)

        if self.prediction == True:
            self.infrence_model  = CLASSIFIERS[final_data['Model'][0]]().fit(np.array(self.X), np.array(self.Y))
        if self.get_prediction_model==True and self.result_dataframe==True:
            self.infrence_model  = CLASSIFIERS[final_data['Model'][0]]().fit(np.array(self.X), np.array(self.Y))
            return self.infrence_model,final_data
        if self.get_prediction_model==True:
            self.infrence_model  = CLASSIFIERS[final_data['Model'][0]]().fit(np.array(self.X), np.array(self.Y))
            return self.infrence_model
        if self.result_dataframe==True:
            return final_data
            
    def predict(self,audio_data):
        self.audio_data = pd.DataFrame({"File_List":audio_data,"Label":None},index=[0])
        self.audio_data,_ = select_FE(self.audio_data,self.feature)
        return self.label_encoder.inverse_transform(self.infrence_model.predict(self.audio_data))