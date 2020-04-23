import pandas as pd
import numpy as np

class preprocess:
        
    def __init__(self,file):
        self.file = file
    def store_to_dataframe(self):
        self.read_file = pd.read_csv('./data_banknote_authentication.txt',names = ['feature1','feature2','feature3','feature4','class'])
        return self.read_file
    def normalize(self,df):
        max = -9999999
        min = 9999999       
        for i in range(0,len(df['feature1'])):
            for j in range(0,4):
                if df.iloc[i,:][j] > max:
                    max = df.iloc[i,:][j]
                if df.iloc[i,:][j] < min:
                    min = df.iloc[i,:][j]
        feature_vect = ['feature1','feature2','feature3','feature4']
        for i in range(0,len(df['feature1'])):            
            for j in range(0,4):
                df.at[i,feature_vect[j]] = (df.iloc[i,:][j] - min)/(max - min)
        return df

file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
df = prep.store_to_dataframe()
print(prep.normalize(df).head())