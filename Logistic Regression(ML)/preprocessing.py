import pandas as pd
import numpy as np

class preprocess:
    def __init__(self,file):
        self.file = file
    def store_to_dataframe(self):
        read_file = pd.read_csv('./data_banknote_authentication.txt',names = ['feature1','feature2','feature3','feature4','class'])
        print(read_file['feature1'])
        print(read_file.iloc[0,:][0])
        return read_file        
        

file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
print(prep.store_to_dataframe())