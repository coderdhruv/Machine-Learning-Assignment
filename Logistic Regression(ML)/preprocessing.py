import pandas as pd
import numpy as np

class preprocess:        
    def __init__(self,file):
        self.file = file
    def store_to_dataframe(self):
        self.read_file = pd.read_csv(self.file.name,names = ['feature1','feature2','feature3','feature4','class'])
        return self.read_file
    def normalize(self,df):
        max = -9999999
        min = 9999999       
        for i in range(0,len(df['feature1'])):
            for j in range(0,4):
                if float(df.iloc[i,:][j]) > max:
                    max = float(df.iloc[i,:][j])
                if float(df.iloc[i,:][j]) < min:
                    min = float(df.iloc[i,:][j])
        feature_vect = ['feature1','feature2','feature3','feature4']
        for i in range(0,len(df['feature1'])):            
            for j in range(0,4):
                df.at[i,feature_vect[j]] = float((df.iloc[i,:][j] - min)/(max - min))
        return df
    def test_train_data_set(self,df,test_ratio):
        shuffled_indices=np.random.permutation(len(df))
        test_set_size=int(test_ratio*len(df))
        test_indices=shuffled_indices[:test_set_size]
        train_indices=shuffled_indices[test_set_size:]
        indices_train = []
        indices_test = []
        for i in range(0,len(train_indices)):
            indices_train.append(i)
        for i in range(0,len(test_indices)):
            indices_test.append(i)
        return df.iloc[train_indices].set_index([pd.Index(indices_train)]),df.iloc[test_indices].set_index([pd.Index(indices_test)])
        


file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
df = prep.store_to_dataframe()
df = prep.normalize(df)
df1 , df2 = prep.test_train_data_set(df,0.2)
print(df1['feature1'][0])
print(df1.head())
print(df2.head())