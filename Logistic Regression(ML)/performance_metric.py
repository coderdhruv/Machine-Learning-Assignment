from preprocessing import preprocess
import math
import numpy as np

class performance:
    cnt = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    def sigmoid_z1(self,z):
        return 1/(1 + np.exp(-z))

    def accuracy(self,weight_list,df):        
        for i in range(0,len(df)):
            feature_vector = list(df.iloc[i,:])
            label =  feature_vector[len(feature_vector)-1]
            feature_vector = feature_vector[:len(feature_vector)-1]            
            feature_vector.insert(0,1)
            feature_vector_arr = np.asarray(feature_vector)
            value_out = self.sigmoid_z1(np.dot(weight_list,feature_vector_arr))
            print(value_out)
            if value_out >= 0.5:
                if label == 1:
                    performance.cnt += 1
                    performance.tp += 1
                elif label == 0:
                    performance.fp += 1
            elif value_out < 0.5:
                if label == 1:
                    performance.fn += 1
                elif label == 0:
                    performance.tn += 1
                    performance.cnt += 1                
        return (performance.cnt/len(df))*100
    def f_score(self):
        precision = performance.tp/((performance.tp)+(performance.fp))
        recall = performance.tp/((performance.tp)+(performance.fn))
        fscore = 2*(recall*precision)/(recall + precision)
        return fscore