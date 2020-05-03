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

    def accuracy(self,weight_list,df,file):        
        for i in range(0,len(df)):
            feature_vector = list(df.iloc[i,:])
            label =  feature_vector[len(feature_vector)-1]
            feature_vector = feature_vector[:len(feature_vector)-1]            
            feature_vector.insert(0,1)
            feature_vector_arr = np.asarray(feature_vector)
            value_out = self.sigmoid_z1(np.dot(weight_list,feature_vector_arr))
            print('value_out:',value_out,' label:',label)
            file.write('value predicted :' + str(value_out) + " " + 'label: ' + str(label) + "\n")
            if value_out >= 0.5:
                if label == 1.0:
                    performance.cnt += 1
                    performance.tp += 1
                elif label == 0.0:
                    performance.fp += 1
            elif value_out < 0.5:
                if label == 1.0:
                    performance.fn += 1
                elif label == 0.0:
                    performance.tn += 1
                    performance.cnt += 1
        print('cnt:',performance.cnt)                
        return (float(performance.cnt)/len(df))*100
    def f_score(self):
        precision = float(performance.tp)/(float(performance.tp)+float(performance.fp))
        print('precision',precision)
        recall = float(float(performance.tp)/(float(performance.tp)+float(performance.fn)))
        print('recall',recall)
        fscore = 2*(recall*precision)/(recall + precision)
        return fscore