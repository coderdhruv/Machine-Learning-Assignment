from preprocessing import preprocess
import math
import numpy as np


def sigmoid_z(z):
    return 1/(1 + np.exp(-z))


def calc_likehihood_est(weight_list_1,pred_out_vect_arr,df):
    sum = 0
    for i in range(0,len(df['feature1'])):
        feature_vector = list(df.iloc[i,:])
        feature_vector.insert(0,1)
        feature_vector = feature_vector[:len(feature_vector)-1]
        feature_vector_arr = np.asarray(feature_vector)
        like_est = -((df['class'][i]*np.log(sigmoid_z(pred_out_vect_arr[i]))) + (1-df['class'][i])*(np.log(1 - sigmoid_z(pred_out_vect_arr[i]))))
        sum = sum + like_est
    return sum    


def logistic_regression(df,alpha):
    weight_list = []
    for i in range(0,5):
        weight_list.append(1)
    weight_list_1 = np.asarray(weight_list)
    pred_out_vect = []
    for i in range(0,len(df['feature1'])):
        feature_vector = list(df.iloc[i,:])
        feature_vector = feature_vector[:len(feature_vector)-1]
        feature_vector.insert(0,1)
        feature_vector_arr = np.asarray(feature_vector)
        pred_out_vect.append(sigmoid_z(np.dot(feature_vector_arr,weight_list_1)))
    pred_out_vect_arr = np.asarray(pred_out_vect)
    orignal_cost = calc_likehihood_est(weight_list_1,pred_out_vect_arr,df)
    for i in range(0,5):
        for j in range(0,len(df['feature1'])):
            weight_list_1[i] = weight_list_1[i] - (alpha)*(pred_out_vect_arr[j] - df['class'][j])*(df.iloc[j,:][i])
    new_cost = calc_likehihood_est(weight_list_1,pred_out_vect_arr,df)
    return orignal_cost,new_cost,weight_list_1


file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
df = prep.store_to_dataframe()
orignal_cost,new_cost,weight_list_1 = logistic_regression(df,0.2)
final_weight_list = []
while orignal_cost - new_cost > 0.001:
    orignal_cost,new_cost,weight_list_1 = logistic_regression(df,0.2)
final_weight_list = weight_list_1.tolist()
test_list = [1,-3.8203,-13.0551,16.9583,-2.3052]
print(sigmoid_z(np.dot(final_weight_list,test_list)))
