from preprocessing import preprocess
import math
import numpy as np


def sigmoid_z(z):
    return 1/(1 + np.exp(-z))


def calc_likehihood_est(weight_list_1,df):
    sum = 0
    epsilon = 1e-4
    #print(weight_list_1,"inside calc_likelihood_est")
    for i in range(0,len(df['feature1'])):
        feature_vector = list(df.iloc[i,:])
        label = feature_vector[len(feature_vector)-1]
        feature_vector.insert(0,1)
        feature_vector = feature_vector[:len(feature_vector)-1]
        feature_vector_arr = np.asarray(feature_vector)
        #like_est = -((df['class'][i]*(np.log(sigmoid_z(np.dot(feature_vector_arr,weight_list_1))))) + (1-df['class'][i])*(np.log(1 - sigmoid_z(np.dot(feature_vector_arr,weight_list_1)))))
        #print(np.dot(weight_list_1,feature_vector_arr),"inside calc")
        #value = sigmoid_z(np.dot(weight_list_1,feature_vector_arr))
        #print(value,"inside calc")
        z = sigmoid_z(np.dot(weight_list_1,feature_vector_arr))
        a = label*(np.log(z + epsilon))
        b = (1 - label)*(np.log(1 - z + epsilon))
        like_est = -a + -b
        sum = sum + like_est
    return sum    


def logistic_regression(df,alpha,weight_list):
    weight_list_1 = np.asarray(list(weight_list))
    # pred_out_vect = []
    # for i in range(0,len(df['feature1'])):
    #     feature_vector = list(df.iloc[i,:])
    #     feature_vector = feature_vector[:len(feature_vector)-1]
    #     feature_vector.insert(0,1)
    #     feature_vector_arr = np.asarray(feature_vector)
    #     print("featur vector",feature_vector)
    #     pred_out_vect.append(sigmoid_z(np.dot(feature_vector_arr,weight_list_1)))
    # pred_out_vect_arr = np.asarray(pred_out_vect)
    orignal_cost = calc_likehihood_est(weight_list_1,df)
    print(weight_list_1,"weight_list_1")
    print(orignal_cost,"Before function call")
    for i in range(0,5):
        sum = 0
        for j in range(0,len(df['feature1'])):
            feature_vector = list(df.iloc[j,:])
            label =  feature_vector[len(feature_vector)-1]
            feature_vector = feature_vector[:len(feature_vector)-1]            
            feature_vector.insert(0,1)
            feature_vector_arr = np.asarray(feature_vector)
            sum = sum - alpha*(sigmoid_z(np.dot(feature_vector_arr,weight_list_1)) - label)*feature_vector_arr[i]
        weight_list_1[i] = weight_list_1[i] + sum
    print(weight_list_1,"After modification")
    new_cost = calc_likehihood_est(weight_list_1,df)
    print(new_cost,"After function call")
    return orignal_cost,new_cost,weight_list_1


file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
df = prep.store_to_dataframe()
df = prep.normalize(df)
weight_list_test = [1,0,0,0,0]
orignal_cost,new_cost,weight_list_1 = logistic_regression(df,0.002,weight_list_test)
print(weight_list_1)
print(orignal_cost)
print(new_cost)
final_weight_list = []
cnt = 1
orignal_cost = 2
new_cost = 1
while abs(orignal_cost - new_cost) > 0.001:
    orignal_cost,new_cost,weight_list_1 = logistic_regression(df,0.3,weight_list_1) 
    print("change",orignal_cost - new_cost)   
    print(orignal_cost,new_cost,weight_list_1)
    cnt += 1
final_weight_list_1 = np.asarray(final_weight_list)
test_list = [1,0.60816,7.868,2.8108,6.2612]
test_list_arr = np.asarray(test_list)
print(sigmoid_z(np.dot(final_weight_list_1,test_list_arr)))
