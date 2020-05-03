from preprocessing import preprocess
from performance_metric import performance
import math
import numpy as np

f = open('output_l2.txt','w')
f.write('weight list after each iteration:\n')
def sigmoid_z(z):
    return 1/(1 + np.exp(-z))
    
def list_to_string(list1):
    s = ""
    for i in list1:
        s = s + str(i) + " "
    return s

def calc_likehihood_est(weight_list_1,df,lmbda):    
    sum = 0
    reg_term = 0
    for i in range(0,len(weight_list_1)):
        reg_term = reg_term + (lmbda*(weight_list_1[i])*(weight_list_1[i]))
    epsilon = 1e-5
    n = len(df['feature1'])
    for i in range(0,len(df['feature1'])):
        feature_vector = list(df.iloc[i,:])
        label = feature_vector[len(feature_vector)-1]
        feature_vector.insert(0,1)
        feature_vector = feature_vector[:len(feature_vector)-1]
        feature_vector_arr = np.array(feature_vector)
        z = sigmoid_z(np.dot(weight_list_1,feature_vector_arr))
        a = label*(np.log(z + epsilon))
        b = (1 - label)*(np.log(1 - z + epsilon))
        like_est = (-a + -b)/(n)
        sum = sum + like_est
    return sum + reg_term    


def logistic_regression(df,alpha,weight_list,lmbda):
    weight_list_1 = np.array(list(weight_list))
    orignal_cost = calc_likehihood_est(weight_list_1,df,0.01)
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
            sum = sum - alpha*(((sigmoid_z(np.dot(feature_vector_arr,weight_list_1)) - label)*feature_vector_arr[i]))
        weight_list_1[i] = weight_list_1[i] + sum - alpha*(lmbda*weight_list_1[i])
    print(weight_list_1,"After modification")
    f.write(list_to_string(weight_list_1) + "\n")
    new_cost = calc_likehihood_est(weight_list_1,df,1)
    print(new_cost,"After function call")
    return orignal_cost,new_cost,weight_list_1

alpha = input("Alpha:")
lamda = input("lambda:")
file = open('./data_banknote_authentication.txt','r')
prep = preprocess(file)
df = prep.store_to_dataframe()
df1,df2 = prep.test_train_data_set(df,0.2)
weight_list_test = np.random.randn(5)
orignal_cost,new_cost,weight_list_1 = logistic_regression(df1,alpha,weight_list_test,lamda)
print(weight_list_1)
print(orignal_cost)
print(new_cost)
final_weight_list = []
cnt = 1
orignal_cost = 2
new_cost = 1
epoch = 0
while abs(orignal_cost - new_cost) > 0.005 and epoch < 100:
    orignal_cost,new_cost,weight_list_1 = logistic_regression(df1,alpha,weight_list_1,lamda)
    epoch += 1 
    print("change",orignal_cost - new_cost)   
    print(orignal_cost,new_cost,weight_list_1)
final_weight_list_1 = np.array(weight_list_1)
print(final_weight_list_1,'final_weight_list')
f.write("final weight list" + list_to_string(final_weight_list_1) + "\n")
p = performance()
accuracy = p.accuracy(final_weight_list_1,df2,f)
fscore = p.f_score()
print('accuracy:',accuracy)
f.write("accuracy:" + str(accuracy) + "\n")
print('fscore:',fscore)
f.write("fscore:" + str(fscore) + "\n")
f.close()
