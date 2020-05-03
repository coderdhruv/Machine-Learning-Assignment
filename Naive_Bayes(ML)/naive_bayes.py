from preprocessing import preprocess
import math

f = open("ouput_naive_bayes_1.txt","w")

def finding_words_prob(words_0,words_1):
    prob_words_0 = {}
    prob_words_1 = {}
    for key,value in words_0.items():
        if key not in prob_words_0:
            prob_words_0[key] = words_0[key]/cnt_0
    for key,value in words_1.items():
        if key not in prob_words_1:
            prob_words_1[key] = words_1[key]/cnt_1
    return prob_words_0,prob_words_1

def laplace_smoothing_prob(sentence,prob_words_0,prob_words_1,cnt_0,cnt_1):
    sentence_list = sentence.split(' ')
    dimension_d = len(sentence_list)
    for i in sentence_list:
        if i in prob_words_1:
            no_of_words_words_1 = prob_words_1[i]*cnt_1
            num_for_lap_smoothing = no_of_words_words_1 + 1
            denom_for_lap_smoothing = cnt_1 + dimension_d
            prob_words_1[i] = num_for_lap_smoothing/denom_for_lap_smoothing
        else:
            prob_words_1[i] = 1/(cnt_1 + dimension_d)
    for i in sentence_list:
        if i in prob_words_0:
            no_of_words_words_0 = prob_words_0[i]*cnt_0
            num_for_lap_smoothing = no_of_words_words_0 + 1
            denom_for_lap_smoothing = cnt_0 + dimension_d
            prob_words_0[i] = num_for_lap_smoothing/denom_for_lap_smoothing
        else:
            prob_words_0[i] = 1/(cnt_0 + dimension_d)
    return prob_words_0,prob_words_1

def find_naive_bayes_prob(sentence,prob_words_0,prob_words_1,cnt_0,cnt_1):
    sentence_list = sentence.split(' ')
    prob_words_0,prob_words_1 = laplace_smoothing_prob(sentence,prob_words_0,prob_words_1,cnt_0,cnt_1)
    prob_1 = 1
    prob_0 = 1
    #print(prob_words_0,prob_words_1)
    for i in sentence_list:
        prob_1 = prob_1 * prob_words_1[i] 
    for i in sentence_list:
        prob_0 = prob_0 * prob_words_0[i]
    final_prob_1 = (prob_1 * total_prob_1)/((prob_1 * total_prob_1) + (prob_0 * total_prob_0))
    return final_prob_1
def accuracy(file1,prob_words_0,prob_words_1,cnt_0,cnt_1):
    lines = []
    with open(file1.name) as f:
        lines = [line for line in f if line.strip()]
    cnt = 0    
    for i in range(0,len(lines)):
        #print(lines[i])
        # #print(label)
        sentence = lines[i].split(' ')
        label = sentence[len(sentence)-1][-2]
        #print(label)
        sentence[len(sentence)-1] = sentence[len(sentence)-1][:-3]
        r = ''
        for i in sentence:
            r = r + i + ' '
        value_pred = find_naive_bayes_prob(r,prob_words_0,prob_words_1,cnt_0,cnt_1)
        if value_pred >= 0.5:
            if label == '1':
                cnt += 1
        elif value_pred < 0.5:
            if label == '0':
                cnt += 1
    return cnt/len(lines)
def f_score(file1,prob_words_0,prob_words_1,cnt_0,cnt_1):
    lines = []
    with open(file1.name) as f:
        lines = [line for line in f if line.strip()]
    cnt = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0    
    for i in range(0,len(lines)):
        #print(lines[i])
        # #print(label)
        sentence = lines[i].split(' ')
        label = sentence[len(sentence)-1][-2]
        #print(label)
        sentence[len(sentence)-1] = sentence[len(sentence)-1][:-3]
        r = ''
        for i in sentence:
            r = r + i + ' '
        value_pred = find_naive_bayes_prob(r,prob_words_0,prob_words_1,cnt_0,cnt_1)
        if value_pred >= 0.5:
            if label == '1':
                cnt += 1
                tp += 1
            elif label == '0':
                fp += 1
        elif value_pred < 0.5:
            if label == '0':
                cnt += 1
                tn += 1
            elif label == '1':
                fn += 1
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    fscore = (2*(recall)*(precision))/(recall + precision)
    return fscore 
    


acc = []
fscore = []
l = 0
#findind accuracy for k-folds
for i in range(0,5): 
    file = open("a1_d3.txt","r")
    prep1 = preprocess(file)
    prep1.split_file_test_train(l,l+0.2)   
    file1 = open("train.txt","r")
    prep = preprocess(file1)
    print(prep.text_to_list())
    words_0,words_1,cnt_0,cnt_1 = prep.store_to_dict()
    print(cnt_0,cnt_1)
    total = cnt_0 + cnt_1
    print(total)
    total_prob_0 = (cnt_0)/(cnt_0 + cnt_1)
    total_prob_1 = (cnt_1)/(cnt_1 + cnt_0)
    print(total_prob_0,total_prob_1)
    prob_words_0,prob_words_1 = finding_words_prob(words_0,words_1)
    print(prob_words_0,prob_words_1)
    file2 = open("test.txt","r")
    acc.append(accuracy(file2,prob_words_0,prob_words_1,cnt_0,cnt_1))
    fscore.append(f_score(file2, prob_words_0, prob_words_1, cnt_0, cnt_1))
    l = l + 0.2
print("accuracy for validation step for 5 fold cross validation")
f.write("accuracy for validation step for 5 fold cross validation:\n")
avg_acc = 0
cnt = 1
for i in acc:
    avg_acc = avg_acc + i
    f.write("accuracy for " + str(cnt) + " fold is : " + str(i) + "\n")
    cnt += 1
    print(i)
print('avg accuracy:',avg_acc/5)
avg_acc = avg_acc/5
#standard deviation for accuracy
std_dv_acc = 0
for i in acc:
    std_dv_acc = std_dv_acc + ((i - avg_acc)*(i - avg_acc))
std_dv_acc = math.sqrt(std_dv_acc/5)
f.write('average accuracy is: ' + str(avg_acc) + " +- " + str(std_dv_acc) + "\n")
print("fscore for validation step for 5 fold cross validation")
f.write("fscore for validation step for 5 fold cross validation:\n")
avg_fscore = 0
cnt = 1
for i in fscore:
    avg_fscore = avg_fscore + i
    f.write("fscore for " + str(cnt) + " fold is : " + str(i) + "\n")
    cnt += 1
    print(i)
print('avg fscore:',avg_fscore/5)
avg_fscore = avg_fscore/5
std_dv_fsc = 0
#standard deviation for fscore
for i in fscore:
    std_dv_fsc = std_dv_fsc + ((i-avg_fscore)*(i-avg_fscore))
std_dv_fsc = math.sqrt(std_dv_fsc/5)
f.write('average fscore is: ' + str(avg_fscore) + " +- " + str(std_dv_fsc) + "\n")