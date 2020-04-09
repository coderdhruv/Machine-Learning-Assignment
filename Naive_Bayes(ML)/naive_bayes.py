from preprocessing import preprocess

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
    print(prob_words_0,prob_words_1)
    for i in sentence_list:
        prob_1 = prob_1 * prob_words_1[i] 
    for i in sentence_list:
        prob_0 = prob_0 * prob_words_0[i]
    final_prob_1 = (prob_1 * total_prob_1)/((prob_1 * total_prob_1) + (prob_0 * total_prob_0))
    return final_prob_1


    
file1 = open("a1_d3.txt","r")
prep = preprocess(file1)
print(prep.text_to_list())
words_0,words_1,cnt_0,cnt_1 = prep.store_to_dict()
#print(words_0,words_1,cnt_0,cnt_1)
prob_words_0,prob_words_1 = finding_words_prob(words_0,words_1)
#print(prob_words_0,prob_words_1)
total_prob_0 = cnt_0/(cnt_0+cnt_1)
total_prob_1 = cnt_1/(cnt_1+cnt_0)
print(total_prob_0,total_prob_1)
input_sentence = input()
print(find_naive_bayes_prob(input_sentence,prob_words_0,prob_words_1,cnt_0,cnt_1))
