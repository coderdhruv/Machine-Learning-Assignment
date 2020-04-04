file1 = open("a1_d3.txt","r")
content = file1.readlines()
content = [x.strip() for x in content] 
#method to assign each word a frequency and its output parameter(0/1)
words_0 = {}
words_1 = {}
cnt_0 = 0
cnt_1 = 0
for i in range(0,len(content)):
    flag = content[i][len(content[i])-1]    
    sentence = content[i].split(' ')    
    sentence[len(sentence)-1] = sentence[len(sentence)-1][:-3]        
    for j in sentence:
        print("j",j) 
        print(flag)       
        if flag == '0':
            cnt_0 += 1            
            if j not in words_0:
                words_0[j] = 1
            else:
                words_0[j] += 1
        if flag == '1':
            cnt_1 += 1            
            if j in words_1:
                words_1[j] += 1
            else:
                words_1[j] = 1
prob_words_0 = {}
prob_words_1 = {}
for key,value in words_0.items():
    if key not in prob_words_0:
        prob_words_0[key] = words_0[key]/cnt_0

for key,value in words_1.items():
    if key not in prob_words_1:
        prob_words_1[key] = words_1[key]/cnt_1

print(prob_words_0,"prob_words_0")
print(prob_words_1,"prob_words_1")