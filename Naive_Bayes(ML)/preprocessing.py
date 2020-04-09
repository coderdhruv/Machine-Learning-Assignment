class preprocess:
    words_0 = {}
    words_1 = {}
    cnt_0 = 0
    cnt_1 = 0
    content = []
    def __init__(self,file1):
        self.file1 = file1
    def text_to_list(self):
        self.content = self.file1.readlines()
        self.content = [x.strip() for x in self.content]
        return self.content
    def store_to_dict(self):
        for i in range(0,len(self.content)):
            flag = self.content[i][len(self.content[i])-1]    
            sentence = self.content[i].split(' ')    
            sentence[len(sentence)-1] = sentence[len(sentence)-1][:-3]       
            for j in sentence:
                if j != "," and j != "!" and j!= "-" and j != "/":
                    print(j)       
                    if flag == '0':
                        self.cnt_0 += 1            
                        if j not in self.words_0:
                            self.words_0[j] = 1
                        else:
                            self.words_0[j] += 1
                    if flag == '1':
                        self.cnt_1 += 1            
                        if j in self.words_1:
                            self.words_1[j] += 1
                        else:
                            self.words_1[j] = 1
        return self.words_0,self.words_1,self.cnt_0,self.cnt_1
    
    
    


