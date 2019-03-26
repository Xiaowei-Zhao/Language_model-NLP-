from lang_model import *
import numpy as np

# 1.2 Use langID.test.txt together with langID.gold.txt together with the trained language models to
# see how accurately we can classify a sentence into the right language.
def raw_accuracy():
    testfile = './langID_data_tok/LangID.test.txt'
    test_file = 'test.txt'
    en=LangModel("./langID_data_tok/HW1english.txt")
    fr=LangModel("./langID_data_tok/HW1french.txt")
    ge=LangModel("./langID_data_tok/HW1german.txt")
    result1=[]
    result2=[]
    result3=[]
    with open(testfile, 'r',encoding='latin-1') as f:
        for i in f:
            line = i.split('.')[1:]
            line = '.'.join(line)[1:]
            with open(test_file, 'w',encoding='latin-1') as output:
                output.write(line)
            test=corpus_reader("test.txt", lexicon=en.lexicon)
            en1=en.perplexity(test,1) 
            test=corpus_reader("test.txt", lexicon=en.lexicon)               
            en2=en.perplexity(test,2)  
            test=corpus_reader("test.txt", lexicon=en.lexicon) 
            en3=en.perplexity(test,3)
    
            test=corpus_reader("test.txt", lexicon=fr.lexicon)
            fr1=fr.perplexity(test,1)
            test=corpus_reader("test.txt", lexicon=fr.lexicon)
            fr2=fr.perplexity(test,2)
            test=corpus_reader("test.txt", lexicon=fr.lexicon)
            fr3=fr.perplexity(test,3)
            
            test=corpus_reader("test.txt", lexicon=ge.lexicon)
            ge1=ge.perplexity(test,1)
            test=corpus_reader("test.txt", lexicon=ge.lexicon)
            ge2=ge.perplexity(test,2)
            test=corpus_reader("test.txt", lexicon=ge.lexicon)
            ge3=ge.perplexity(test,3)
            
            class1 = np.argmin([en1, fr1, ge1])
            class2 = np.argmin([en2, fr2, ge2])
            class3 = np.argmin([en3, fr3, ge3])
            
            classes = ['EN', 'FR', 'GR']
            result1.append(classes[class1])
            result2.append(classes[class2])
            result3.append(classes[class3])
            
    return result1, result2, result3

def cal_raw_accuracy(result,n):
    goldfile='./langID_data_tok/LangID.gold.txt'
    gold=[]
    with open(goldfile, 'r', encoding='latin-1') as f:
        for i in f:
            language = i.split(' ')
            language = language[1][0:2]
            gold.append(language)
    gold=gold[1:]


    dict = {}
    dict['EN'] = 0
    dict['FR'] = 0
    dict['GR'] = 0
    for i in range(len(gold)):
        dict['EN'] += 1
        dict['FR'] += 1
        dict['GR'] += 1
        if gold[i] != result[i]:
            dict[gold[i]] -= 1
            dict[result[i]] -= 1

    for key in dict:
        dict[key] = dict[key] / len(gold)
        
    gold=np.array(gold)
    result=np.array(result)
    acc=sum(gold==result)/len(gold)
    
    if n==1:
        print('The accuracy of unigram model lists as follows:')
    if n==2:
        print('The accuracy of bigram model lists as follows:')
    if n==3:
        print('The accuracy of trigram model lists as follows:')
        
    print('For Yes or No classification, the accuracy is {}'.format(dict))
    print('For all 3-class classification, the accuracy is {}'.format(acc))


result1, result2, result3=raw_accuracy() 
cal_raw_accuracy(result1,1) 
cal_raw_accuracy(result2,2) 
cal_raw_accuracy(result3,3)      
            
