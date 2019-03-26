from lang_model import *
import numpy as np

# Part 3- Implementing Katz back off smoothing

testfile = 'langID_data_tok/LangID.test.txt'
test_file = 'test.txt'
sentences=[]
with open(testfile, 'r',encoding='latin-1') as f:
    for i in f:
        sentence = i.split('.')[1:]
        sentence = '.'.join(sentence)[1:]
        sentences.append(sentence)
with open(test_file, 'w',encoding='latin-1') as w:
    for i in sentences:
        w.write(i)
        
def katz_backoff_smoothed_perplexity():
    en=LangModel("./langID_data_tok/HW1english.txt")
    test=corpus_reader("test.txt", lexicon=en.lexicon)               
    print("The Katz backoff smoothed perplexity for English(Bigram) is : {}".format(en.perplexity_katz_backoff_smoothed_bigram_probability(test))) 
    
    fr=LangModel("./langID_data_tok/HW1french.txt")
    test=corpus_reader("test.txt", lexicon=fr.lexicon)
    print("The Katz backoff smoothed perplexity for French(Bigram) is : {}".format(fr.perplexity_katz_backoff_smoothed_bigram_probability(test)))
    
    ge=LangModel("./langID_data_tok/HW1german.txt")
    test=corpus_reader("test.txt", lexicon=ge.lexicon)
    print("The Katz backoff smoothed perplexity for German(Bigram) is : {}".format(ge.perplexity_katz_backoff_smoothed_bigram_probability(test)))

def katz_backoff_smoothed_accuracy():
    testfile = './langID_data_tok/LangID.test.txt'
    test_file = 'test.txt'
    en=LangModel("./langID_data_tok/HW1english.txt")
    fr=LangModel("./langID_data_tok/HW1french.txt")
    ge=LangModel("./langID_data_tok/HW1german.txt")
    result2=[]
    with open(testfile, 'r',encoding='latin-1') as f:
        for i in f:
            line = i.split('.')[1:]
            line = '.'.join(line)[1:]
            with open(test_file, 'w',encoding='latin-1') as output:
                output.write(line)
            test=corpus_reader("test.txt", lexicon=en.lexicon)               
            en2=en.perplexity_katz_backoff_smoothed_bigram_probability(test)  
    
            test=corpus_reader("test.txt", lexicon=fr.lexicon)
            fr2=fr.perplexity_katz_backoff_smoothed_bigram_probability(test)
            
            test=corpus_reader("test.txt", lexicon=ge.lexicon)
            ge2=ge.perplexity_katz_backoff_smoothed_bigram_probability(test)
            
            class2 = np.argmin([en2, fr2, ge2])
            classes = ['EN', 'FR', 'GR']
            result2.append(classes[class2])
         
    return result2

def cal_katz_backoff_smoothed_accuracy(result):
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
    
    print('The accuracy of Katz backoff smoothed bigram model lists as follows:')   
    print('For Yes or No classification, the accuracy is {}'.format(dict))
    print('For all 3-class classification, the accuracy is {}'.format(acc))

katz_backoff_smoothed_perplexity() 
result2=katz_backoff_smoothed_accuracy() 
cal_katz_backoff_smoothed_accuracy(result2)



