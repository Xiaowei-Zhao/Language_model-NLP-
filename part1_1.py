from lang_model import *

# 1. Use langID.test.txt to compute the perplexity for n=1,2 and 3 for each language
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
def raw_perplexity():
    en=LangModel("./langID_data_tok/HW1english.txt")
    test=corpus_reader("test.txt", lexicon=en.lexicon)
    print("The perplexity for English(Unigram) is : {}".format(en.perplexity(test,1)))
    en=LangModel("./langID_data_tok/HW1english.txt")
    test=corpus_reader("test.txt", lexicon=en.lexicon)               
    print("The perplexity for English(Bigram) is : {}".format(en.perplexity(test,2))) 
    en=LangModel("./langID_data_tok/HW1english.txt")
    test=corpus_reader("test.txt", lexicon=en.lexicon) 
    print("The perplexity for English(Trigram) is : {}".format(en.perplexity(test,3)))  
    
    fr=LangModel("./langID_data_tok/HW1french.txt")
    test=corpus_reader("test.txt", lexicon=fr.lexicon)
    print("The perplexity for French(Unigram) is : {}".format(fr.perplexity(test,1)))
    fr=LangModel("./langID_data_tok/HW1french.txt")
    test=corpus_reader("test.txt", lexicon=fr.lexicon)
    print("The perplexity for French(Bigram) is : {}".format(fr.perplexity(test,2)))
    fr=LangModel("./langID_data_tok/HW1french.txt")
    test=corpus_reader("test.txt", lexicon=fr.lexicon)
    print("The perplexity for French(Triigram) is : {}".format(fr.perplexity(test,3)))
    
    ge=LangModel("./langID_data_tok/HW1german.txt")
    test=corpus_reader("test.txt", lexicon=ge.lexicon)
    print("The perplexity for German(Unigram) is : {}".format(ge.perplexity(test,1)))
    ge=LangModel("./langID_data_tok/HW1german.txt")
    test=corpus_reader("test.txt", lexicon=ge.lexicon)
    print("The perplexity for German(Bigram) is : {}".format(ge.perplexity(test,2)))
    ge=LangModel("./langID_data_tok/HW1german.txt")
    test=corpus_reader("test.txt", lexicon=ge.lexicon)
    print("The perplexity for German(Trigram) is : {}".format(ge.perplexity(test,3)))
    
raw_perplexity()
    