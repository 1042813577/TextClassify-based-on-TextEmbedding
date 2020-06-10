from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import jieba
import config
from collections import Counter
import pickle


stopwords = [word.strip() for word in open(config.stopwords_path,'r',encoding='utf-8').readlines()]

def data_extraction(corpus_path):
    with open(corpus_path,'r',encoding='utf-8') as f:
        data = f.readlines()
        f.close()
    sentences = []
    for line in data:
        line = list(jieba.cut(str(line).strip()))
        for word in line:
            if word not in stopwords:
                sentences.append(word)
    with open(config.sentence_path,'w',encoding='utf-8') as f_in:
        f_in.write(' '.join(sentences))
        f_in.close()



#build vector
def train_w2v(sentence_path):
    w2v_model = Word2Vec(LineSentence(sentence_path),workers=4,min_count=2)
    w2v_model.save(config.w2v_path)

def load_w2v(model_path):
    model = Word2Vec.load(model_path)
    return model


#sif
def word_freq(corpus_path):
    word_list = []
    with open(corpus_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            word_list += line.split()

    cc = Counter(word_list)
    num_all = sum(cc.values())
    word_fre = {}
    for word in cc.keys():
        word_fre[word] = cc[word] / num_all

    with open(config.fre_path,'wb') as f_in:
        pickle.dump(word_fre,f_in,pickle.HIGHEST_PROTOCOL)
    f_in.close()

def load_word_fre(fre_path):
    with open(fre_path,'rb') as f:
        return pickle.load(f)



if __name__ == "__main__":
    # data_extraction(config.corpus_path)
    # train_w2v(config.sentence_path)
    # word_freq(config.sentence_path)
    word_fre = load_word_fre(config.fre_path)
    print(word_fre.get("机油"))
    w2v = load_w2v(config.w2v_path)
    print(w2v["机油"])



