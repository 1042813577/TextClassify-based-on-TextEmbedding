
import tensorflow as tf
import json
import os

#结巴分词自定义词典路径
cut_words_dict_path = "resource/my_dict.txt"

#bert配置


#doc2vec配置文件


# is_develop = json.load(open("./branch.json"))["is_develop"] #开发环境还是发布环境
#
# if is_develop:
#     '''
#     发布环境
#     '''
#     pass
# else:
#     '''
#     开发环境
#     '''
#     pass

max_clustering_sim = 0.99999999
ap_damping = 0.8

# gpu_path = os.veriron['CUDA_VISIBLE_DEVICES'] = '0'
# gpu_options = tf.GPUOptions(allow_growth=True)

stopwords_path = "..\\data\\stopwords.txt"
sentence_path = "..\\data\\small_corpus.txt"
corpus_path = "..\\data\\medical_true.txt"
w2v_path = "..\\data\\w2v.model"
fre_path = "..\\data\\word_freq.pkl"








