# coding=utf-8
import numpy as np
from sklearn.cluster import AffinityPropagation as AP
import config

import jieba
from data.build_vec import load_w2v,load_word_fre

w2v = load_w2v(config.w2v_path)
word_freq = load_word_fre(config.fre_path)

max_clusters_sim = config.max_clustering_sim
apdampling = config.ap_damping


class Cluster:
    def get_cos_similarity(self,v1,v2):
        # np.mat:矩阵
        vector_a = np.mat(v1)
        vector_b = np.mat(v2)
        num = float(vector_a * vector_b.T)
        # 求范数：np.linalg.norm(x, ord=None, axis=None, keepdims=False)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num /denom
        sim = 0.5 + 0.5 * cos
        return sim

    # 计算多个句子之间的相似度
    def get_data_sims(self,x_n):
        text_sim_list = []
        for m in x_n:
            m = np.array(m)
            temps = []
            for n in x_n:
                n = np.array(n)
                sim = self.get_cos_similarity(m,n)
                temps.append(sim)
            text_sim_list.append(temps)

        x_sims = np.array(text_sim_list)
        # 由于要和自己比较，max肯定为1
        max_sim = np.max(x_sims)
        min_sim = np.min(x_sims)
        mid_sim = np.median(x_sims)

        return x_sims,max_sim,min_sim,mid_sim

    # 输入：文本/句子集合，（word2vec矩阵）    返回：句子/文本向量
    def get_text_data(self,sent_list):
        a = 0.001
        row = w2v.wv.vector_size
        col = len(sent_list)

        sent_mat = np.mat(np.zeros((col,row)))
        text_vector_list = []
        # 处理多个句子
        # 出处：论文 A Simple but tought to beat sentence embedding算法1
        for i,sent in enumerate(sent_list):
            text_id = sent_list[i].get('text_id')
            text_content = sent_list[i].get('text_content')
            new_sent = list(jieba.cut(str(text_content).strip()))
            if not new_sent:continue
            sent_vec = np.zeros(row)
            # 惩罚高频词多的句子
            for word in new_sent:
                pw = word_freq.get(word,100)
                w = a / (a + pw)
                try:
                    vec = np.array(w2v.wv[word])
                    sent_vec += w * vec
                except:
                    pass
            text_vector_list.append({"text_id":text_id,"text_vector":sent_vec})
            sent_mat[i,:] += np.mat(sent_vec)
            sent_mat[i,:] /= len(new_sent)
            # 原始论文中还要减去第一主成分，语料在百万级以下时不建议做
        return sent_mat,text_vector_list


# 使用sklearn中的AffinityPropagation。 输入：两两相似度矩阵    输出：聚类结果
class AffinityPropagation(object):
    def get_result_clustering_by_ap(self,text_list):
        cluster = Cluster()
        Xn, text_vector_list = cluster.get_text_data(text_list)

        x_sims, max_sim, min_sim, mid_sim = cluster.get_data_sims(Xn)
        # 两两相似度矩阵
        print('x_sims:',x_sims)

        text_clustering_list = []
        if min_sim > max_clusters_sim:
            for i in range(len(text_vector_list)):
                text_id = text_vector_list[i].get("text_id")
                text_clustering_list.append({"text_id":text_id,"class_num":"0"})
        else:
            ap = AP(damping=apdampling,
                    max_iter=1000,
                    convergence_iter=100,
                    preference=mid_sim,  # 相似度达到多少聚为一类
                    affinity="precomputed").fit(x_sims)
            # labels:聚类结果
            labels = ap.labels_
            monitor_set = set()
            if -1 not in labels:
                for i in range(len(labels)):
                    monitor_set.add(labels[i])
                    class_num = str(labels[i])
                    text_id = text_vector_list[i].get("text_id")
                    text_clustering_list.append({"text_id":text_id,"class_num":class_num})
                return text_clustering_list
            else:
                adapt_ap_damping = 0.5
                ap = AP(
                    max_iter=1000,
                    convergence_iter=100,
                    preference=mid_sim,
                    affinity="precomputed",damping=adapt_ap_damping).fit(x_sims)
                for i in range(len(labels)):
                    monitor_set.add(labels[i])
                    class_num = str(labels[i])
                    text_id = text_vector_list[i].get("text_id")
                    text_clustering_list.append({"text_id":text_id,"class_num":class_num})
                return text_clustering_list




if __name__ == "__main__":
    text_list = [
        {
            "text_id":"123",
            "text_content":"我觉得也是，可是车主是以前没这么重，选吧助理泵换了不行，又把放向机换了"
        },
        {
            "text_id":"1234",
            "text_content":"技师说：你这个有没有电脑检测故障代码"
        },
        {
            "text_id": "1235",
            "text_content": "通用6L45变速箱，原地换挡位PRND车辆闯动，行驶升降档正常，4轮离地换挡无冲击感"
        },
        {
            "text_id": "1236",
            "text_content": "可能液力变矩器轴头磨损，泄压了，需要专用电脑清除变速箱适应值升级变速箱程序"
        }
    ]
    ap = AffinityPropagation()
    results = ap.get_result_clustering_by_ap(text_list)
    print(results)





