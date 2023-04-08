"""
HeterHyperGraph
HyperBipGraph
所有的游走类方法，包括baseline都在这里
"""
import os
import pandas as pd
from scipy.io import loadmat
import networkx as nx
from walks.HeterHyperDeepWalk import HeterHyperDeepWalk
from walks.node2vec import node2vec
from walks.DeepWalk import DeepWalk
from walks.LINE import LINE
from gensim.models import Word2Vec
from time import time
import numpy as np
import heapq
import torch
import random
import pickle
import argparse

# get data of a city
def get_mat_dataset(city):
    sources_directory = '/workspace/dataset/Foursquare/sources'
    city_mat_fp = os.path.join(sources_directory, 'dataset_connected_{city}.mat')
    mat_fp = city_mat_fp.format(city=city)
    mat_dict = loadmat(mat_fp)
    # print(mat_dict.keys())
    checkins_np = mat_dict['selected_checkins']
    friendship_new_np = mat_dict['friendship_new']
    friendship_old_np = mat_dict['friendship_old']
    users_np = mat_dict['selected_users_IDs']
    venue_np = mat_dict['selected_venue_IDs']
    # print(users_np.shape, venue_np.shape, checkins_np.shape, friendship_new_np.shape, friendship_old_np.shape)
    venues_df = pd.DataFrame(venue_np, columns=["vid"])
    venues_df.vid = venues_df.vid.map(lambda x: x[0])
    users_df = pd.DataFrame(users_np, columns=["uid"])
    checkins_df = pd.DataFrame(checkins_np, columns=['uid', 'time', 'vid', 'category'])
    friendship_new_df = pd.DataFrame(friendship_new_np, columns=['uid1', 'uid2'])
    friendship_old_df = pd.DataFrame(friendship_old_np, columns=['uid1', 'uid2'])
    # print('选定用户\n',users_df.head())
    # print('选定位置\n',venues_df.head())
    # print('朋友新\n',friendship_new_df.head())
    # print('朋友旧\n',friendship_old_df.head())
    # print('签到\n',checkins_df.head())

    # # 规范社交链接的数据，有friendship中有的点没有签到数据
    # set_check_usr = set(checkins_df['uid'])
    # delete_ind_friends = []
    # for ii in range(len(friendship_new_df)):
    #     if friendship_new_df['uid1'][ii] not in set_check_usr  or  friendship_new_df['uid2'][ii] not in set_check_usr:
    #         delete_ind_friends.append(ii)
    # friendship_new_df = friendship_new_df.drop(delete_ind_friends, axis=0)
    # friendship_new_df.index = range(0, len(friendship_new_df))
    #
    # delete_ind_friends = []
    # for ii in range(len(friendship_old_df)):
    #     if friendship_old_df['uid1'][ii] not in set_check_usr or friendship_old_df['uid2'][ii] not in set_check_usr:
    #         delete_ind_friends.append(ii)
    # friendship_old_df = friendship_old_df.drop(delete_ind_friends, axis=0)
    # friendship_old_df.index = range(0, len(friendship_old_df))
    #
    # # 签到中出现的friend没有的用户也删掉
    # set_friends_old_usr = set(friendship_old_df['uid1']) | set(friendship_old_df['uid2'])
    # delete_ind_check = []
    # for ii in range(len(checkins_df)):
    #     if checkins_df['uid'][ii] not in set_friends_old_usr:
    #         delete_ind_check.append(ii)
    # checkins_df = checkins_df.drop(delete_ind_check, axis=0)
    # checkins_df.index = range(0, len(checkins_df))
    #
    # # 新的社交关系出现的旧社交关系中没有的用户也删掉
    # delete_ind_friends = []
    # for ii in range(len(friendship_new_df)):
    #     if friendship_new_df['uid1'][ii] not in set_friends_old_usr or friendship_new_df['uid2'][ii] not in set_friends_old_usr:
    #         delete_ind_friends.append(ii)
    # friendship_new_df = friendship_new_df.drop(delete_ind_friends, axis=0)
    # friendship_new_df.index = range(0, len(friendship_new_df))

    return users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df

# 对checkins的id重新赋值
def remakeID_checkins(checkins_df, friendship_new_df, friendship_old_df):
    set_u = set(checkins_df['uid']) | set(friendship_new_df['uid1']) | set(friendship_new_df['uid2']) | \
            set(friendship_old_df['uid1']) | set(friendship_old_df['uid2'])
    num_usr = len(set_u)
    num_time = len(set(checkins_df['time']))
    num_vid = len(set(checkins_df['vid']))
    num_category = len(set(checkins_df['category']))

    #偏移已考虑id从1开始
    timeID_offset = num_usr + 1
    vid_offset = timeID_offset + num_time
    category_offset = vid_offset + num_vid
    offset_DIC = {'time': timeID_offset, 'vid': vid_offset, 'category': category_offset}

    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        checkins_df['time'][ii] = checkins_df['time'][ii] + timeID_offset - 1

    count_ID = -1
    last_ID = -1
    checkins_df.sort_values(by="vid", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        tmp_ID = checkins_df['vid'][ii]
        if tmp_ID != last_ID:
            count_ID += 1
            last_ID = tmp_ID
            checkins_df['vid'][ii] = count_ID + vid_offset
        else:
            checkins_df['vid'][ii] = count_ID + vid_offset

    count_ID = -1
    last_ID = -1
    checkins_df.sort_values(by="category", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        tmp_ID = checkins_df['category'][ii]
        if tmp_ID != last_ID:
            count_ID += 1
            last_ID = tmp_ID
            checkins_df['category'][ii] = count_ID + category_offset
        else:
            checkins_df['category'][ii] = count_ID + category_offset
    checkins_df.sort_values(by="uid", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))

    return offset_DIC

# get Graph
def get_Graph(friendship_old_df, checkins_df, offset_DIC):
    # 个数
    num_usr = offset_DIC["time"] - 1
    num_t = len(set(checkins_df['time']))
    num_v = len(set(checkins_df['vid']))
    num_c = len(set(checkins_df['category']))

    # 计算签到频次 user考虑id从1开始 tvc在偏移中考虑了id
    usr_t_pc_list = [[0 for _t in range(num_t)] for _ in range(num_usr + 1)]
    usr_v_pc_list = [[0 for _v in range(num_v)] for _ in range(num_usr + 1)]
    usr_c_pc_list = [[0 for _c in range(num_c)] for _ in range(num_usr + 1)]

    for uid, time, vid, cate in zip(checkins_df['uid'], checkins_df['time'], checkins_df['vid'], checkins_df['category']):
        usr_t_pc_list[uid][time - offset_DIC['time']] += 1
        usr_v_pc_list[uid][vid - offset_DIC['vid']] += 1
        usr_c_pc_list[uid][cate - offset_DIC['category']] += 1

    #构建带权二部图 hyperBipGraph
    bipG_t = nx.Graph()
    bipG_v = nx.Graph()
    bipG_c = nx.Graph()
    for uid in range(1,num_usr + 1):
        for tid in range(num_t):
            if usr_t_pc_list[uid][tid]>0:
                bipG_t.add_edge(uid, tid + offset_DIC['time'], weight=usr_t_pc_list[uid][tid])
        for vid in range(num_v):
            if usr_v_pc_list[uid][vid] > 0:
                bipG_v.add_edge(uid, vid + offset_DIC['vid'], weight=usr_v_pc_list[uid][vid])
        for cid in range(num_c):
            if usr_c_pc_list[uid][cid] > 0:
                bipG_c.add_edge(uid, cid + offset_DIC['category'], weight=usr_c_pc_list[uid][cid])

    # 构建社交图
    social_G = nx.Graph()
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        social_G.add_edge(uid1, uid2, weight=1)

    # 构建异构社交图(异构边和社交边等价)
    HeterHyper_G = nx.Graph()
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        HeterHyper_G.add_edge(uid1, uid2, weight=1)
    for uid in range(1,num_usr + 1):
        for tid in range(num_t):
            if usr_t_pc_list[uid][tid]>0:
                HeterHyper_G.add_edge(uid, tid + offset_DIC['time'], weight=1)
        for vid in range(num_v):
            if usr_v_pc_list[uid][vid] > 0:
                HeterHyper_G.add_edge(uid, vid + offset_DIC['vid'], weight=1)
        for cid in range(num_c):
            if usr_c_pc_list[uid][cid] > 0:
                HeterHyper_G.add_edge(uid, cid + offset_DIC['category'], weight=1)

    return num_usr, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c, social_G, HeterHyper_G

# 余弦距离
def countCos(x,y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 计算accuracy，recall计算,超二部图用,不考虑新社交链接
def count_Acc_Recall(K, userVec_Bip_dic, num_user, friendship_df):
    start = time()

    # 计算余弦距离 考虑id从1开始
    prob_adj = np.zeros((num_user+1, num_user+1))
    for ii in range(1, num_user + 1):
        for jj in range(1, num_user + 1):
            if jj < ii:
                continue
            x = userVec_Bip_dic[ii]
            y = userVec_Bip_dic[jj]
            cos_dis = countCos(x, y)  # 后面是选值最大的，所以给距离加个负号
            prob_adj[ii][jj] = cos_dis
            prob_adj[jj][ii] = cos_dis

    P_K_list = []
    R_K_list = []
    # 计算精确度和召回率

    # 社交网络邻接矩阵 考虑id从1开始
    link_labels_M = np.zeros((num_user + 1, num_user + 1))
    for old_uid1, old_uid2 in zip(friendship_df['uid1'],friendship_df['uid2']):
        link_labels_M[old_uid1][old_uid2] = 1
        link_labels_M[old_uid2][old_uid1] = 1

    # 对每个user进行计算
    for link_u_p, link_u_l in zip(prob_adj, link_labels_M):
        # 只对正样本和抽取的负样本进行预测
        pre_ind = []
        tensor_link_u_l = torch.tensor(link_u_l)
        pre_ind = torch.eq(tensor_link_u_l, 1).nonzero().numpy().tolist()
        for preI in pre_ind:
            link_u_p[preI[0]] += 1000  # 正样本全参与预测
        pre_ind = torch.eq(tensor_link_u_l, 0).nonzero().numpy().tolist()
        for preI in random.sample(pre_ind,50):
            link_u_p[preI[0]] += 1000  # 负样本随机抽取50个

        num_1 = sum(link_u_l)
        if (num_1 < 1):
            continue
        link_u_p = list(link_u_p)
        link_u_l = list(link_u_l)
        max_num_index_list = list(map(link_u_p.index, heapq.nlargest(K, link_u_p)))
        N_u_true = 0
        for index in max_num_index_list:
            N_u_true += link_u_l[index]

        P_K_list.append(N_u_true / K)
        R_K_list.append(N_u_true / num_1)

    stop = time()
    print("运行时间/s：", str(stop - start) + "s")
    print("运行时间/min：", str((stop - start) / 60) + "min")
    print("运行时间/h：", str((stop - start) / (60 * 60)) + "H")

    return sum(P_K_list) / len(P_K_list), sum(R_K_list) / len(R_K_list)

# 计算accuracy，recall计算,考虑新旧的社交链接问题
def count_o2n_Acc_Recall(K, userVec, num_user, friendship_new_df, friendship_old_df):
    start = time()

    # 计算余弦距离 考虑id从1开始
    prob_adj = np.zeros((num_user+1, num_user+1))
    for ii in range(1, num_user + 1):
        for jj in range(1, num_user + 1):
            if jj < ii:
                continue
            x = userVec[ii]
            y = userVec[jj]
            cos_dis = countCos(x, y)  # 后面是选值最大的，所以给距离加个负号
            prob_adj[ii][jj] = cos_dis
            prob_adj[jj][ii] = cos_dis

    P_K_list = []
    R_K_list = []
    # 计算精确度和召回率

    # 社交网络邻接矩阵 考虑id从1开始
    link_new_labels_M = np.zeros((num_user + 1, num_user + 1))
    for uid1, uid2 in zip(friendship_new_df['uid1'],friendship_new_df['uid2']):
        link_new_labels_M[uid1][uid2] = 1
        link_new_labels_M[uid2][uid1] = 1
    # 旧社交网络邻接矩阵
    for uid1, uid2 in zip(friendship_old_df['uid1'],friendship_old_df['uid2']):
        link_new_labels_M[uid1][uid2] = 2  # 原new的社交链接中实际上是old+新增的社交链接，所以这里将新增社交链接中的旧链接去掉，防止影响后面预测（特别是召回率分母）
        link_new_labels_M[uid2][uid1] = 2

    # 对每个user进行计算
    for link_u_p, link_u_new_l in zip(prob_adj, link_new_labels_M):
        # 只对正样本和抽取的负样本进行预测
        tensor_link_u_new_l = torch.tensor(link_u_new_l)
        pre_ind = torch.eq(tensor_link_u_new_l, 1).nonzero().numpy().tolist()
        for preI in pre_ind:
            link_u_p[preI[0]] += 1000  # 正样本全参与预测
        pre_ind = torch.eq(tensor_link_u_new_l, 0).nonzero().numpy().tolist()
        for preI in random.sample(pre_ind,50):
            link_u_p[preI[0]] += 1000  # 负样本随机抽取50个

        link_u_p = list(link_u_p)   # ndarray 转 list
        link_u_l = list(link_u_new_l)
        num_1 = list(link_u_l).count(1)  # 该用户新增的链接个数
        if (num_1 < 1):
            continue
        max_num_index_list = list(map(link_u_p.index, heapq.nlargest(K, link_u_p)))
        N_u_true = 0
        for index in max_num_index_list:
            N_u_true += link_u_l[index]

        P_K_list.append(N_u_true / K)
        R_K_list.append(N_u_true / num_1)

    stop = time()
    print("运行时间/min：", str((stop - start) / 60) + "min")

    P_ = sum(P_K_list) / len(P_K_list)
    R_ = sum(R_K_list) / len(R_K_list)
    if P_+R_ != 0:
        F1_ = 2 * (P_ * R_) / (P_ + R_)
    else:
        F1_ = 0

    return P_, R_, F1_

# deepWalk主函数
def main_of_deepWalk(Graph, num_u, friendship_new_df, friendship_old_df):
    # 一些参数
    args = {'r': 10, 'l': 80, 'alpha': 0.2}

    kwargs = {}
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = 128
    kwargs["sg"] = 1  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = 3
    kwargs["window"] = 5
    kwargs["epochs"] = 5

    start = time()
    # print("-" * 10, "社交网络图上游走", "-" * 10)
    socialDeepWalk_mode = DeepWalk(args, Graph)
    socialWalks = socialDeepWalk_mode.HBdeepWalk()
    stop = time()
    # print("运行时间/min：", str((stop - start) / 60) + "min")

    # 将游走序列转换为字符
    for i in range(len(socialWalks)):
        for j in range(len(socialWalks[i])):
            socialWalks[i][j] = str(socialWalks[i][j])

    # print("-" * 10, "社交网络图word2vec开始训练", "-" * 10)
    start = time()
    kwargs["sentences"] = socialWalks
    # 直接使用现有的word2vec工具包
    social_w2v_model = Word2Vec(**kwargs)
    stop = time()
    print("运行时间/min：", str((stop - start) / 60) + "min")

    # 计算ACC和recall
    # print("-" * 10, "社交网络图计算ACC和recall", "-" * 10)
    # K = 10
    social_id_vec_list = [[]]  # 考虑id从1开始
    for uid in range(1, num_u + 1):
        social_id_vec_list.append(social_w2v_model.wv[str(uid)])

    # ACC, Recall, F1 = count_o2n_Acc_Recall(K, social_id_vec_list, num_node, friendship_new_df, friendship_old_df)
    ACC, Recall, F1 = 1, 1, 1

    return ACC, Recall, F1, social_id_vec_list

# node2vec主函数
def main_of_node2vec(Graph, num_u, friendship_new_df, friendship_old_df):
    # 一些参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--p', type=float, default=1, help='return parameter')
    parser.add_argument('--q', type=float, default=0.5, help='in-out parameter')
    parser.add_argument('--d', type=int, default=128, help='dimension')
    parser.add_argument('--r', type=int, default=10, help='walks per node')
    parser.add_argument('--l', type=int, default=80, help='walk length')
    parser.add_argument('--k', type=float, default=10, help='window size')

    args = parser.parse_args()

    vec = node2vec(args, Graph)
    embeddings = vec.learning_features(num_u)
    embeddings.insert(0,[])  #保持id从1开始的一致处理

    # # 计算ACC和recall
    # print("-" * 10, "社交网络图计算ACC和recall", "-" * 10)
    # K = 10
    # ACC, Recall, F1 = count_o2n_Acc_Recall(K, embeddings, num_node, friendship_new_df, friendship_old_df)
    ACC, Recall, F1 = 1, 1, 1

    return embeddings, ACC, Recall, F1

# LINE主函数
def main_of_LINE(Graph, num_u, friendship_new_df, friendship_old_df):
    model = LINE(Graph, embedding_size=128, order='second')
    model.train(batch_size=1024, epochs=70, verbose=2)
    embeddings = model.get_embeddings()
    social_id_vec_list = [[]] #保持id从1开始的一致处理

    for i in range(1, num_u + 1):
        social_id_vec_list.append(embeddings[i])

    # 计算ACC和recall
    # print("-" * 10, "计算ACC和recall", "-" * 10)
    # K = 10
    # ACC, Recall, F1 = count_o2n_Acc_Recall(K, social_id_vec_list, num_node, friendship_new_df, friendship_old_df)
    ACC, Recall, F1 = 1, 1, 1

    return embeddings, ACC, Recall, F1

# 异构超图的主函数
def main_of_hyperHeter(num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c, friendship_new_df, friendship_old_df):
    # 一些参数
    args = {'r': 10, 'l': 80, 'alpha': 0.2}

    kwargs = {}
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = 64
    kwargs["sg"] = 1  # skip gram
    kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
    kwargs["workers"] = 3
    kwargs["window"] = 5
    kwargs["epochs"] = 5

    start = time()
    print("-" * 10, "异构超图上游走", "-" * 10)
    HeterHyperDeepWalk_mode = HeterHyperDeepWalk(args, num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c,)
    HeterHyperWalks = HeterHyperDeepWalk_mode.HBdeepWalk()
    stop = time()
    print("运行时间/min：", str((stop - start) / 60) + "min")

    # 将游走序列转换为字符
    for i in range(len(HeterHyperWalks)):
        for j in range(len(HeterHyperWalks[i])):
            HeterHyperWalks[i][j] = str(HeterHyperWalks[i][j])

    print("-" * 10, "异构超图word2vec开始训练", "-" * 10)
    start = time()
    kwargs["sentences"] = HeterHyperWalks
    # 直接使用现有的word2vec工具包
    HeterHyper_w2v_model = Word2Vec(**kwargs)
    stop = time()
    print("运行时间/min：", str((stop - start) / 60) + "min")

    # 计算ACC和recall
    print("-" * 10, "异构超图计算ACC和recall", "-" * 10)
    K = 10
    HeterHyper_id_vec_list = [[]]  # 考虑id从1开始
    HeterHyper_all_id_vec_list = [[]]  # 考虑id从1开始，并且将utcv的vec都存下来
    for id in range(1, num_u + num_t + num_c + num_v + 1):
        if id <= num_u:
            HeterHyper_id_vec_list.append(HeterHyper_w2v_model.wv[str(id)])
        HeterHyper_all_id_vec_list.append(HeterHyper_w2v_model.wv[str(id)])

    # ACC, Recall, F1 = count_o2n_Acc_Recall(K, HeterHyper_id_vec_list, num_u, friendship_new_df, friendship_old_df)
    ACC, Recall, F1 = 1,1,1

    return ACC, Recall, F1, HeterHyper_id_vec_list, HeterHyper_all_id_vec_list

if __name__ == '__main__':
    cities = ["Istanbul", "Jakarta", "KualaLampur", "NYC", "SaoPaulo", "TKY"]
    # cities = ["NYC", "SaoPaulo"]

    for city in cities:
        print("*" * 20, "开始处理" + city + "的数据", "*" * 20)
        # get data
        print("-" * 10, "读取" + city + "原数据", "-" * 10)
        users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df = get_mat_dataset(city)

        # 重新编排id
        print("-" * 10, "重新编排id", "-" * 10)
        offset_DIC = remakeID_checkins(checkins_df, friendship_new_df, friendship_old_df)

        # 构建异构超图
        print("-" * 10, "构建异构超图", "-" * 10)
        num_u, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c, social_G, HeterHyper_G = get_Graph(friendship_old_df, checkins_df, offset_DIC)
        num_all_nodes = num_u + num_t + num_v + num_c

        # """
        # deepWalk
        # """
        # # deepwalk_S
        # print("-" * 10, "deepwalk_S", "-" * 10)
        # ACC, Recall, F1, deepwalk_S_id_vec_list = main_of_deepWalk(social_G, num_u, friendship_new_df, friendship_old_df)
        # print("deepwalk_S： ACC:", ACC, " Recall:", Recall, " F1:", F1)
        #
        # # deepwalk_SM
        # print("-" * 10, "deepwalk_SM", "-" * 10)
        # ACC, Recall, F1, deepwalk_SM_id_vec_list = main_of_deepWalk(HeterHyper_G, num_u, friendship_new_df, friendship_old_df)
        # print("deepwalk_SM： ACC:", ACC, " Recall:", Recall, " F1:", F1)
        #
        # """
        # node2vec
        # """
        # # node2vec_S
        # print("-" * 10, "node2vec_S", "-" * 10)
        # node2vec_S_id_vec_list, ACC, Recall, F1 = main_of_node2vec(social_G, num_u, friendship_new_df, friendship_old_df)
        # print("node2vec_S： ACC:", ACC, " Recall:", Recall, " F1:", F1)
        #
        # # node2vec_SM
        # print("-" * 10, "node2vec_SM", "-" * 10)
        # node2vec_SM_id_vec_list, ACC, Recall, F1 = main_of_node2vec(HeterHyper_G, num_u, friendship_new_df, friendship_old_df)
        # print("node2vec_SM： ACC:", ACC, " Recall:", Recall, " F1:", F1)
        #
        # """
        # LINE
        # """
        # print("-" * 10, "node2vec_S", "-" * 10)
        # LINE_S_id_vec_list, ACC, Recall, F1 = main_of_LINE(nx.DiGraph(social_G), num_u, friendship_new_df,
        #                                                    friendship_old_df)
        # print("LINE_S： ACC:", ACC, " Recall:", Recall, " F1:", F1)
        #
        # print("-" * 10, "node2vec_SM", "-" * 10)
        # LINE_SM_id_vec_list, ACC, Recall, F1 = main_of_LINE(nx.DiGraph(HeterHyper_G), num_u, friendship_new_df,
        #                                                    friendship_old_df)
        # print("LINE_SM： ACC:", ACC, " Recall:", Recall, " F1:", F1)


        """
        异构超图上超图跳跃游走计算
        """
        ACC, Recall, F1, HeterHyper_id_vec_list, HeterHyper_all_id_vec_list = main_of_hyperHeter(num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c,
                                             friendship_new_df, friendship_old_df)
        print("超图跳跃游走结果： ACC:", ACC, " Recall:", Recall, " F1:", F1)


        # # 存储下异构超图得到的vec
        # data_heterHyper ={
        #     "friendship_new_df" : friendship_new_df,
        #     "friendship_old_df" : friendship_old_df,
        #     "checkins_df" : checkins_df,
        #     "deepwalk_S_id_vec_list": deepwalk_S_id_vec_list,
        #     "deepwalk_SM_id_vec_list": deepwalk_SM_id_vec_list,
        #     "node2vec_S_id_vec_list": node2vec_S_id_vec_list,
        #     "node2vec_SM_id_vec_list": node2vec_SM_id_vec_list,
        #     "LINE_S_id_vec_list": LINE_S_id_vec_list,
        #     "LINE_SM_id_vec_list": LINE_SM_id_vec_list,
        #     "HeterHyper_id_vec_list": HeterHyper_id_vec_list,
        #     "HeterHyper_all_id_vec_list": HeterHyper_all_id_vec_list,
        #     "bipG_t": bipG_t,
        #     "bipG_v": bipG_v,
        #     "bipG_c": bipG_c
        # }
        # file = open('data_proces/'+ city +'data_heterHyper_d1_64.pickle', 'wb')
        # pickle.dump(data_heterHyper, file)
        # file.close()

        # 存储d1=64下异构超图得到的vec
        data_heterHyper = {
            "HeterHyper_id_vec_list": HeterHyper_id_vec_list,
            "HeterHyper_all_id_vec_list": HeterHyper_all_id_vec_list,
        }
        file = open('data_proces/' + city + 'data_heterHyper_d1_64.pickle', 'wb')
        pickle.dump(data_heterHyper, file)
        file.close()
