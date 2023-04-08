"""
超二部图主函数
废弃状态，超二部图部分在HeterHyperGraph中有
时间,类型,地点的普通二部图游走然后计算结果这里独有
"""

import os
import pandas as pd
from scipy.io import loadmat
import networkx as nx
from walks.HyperBipDeepWalk import HyperBipDeepWalk
from walks.DeepWalk import DeepWalk
from gensim.models import Word2Vec
from time import time
import numpy as np
import heapq
import torch
import random
import pickle

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

    # 规范社交链接的数据，有friendship中有的点没有签到数据
    set_check_usr = set(checkins_df['uid'])
    delete_ind_friends = []
    for ii in range(len(friendship_new_df)):
        if friendship_new_df['uid1'][ii] not in set_check_usr  or  friendship_new_df['uid2'][ii] not in set_check_usr:
            delete_ind_friends.append(ii)
    friendship_new_df = friendship_new_df.drop(delete_ind_friends, axis=0)

    delete_ind_friends = []
    for ii in range(len(friendship_old_df)):
        if friendship_old_df['uid1'][ii] not in set_check_usr or friendship_old_df['uid2'][ii] not in set_check_usr:
            delete_ind_friends.append(ii)
    friendship_old_df = friendship_old_df.drop(delete_ind_friends, axis=0)

    return users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df

# 对checkins的id重新赋值
def remakeID_checkins(checkins_df):
    user_N_O_list = []
    time_N_O_list = []
    vid_N_O_list = []
    category_N_O_list = []
    user_O_N_dic = {}
    time_O_N_dic = {}
    vid_O_N_dic = {}
    category_O_N_dic = {}

    num_usr = len(set(checkins_df['uid']))
    num_time = len(set(checkins_df['time']))
    num_vid = len(set(checkins_df['vid']))
    num_category = len(set(checkins_df['category']))

    for i in range(num_usr + 1): user_N_O_list.append(0)
    for i in range(num_time + 1): time_N_O_list.append(0)
    for i in range(num_vid + 1): vid_N_O_list.append(0)
    for i in range(num_category + 1): category_N_O_list.append(0)

    #偏移已考虑id从1开始
    timeID_offset = num_usr + 1
    vid_offset = timeID_offset + num_time
    category_offset = vid_offset + num_vid
    offset_DIC = {'time':timeID_offset, 'vid':vid_offset, 'category':category_offset}

    count_ID = 0
    last_ID = -1
    checkins_df.sort_values(by="uid", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        tmp_ID = checkins_df['uid'][ii]
        if tmp_ID != last_ID:
            count_ID += 1
            last_ID = tmp_ID
            user_N_O_list[count_ID] = tmp_ID
            user_O_N_dic[tmp_ID] = count_ID
            checkins_df['uid'][ii] = count_ID
        else:
            checkins_df['uid'][ii] = count_ID

    count_ID = -1
    last_ID = -1
    checkins_df.sort_values(by="time", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        tmp_ID = checkins_df['time'][ii]
        if tmp_ID != last_ID:
            count_ID += 1
            last_ID = tmp_ID
            time_N_O_list[count_ID] = tmp_ID
            time_O_N_dic[tmp_ID] = count_ID + timeID_offset #需要加上偏移
            checkins_df['time'][ii] = count_ID + timeID_offset
        else:
            checkins_df['time'][ii] = count_ID + timeID_offset

    count_ID = -1
    last_ID = -1
    checkins_df.sort_values(by="vid", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))
    for ii in range(len(checkins_df)):
        tmp_ID = checkins_df['vid'][ii]
        if tmp_ID != last_ID:
            count_ID += 1
            last_ID = tmp_ID
            vid_N_O_list[count_ID] = tmp_ID
            vid_O_N_dic[tmp_ID] = count_ID + vid_offset  # 需要加上偏移
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
            category_N_O_list[count_ID] = tmp_ID
            category_O_N_dic[tmp_ID] = count_ID + category_offset  # 需要加上偏移
            checkins_df['category'][ii] = count_ID + category_offset
        else:
            checkins_df['category'][ii] = count_ID + category_offset
    checkins_df.sort_values(by="uid", inplace=True, ascending=True)
    checkins_df.index = range(0, len(checkins_df))

    All_N_O_list_list = [user_N_O_list, time_N_O_list, vid_N_O_list, category_N_O_list]
    All_O_N_list_list = [user_O_N_dic, time_O_N_dic, vid_O_N_dic, category_O_N_dic]
    return   All_N_O_list_list, All_O_N_list_list, offset_DIC

# get hyperBipGraph
def get_hyperBipGraph(checkins_df, offset_DIC):
    # 个数
    num_usr = len(set(checkins_df['uid']))
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

    return num_usr, num_v, num_t, num_c, bipG_t, bipG_v,  bipG_c

# 余弦距离
def countCos(x,y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# 计算accuracy，recall计算
def count_Acc_Recall(K, userVec_Bip_dic, num_user, friendship_df, userId_old2new):
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
        link_labels_M[userId_old2new[old_uid1]][userId_old2new[old_uid2]] = 1
        link_labels_M[userId_old2new[old_uid2]][userId_old2new[old_uid1]] = 1

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

# 在超二部图上学习并输出acc和recall
def learn_hyper_bipG(args, kwargs, friendship_df, userId_old2new, num_u, num_t, num_v, num_c, bipG_v, bipG_t, bipG_c):
    # 超二部图上游走
    start = time()
    print("-" * 10, "超二部图上游走", "-" * 10)
    HyperBipDeepWalk_mode = HyperBipDeepWalk(args, num_u, num_t, num_v, num_c, bipG_v, bipG_t, bipG_c)
    hyperBipWalks = HyperBipDeepWalk_mode.HBdeepWalk()
    stop = time()
    print("运行时间/s：", str(stop - start) + "s")
    print("运行时间/min：", str((stop - start) / 60) + "min")
    print("运行时间/h：", str((stop - start) / (60 * 60)) + "H")

    # 将游走序列转换为字符
    for i in range(len(hyperBipWalks)):
        for j in range(len(hyperBipWalks[i])):
            hyperBipWalks[i][j] = str(hyperBipWalks[i][j])

    print("-" * 10, "超二部图word2vec开始训练", "-" * 10)
    start = time()
    kwargs["sentences"] = hyperBipWalks
    # 直接使用现有的word2vec工具包
    hyperBip_w2v_model = Word2Vec(**kwargs)
    stop = time()
    print("运行时间/s：", str(stop - start) + "s")
    print("运行时间/min：", str((stop - start) / 60) + "min")
    print("运行时间/h：", str((stop - start) / (60 * 60)) + "H")

    # 计算ACC和recall
    print("-" * 10, "超二部图计算ACC和recall", "-" * 10)
    K = 10
    hyperBip_id_vec_list = [[]]  # 考虑id从1开始
    for uid in range(1, num_u + 1):
        hyperBip_id_vec_list.append(hyperBip_w2v_model.wv[str(uid)])

    hp_acc, hp_recall = count_Acc_Recall(K, hyperBip_id_vec_list, num_u, friendship_df, userId_old2new)

    return hp_acc, hp_recall, hyperBip_id_vec_list

# 学习某一个单独类型的二部图并输出acc和recall
def learn_one_bipG(defName, args, kwargs, friendship_df, userId_old2new, bipG_one, num_u):
    start = time()
    print("-" * 10, defName + "上游走", "-" * 10)
    deepWalk_model = DeepWalk(args, bipG_one)
    loc_walks = deepWalk_model.HBdeepWalk()
    stop = time()
    print("运行时间/s：", str(stop - start) + "s")
    print("运行时间/min：", str((stop - start) / 60) + "min")
    print("运行时间/h：", str((stop - start) / (60 * 60)) + "H")

    for i in range(len(loc_walks)):
        for j in range(len(loc_walks[i])):
            loc_walks[i][j] = str(loc_walks[i][j])

    # word2vec学习

    print("-" * 10, defName + "word2vec开始训练", "-" * 10)
    start = time()
    kwargs["sentences"] = loc_walks
    # 直接使用现有的word2vec工具包
    locBip_w2v_model = Word2Vec(**kwargs)
    stop = time()
    print("运行时间/s：", str(stop - start) + "s")
    print("运行时间/min：", str((stop - start) / 60) + "min")
    print("运行时间/h：", str((stop - start) / (60 * 60)) + "H")

    # 计算ACC和recall
    print("-" * 10, defName + "计算ACC和recall", "-" * 10)
    K = 10
    locBip_id_vec_list = [[]]  # 考虑id从1开始
    for uid in range(1, num_u + 1):
        locBip_id_vec_list.append(locBip_w2v_model.wv[str(uid)])

    return count_Acc_Recall(K, locBip_id_vec_list, num_u, friendship_df, userId_old2new)


if __name__ == '__main__':
    cities = ["Istanbul", "Jakarta", "KualaLampur", "NYC", "SaoPaulo", "TKY"]

    for city in cities:
        print("*" * 20, city, "*" * 20)
        # get data
        users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df = get_mat_dataset(city)

        # 重新编排id
        print("-" * 10, "重新编排id", "-" * 10)
        All_N_O_list_list, All_O_N_list_list, offset_DIC = remakeID_checkins(checkins_df)

        # 构建超二部图
        print("-" * 10, "构建超二部图", "-" * 10)
        num_u, num_v, num_t, num_c, bipG_t, bipG_v, bipG_c = get_hyperBipGraph(checkins_df, offset_DIC)

        # 一些参数
        args = {'r': 10, 'l': 80}

        kwargs = {}
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = 128
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = 3
        kwargs["window"] = 5
        kwargs["epochs"] = 5

        hp_acc, hp_recall, hyperBip_id_vec_list = learn_hyper_bipG(args, kwargs, friendship_old_df,
                                                                   All_O_N_list_list[0], num_u, num_t, num_v, num_c,
                                                                   bipG_v, bipG_t, bipG_c)
        # t_acc, t_recall = learn_one_bipG("时间二部图", args, kwargs, friendship_old_df, All_O_N_list_list[0], bipG_t, num_u)
        # v_acc, v_recall = learn_one_bipG("地点二部图", args, kwargs, friendship_old_df, All_O_N_list_list[0], bipG_v, num_u)
        # c_acc, c_recall = learn_one_bipG("活动类型二部图", args, kwargs, friendship_old_df, All_O_N_list_list[0], bipG_c,
        #                                  num_u)
        print("  类型  ", "  ACC", "  Recall")
        print("超二部图：", hp_acc, hp_recall)
        # print("时间二部图：", t_acc, t_recall)
        # print("地点二部图：", v_acc, v_recall)
        # print("活动类型二部图：", c_acc, c_recall)

        # data_heterHyper = {
        #     "friendship_new_df": friendship_new_df,
        #     "friendship_old_df": friendship_old_df,
        #     "checkins_df": checkins_df,
        #     "All_N_O_list_list": All_N_O_list_list,
        #     "All_O_N_list_list": All_O_N_list_list,
        #     "HeterHyper_id_vec_list": hyperBip_id_vec_list
        # }
        # file = open('data_proces/' + city + 'data_hyperBip_Vec.pickle', 'wb')
        # pickle.dump(data_heterHyper, file)
        # file.close()
