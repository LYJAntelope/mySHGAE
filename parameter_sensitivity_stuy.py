"""
跑参数敏感实验的函数
"""
import os
import pandas as pd
from scipy.io import loadmat
import networkx as nx
from walks.SHJ_walk import SHJ_walk
from gensim.models import Word2Vec
import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, to_undirected
import heapq
import random
from GATmodel2 import GATNet
from torch_geometric.data import Data
import params
from importlib import reload
import pickle

def load_params():
    # parameter of SHJ-walk
    args = params.args
    # parameter of skip-gram
    kwargs = params.kwargs
    # TOP-K
    top_K = params.top_K
    # Number of negative sampling candidates
    negative_candidates = params.negative_candidates
    # GAT_dimension
    GAT_dimension = params.GAT_dimension
    # GAT_num
    head_Num = params.head_Num

    return args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num

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

    # id starts from 0
    for ii in range(len(friendship_old_df)):
        friendship_old_df['uid1'][ii] -= 1
        friendship_old_df['uid2'][ii] -= 1
    for ii in range(len(friendship_new_df)):
        friendship_new_df['uid1'][ii] -= 1
        friendship_new_df['uid2'][ii] -= 1
    for ii in range(len(checkins_df)):
        checkins_df['uid'][ii] -= 1

    return users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df

def remakeID_checkins(checkins_df, friendship_new_df, friendship_old_df):
    set_u = set(checkins_df['uid']) | set(friendship_new_df['uid1']) | set(friendship_new_df['uid2']) | \
            set(friendship_old_df['uid1']) | set(friendship_old_df['uid2'])
    num_usr = len(set_u)
    num_time = len(set(checkins_df['time']))
    num_vid = len(set(checkins_df['vid']))
    num_category = len(set(checkins_df['category']))

    # id starts from 0
    timeID_offset = num_usr
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

def get_data(city):
    with open('data_proces/' + city + 'data_heterHyper.pickle', 'rb') as file:
        data_heterHyper = pickle.load(file)

    HeterHyper_id_vec_list = data_heterHyper['HeterHyper_id_vec_list']
    HeterHyper_all_id_vec_list = data_heterHyper['HeterHyper_all_id_vec_list']

    # 为了方便GAT训练，这里把用户的id都-1，让id从0开始
    HeterHyper_id_vec_list.pop(0)
    HeterHyper_all_id_vec_list.pop(0)

    return HeterHyper_id_vec_list, HeterHyper_all_id_vec_list

# get Graph
def get_Graph(friendship_old_df, checkins_df, offset_DIC):
    num_usr = offset_DIC["time"]
    num_t = len(set(checkins_df['time']))
    num_v = len(set(checkins_df['vid']))
    num_c = len(set(checkins_df['category']))

    # 计Calculate check-in frequency
    usr_t_pc_list = [[0 for _t in range(num_t)] for _ in range(num_usr + 1)]
    usr_v_pc_list = [[0 for _v in range(num_v)] for _ in range(num_usr + 1)]
    usr_c_pc_list = [[0 for _c in range(num_c)] for _ in range(num_usr + 1)]

    for uid, time, vid, cate in zip(checkins_df['uid'], checkins_df['time'], checkins_df['vid'], checkins_df['category']):
        usr_t_pc_list[uid][time - offset_DIC['time']] += 1
        usr_v_pc_list[uid][vid - offset_DIC['vid']] += 1
        usr_c_pc_list[uid][cate - offset_DIC['category']] += 1

    bipG_t = nx.Graph()
    bipG_v = nx.Graph()
    bipG_c = nx.Graph()
    for uid in range(0,num_usr):
        for tid in range(num_t):
            if usr_t_pc_list[uid][tid]>0:
                bipG_t.add_edge(uid, tid + offset_DIC['time'], weight=usr_t_pc_list[uid][tid])
        for vid in range(num_v):
            if usr_v_pc_list[uid][vid] > 0:
                bipG_v.add_edge(uid, vid + offset_DIC['vid'], weight=usr_v_pc_list[uid][vid])
        for cid in range(num_c):
            if usr_c_pc_list[uid][cid] > 0:
                bipG_c.add_edge(uid, cid + offset_DIC['category'], weight=usr_c_pc_list[uid][cid])

    social_G = nx.Graph()
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        social_G.add_edge(uid1, uid2, weight=1)

    return num_usr, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c, social_G

# Split test set and validation set
def train_test_split_edges_m(data, num_u, val_ratio, val_neg_ratio):
    """
    val_ratio:验证集比例
    test_ratio：测试集比例
    val_neg_ratio:验证集负样本比例
    test_neg_candidates_num：测试集负样本数量（SOTA为50）
    """
    num_nodes = data.num_nodes
    # num_u = data.num_nodes
    row, col = data.edge_index_uid
    row_tvc, col_tvc = data.edge_index_uid_tvc  # 和tvc的链接
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(val_ratio * row.size(0)))

    # Positive edges.
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    # r, c = row, col
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v:], col[n_v:]
    # r, c = row, col
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index_uWithTvc = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_uWithTvc = to_undirected(data.train_pos_edge_index_uWithTvc)
    data.train_pos_edge_index_uWithTvc = torch.cat([data.train_pos_edge_index_uWithTvc,
                                                    torch.stack([row_tvc, col_tvc], dim=0)], 1)

    data.train_pos_edge_index_tvc = torch.stack([row_tvc, col_tvc], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    # for i in range(num_u):
    #     for j in range(num_u):
    #         neg_adj_mask[i][j] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    data.train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)  # 训练负采样的样本 是全部的负样本

    neg_adj_mask = torch.ones(num_u, num_u, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    row, col = neg_row[:n_v * val_neg_ratio], neg_col[:n_v * val_neg_ratio]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    return data

def count_link_lables(num_user, friendship_new_df, friendship_old_df, neg_num):
    # 社交网络邻接矩阵 考虑id从1开始
    link_new_labels_M = np.zeros((num_user, num_user))
    for uid1, uid2 in zip(friendship_new_df['uid1'], friendship_new_df['uid2']):
        link_new_labels_M[uid1][uid2] = 1
        link_new_labels_M[uid2][uid1] = 1
    # 旧社交网络邻接矩阵
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        link_new_labels_M[uid1][uid2] = 2  # 原new的社交链接中实际上是old+新增的社交链接，所以这里将新增社交链接中的旧链接去掉，防止影响后面预测（特别是召回率分母）
        link_new_labels_M[uid2][uid1] = 2
    # 进行负采样，只让采样上的为3，其他为0
    for link_u_new_l in link_new_labels_M:
        tensor_link_u_new_l = torch.tensor(link_u_new_l)
        pre_ind = torch.eq(tensor_link_u_new_l, 0).nonzero().numpy().tolist()
        for preI in random.sample(pre_ind, neg_num):  # 负样本随机抽取neg_num(50)个
            link_u_new_l[preI[0]] = 3
    return link_new_labels_M

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# train of HMT-GAT训练函数
def train(data, model, optimizer):
    model.train()

    # 进行节点负采样 正负样本1:nn
    nn = 1
    # 自己进行采样
    perm = torch.randperm(data.train_pos_edge_index_uWithTvc.size(1) * nn)
    neg_row, neg_col = data.train_neg_edge_index[0][perm], data.train_neg_edge_index[1][perm]
    neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

    optimizer.zero_grad()
    # 用异构超图节点+用户节点编码
    z = model.encode(data.x, data.train_pos_edge_index_uWithTvc)
    # 用用户节点之间的链接关系进行解码
    link_logits = model.decode(z, data.train_pos_edge_index_uWithTvc, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index_uWithTvc, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits.cpu(), link_labels)
    loss.backward()
    optimizer.step()

    return loss

# Define a single epoch verification process
@torch.no_grad()
def val(data, model):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index_uWithTvc)

    prefix = 'val'
    pos_edge_index = data[f'{prefix}_pos_edge_index']
    neg_edge_index = data[f'{prefix}_neg_edge_index']
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    # link_probs = model.decode_val(z, pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    return roc_auc_score(link_labels.cpu(), link_probs.cpu())

# Calculate cosine similarity
def cosinematrix(A):
    prod = torch.mm(A, A.t())  # 分子
    norm = torch.norm(A, p=2, dim=1).unsqueeze(0)  # 分母
    cos = prod.div(torch.mm(norm.t(), norm))
    for i in range(len(cos)):
        cos[i][i] = 0
    return cos

# calculate P@K, R@K, F1@K
def count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M):
    P_K_list = []
    R_K_list = []
    # 计算精确度和召回率

    # 对每个user进行计算  这是以前的计算方法，目的是仅采样一部分负样本
    prob_adj = prob_adj.detach().numpy()
    for link_u_p, link_u_new_l in zip(prob_adj, link_new_labels_M):
        # 只对正样本和抽取的负样本进行预测
        tensor_link_u_new_l = torch.tensor(link_u_new_l)
        pre_ind = torch.eq(tensor_link_u_new_l, 1).nonzero().numpy().tolist()
        for preI in pre_ind:
            link_u_p[preI[0]] += 1000  # 正样本全参与预测
        pre_ind = torch.eq(tensor_link_u_new_l, 3).nonzero().numpy().tolist()
        for preI in pre_ind:
            link_u_p[preI[0]] += 1000  # 抽取的负样本参与预测

        link_u_p = list(link_u_p)  # ndarray 转 list
        link_u_l = list(link_u_new_l)
        num_1 = list(link_u_l).count(1)  # 该用户新增的链接个数
        if (num_1 < 1):
            continue
        max_num_index_list = list(map(link_u_p.index, heapq.nlargest(K, link_u_p)))
        N_u_true = 0
        for index in max_num_index_list:
            if link_u_l[index] == 1:
                N_u_true += 1

        P_K_list.append(N_u_true / K)
        R_K_list.append(N_u_true / num_1)

    P_ = sum(P_K_list) / len(P_K_list)
    R_ = sum(R_K_list) / len(R_K_list)
    if P_ + R_ != 0:
        F1_ = 2 * (P_ * R_) / (P_ + R_)
    else:
        F1_ = 0

    return P_, R_, F1_

def main_of_hyperHeter(num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c):
    SHJ_mode = SHJ_walk(args, num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c,)
    HeterHyperWalks = SHJ_mode.HBdeepWalk()

    for i in range(len(HeterHyperWalks)):
        for j in range(len(HeterHyperWalks[i])):
            HeterHyperWalks[i][j] = str(HeterHyperWalks[i][j])
    kwargs["sentences"] = HeterHyperWalks
    HeterHyper_w2v_model = Word2Vec(**kwargs)

    HeterHyper_all_id_vec_list = []
    for id in range(0, num_u + num_t + num_c + num_v):
        HeterHyper_all_id_vec_list.append(HeterHyper_w2v_model.wv[str(id)])

    return HeterHyper_all_id_vec_list

def main_of_SHGAE(num_u, HeterHyper_all_id_vec_list, friendship_old_df, bipG_t, bipG_v, bipG_c):
    # 构建训练的data
    x_curLink = torch.tensor(np.array(HeterHyper_all_id_vec_list), dtype=torch.float32)  # 先转array再转tensor更快
    edge_index_uid = [[], []]
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        edge_index_uid[0].append(uid1)
        edge_index_uid[1].append(uid2)
    edge_index_uid = torch.tensor(edge_index_uid, dtype=torch.long)

    # 把tvc的边放进去
    edge_index_uid_tvc = [[], []]
    for edge_t in bipG_t.edges():
        # (a,b) a为uid，b为tvc，消息传播方向为 a-b
        edge_index_uid_tvc[0].append(min(edge_t[0], edge_t[1]))
        edge_index_uid_tvc[1].append(max(edge_t[0], edge_t[1]))
    for edge_v in bipG_v.edges():
        edge_index_uid_tvc[0].append(min(edge_v[0], edge_v[1]))
        edge_index_uid_tvc[1].append(max(edge_v[0], edge_v[1]))
    for edge_c in bipG_c.edges():
        edge_index_uid_tvc[0].append(min(edge_c[0], edge_c[1]))
        edge_index_uid_tvc[1].append(max(edge_c[0], edge_c[1]))
    edge_index_uid_tvc = torch.tensor(edge_index_uid_tvc, dtype=torch.long)

    data = Data(x=x_curLink, edge_index_uid=edge_index_uid, edge_index_uid_tvc=edge_index_uid_tvc)  # 构建data

    # 网络数据放入GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU
    # ground_truth_edge_index = data.edge_index_uid.to(device)  # 数据放入GPU
    data.edge_index_uid.to(device)  # 数据放入GPU
    data.edge_index_uid_tvc.to(device)  # 数据放入GPU
    # 样本负采样,测试集验证集分裂,按学习难度分化数据集
    data = train_test_split_edges_m(data, num_u, val_ratio=0.05, val_neg_ratio=5)
    # data = train_Diff_split_edges(data, node_diffASC_list, NUM_diff_leve)  # 根据学习难度，对边数据进行分级
    data = data.to(device)  # 将数据放入device

    # 构建网络
    num_node_features = len(data.x[0])
    model = GATNet(num_node_features, GAT_dimension, headNum = head_Num).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    x_epoch = []
    y_loss = []
    y_val_auc = []
    num_fit = 10
    fit_x = np.array([i + 1 for i in range(num_fit)])
    for epoch in range(1, 3000 + 1):
        loss = train(data, model, optimizer)
        val_auc = val(data, model)
        # if epoch % 5 == 0:
        #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')
        x_epoch.append(epoch)
        y_loss.append(float(loss))
        y_val_auc.append(float(val_auc))

        # 验证集用来控制何时结束训练,拟合直线斜率小于等于0则停止
        if epoch > 500 and len(y_val_auc) > num_fit:
            fit_y = np.array(y_val_auc[-num_fit:])
            f1 = np.polyfit(fit_x, fit_y, 1)
            if f1[0] <= 0:
                break

    zzz = model.encode(data.x, data.train_pos_edge_index_uWithTvc)
    zzz = zzz[:num_u]
    prob_adj = cosinematrix(zzz)
    return prob_adj

def main_of_one_group_params():
    P_K_city = []
    R_K_city = []
    F1_K_city = []
    for city in cities:
        # print("*" * 20, "Loading " + city + " data", "*" * 20)
        # get data
        users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df = get_mat_dataset(city)

        # remake id
        # print("-" * 10, "Remaking id", "-" * 10)
        offset_DIC = remakeID_checkins(checkins_df, friendship_new_df, friendship_old_df)

        # heterogeneous hypergraphs
        # print("-" * 10, "Building heterogeneous hypergraphs", "-" * 10)
        num_u, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c, social_G = get_Graph(friendship_old_df, checkins_df,
                                                                                 offset_DIC)

        """
        SHJ-walk
        """
        # print("-" * 10, "SHJ-walk ing", "-" * 10)
        HeterHyper_all_id_vec_list = main_of_hyperHeter(num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c)

        """
        HMT-GAT
        """
        # print("-" * 10, "HMT-GAT ing", "-" * 10)
        prob_adj = main_of_SHGAE(num_u, HeterHyper_all_id_vec_list, friendship_old_df, bipG_t, bipG_v, bipG_c)

        # 重复10次抽样预测
        pp_l = []
        rr_l = []
        ff_l = []
        for i in range(10):
            link_new_labels_M = count_link_lables(num_u, friendship_new_df, friendship_old_df, negative_candidates)
            P, R, F1 = count_o2n_Acc_Recall(top_K, prob_adj.cpu(), link_new_labels_M)
            pp_l.append(P)
            rr_l.append(R)
            ff_l.append(F1)
        P_K = sum(pp_l)/len(pp_l)
        R_K = sum(rr_l) / len(rr_l)
        F1_ = sum(ff_l) / len(ff_l)
        # print("P_K:", P_K, " R_K:", R_K, " F1:", F1_)
        P_K_city.append(P_K)
        R_K_city.append(R_K)
        F1_K_city.append(F1_)

    return P_K_city, R_K_city, F1_K_city

# GAT进行敏感性实验时只需要游走一次
def main_of_one_group_params_GAT(HeterHyper_all_id_vec_list):
    """
    HMT-GAT
    """

    prob_adj = main_of_SHGAE(num_u, HeterHyper_all_id_vec_list, friendship_old_df, bipG_t, bipG_v, bipG_c)
    # 重复10次实验
    pp_l = []
    rr_l = []
    ff_l = []
    for i in range(10):
        link_new_labels_M = count_link_lables(num_u, friendship_new_df, friendship_old_df, negative_candidates)
        P, R, F1 = count_o2n_Acc_Recall(top_K, prob_adj.cpu(), link_new_labels_M)
        # print(GAT_dimension, i, P, R, F1)
        pp_l.append(P)
        rr_l.append(R)
        ff_l.append(F1)
    P_K = sum(pp_l) / len(pp_l)
    R_K = sum(rr_l) / len(rr_l)
    F1_ = sum(ff_l) / len(ff_l)
    # P_K, R_K, F1_ = count_o2n_Acc_Recall(top_K, prob_adj.cpu(), link_new_labels_M)

    # print("P_K:", P_K, " R_K:", R_K, " F1:", F1_)

    return P_K, R_K, F1_

if __name__ == '__main__':
    # cities = ["NYC", "TKY", "Istanbul", "Jakarta", "KualaLampur", "SaoPaulo"]
    cities = ["Jakarta"]

    # 首先对 阿尔法进行敏感性实验
    args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num = load_params()
    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    P_list = [[] for _ in range(len(cities))]
    R_list = [[] for _ in range(len(cities))]
    F1_list = [[] for _ in range(len(cities))]
    for tmp_alpha in alpha_list:
        args['alpha'] = tmp_alpha
        P_K_city, R_K_city, F1_K_city = main_of_one_group_params()
        for i in range(len(cities)):
            P_list[i].append(P_K_city[i])
            R_list[i].append(R_K_city[i])
            F1_list[i].append(F1_K_city[i])
    print("阿尔法敏感性实验:")
    print("P_liat:", P_list)
    # print("R_liat:", R_list)
    # print("F1_liat:", F1_list)

    # 对SHJ-walk的 vec维度进行实验
    reload(params)
    args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num = load_params()
    d_1_list = [8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    P_list = [[] for _ in range(len(cities))]
    R_list = [[] for _ in range(len(cities))]
    F1_list = [[] for _ in range(len(cities))]
    for d_1 in d_1_list:
        kwargs["vector_size"] = d_1
        P_K_city, R_K_city, F1_K_city = main_of_one_group_params()
        for i in range(len(cities)):
            P_list[i].append(P_K_city[i])
            R_list[i].append(R_K_city[i])
            F1_list[i].append(F1_K_city[i])
    print("SHJ-walk的vec维度敏感性实验:")
    print("P_liat:", P_list)
    # print("R_liat:", R_list)
    # print("F1_liat:", F1_list)

    """
    GAT进行敏感性实验时只需要游走一次
    """
    print("*" * 20, "GAT进行敏感性实验时只需要游走一次", "*" * 20)
    reload(params)
    args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num = load_params()
    for city in cities:
        print("*" * 20, "计算：" + city, "*" * 20)
        users_df, venues_df, checkins_df, friendship_new_df, friendship_old_df = get_mat_dataset(city)
        offset_DIC = remakeID_checkins(checkins_df, friendship_new_df, friendship_old_df)
        num_u, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c, social_G = get_Graph(friendship_old_df, checkins_df,
                                                                                 offset_DIC)
        # HeterHyper_all_id_vec_list = main_of_hyperHeter(num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c)

        HeterHyper_id_vec_list, HeterHyper_all_id_vec_list = get_data(city)
        num_u = len(HeterHyper_id_vec_list)

        # 对HMT-GAT的vec维度进行实验
        reload(params)
        args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num = load_params()
        d_2_list = [8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
        P_list = []
        R_list = []
        F1_list = []
        for d_2 in d_2_list:
            GAT_dimension = d_2
            P_K_city, R_K_city, F1_K_city = main_of_one_group_params_GAT(HeterHyper_all_id_vec_list)
            P_list.append(P_K_city)
            R_list.append(R_K_city)
            F1_list.append(F1_K_city)
        print("HMT-GAT的vec维度敏感性实验:")
        print("P_list:", P_list)
        # print("R_list:", R_list)
        # print("F1_list:", F1_list)

        # 对注意力头个数进行实验
        reload(params)
        args, kwargs, top_K, negative_candidates, GAT_dimension, head_Num = load_params()
        head_list = [1, 2, 3, 4, 5, 6, 7, 8]
        # city_list
        P_list = []
        R_list = []
        F1_list = []
        for head in head_list:
            head_Num = head
            P_K_city, R_K_city, F1_K_city = main_of_one_group_params_GAT(HeterHyper_all_id_vec_list)
            P_list.append(P_K_city)
            R_list.append(R_K_city)
            F1_list.append(F1_K_city)
        print("注意力头敏感性实验:")
        print("P_list:", P_list)
        # print("R_list:", R_list)
        # print("F1_list:", F1_list)


