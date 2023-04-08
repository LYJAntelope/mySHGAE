"""
Social Hypergraph AutoEncode Framework for Social Relationship Inference
"""
# 为了方便GAT训练，这里把用户的id都-1，让id从0开始
import pickle
import numpy as np
import math
from time import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling, to_undirected
import heapq
import random
from GATmodel import GATNet
from torch_geometric.data import Data

# 转换lable
def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# 读取数据
def get_data(city):
    with open('data_proces/' + city + 'data_heterHyper.pickle', 'rb') as file:
        data_heterHyper = pickle.load(file)

    friendship_new_df = data_heterHyper['friendship_new_df']
    friendship_old_df = data_heterHyper['friendship_old_df']
    checkins_df = data_heterHyper['checkins_df']
    deepwalk_S_id_vec_list = data_heterHyper['deepwalk_S_id_vec_list']
    # deepwalk_SM_id_vec_list = data_heterHyper['deepwalk_SM_id_vec_list']
    node2vec_S_id_vec_list = data_heterHyper['node2vec_S_id_vec_list']
    # node2vec_SM_id_vec_list = data_heterHyper['node2vec_SM_id_vec_list']
    LINE_S_id_vec_list = data_heterHyper['LINE_S_id_vec_list']
    # LINE_SM_id_vec_list = data_heterHyper['LINE_SM_id_vec_list']
    HeterHyper_id_vec_list = data_heterHyper['HeterHyper_id_vec_list']
    HeterHyper_all_id_vec_list = data_heterHyper['HeterHyper_all_id_vec_list']
    bipG_t = data_heterHyper["bipG_t"]
    bipG_v = data_heterHyper["bipG_v"]
    bipG_c = data_heterHyper["bipG_c"]

    # d1=64的特殊处理
    with open('data_proces/' + city + 'data_heterHyper_d1_64.pickle', 'rb') as file:
        data_heterHyper = pickle.load(file)

    HeterHyper_id_vec_list = data_heterHyper['HeterHyper_id_vec_list']
    HeterHyper_all_id_vec_list = data_heterHyper['HeterHyper_all_id_vec_list']
    # d1=64的特殊处理

    # LINE特殊处理
    num_u = len(HeterHyper_id_vec_list) - 1
    LINE_S_id_vec_list_new = []
    for i in range(1, num_u + 1):
        LINE_S_id_vec_list_new.append(LINE_S_id_vec_list[i])

    # 为了方便GAT训练，这里把用户的id都-1，让id从0开始
    deepwalk_S_id_vec_list.pop(0)
    # deepwalk_SM_id_vec_list.pop(0)
    node2vec_S_id_vec_list.pop(0)
    # node2vec_SM_id_vec_list.pop(0)
    # LINE_S_id_vec_list.pop(0)
    # LINE_SM_id_vec_list.pop(0)
    HeterHyper_id_vec_list.pop(0)
    HeterHyper_all_id_vec_list.pop(0)
    for ii in range(len(friendship_old_df)):
        friendship_old_df['uid1'][ii] -= 1
        friendship_old_df['uid2'][ii] -= 1
    for ii in range(len(friendship_new_df)):
        friendship_new_df['uid1'][ii] -= 1
        friendship_new_df['uid2'][ii] -= 1

    # return friendship_new_df, friendship_old_df, checkins_df, deepwalk_S_id_vec_list, deepwalk_SM_id_vec_list, \
    #        node2vec_S_id_vec_list, node2vec_SM_id_vec_list, LINE_S_id_vec_list, LINE_SM_id_vec_list, \
    #        HeterHyper_id_vec_list, HeterHyper_all_id_vec_list, bipG_t, bipG_v, bipG_c

    return friendship_new_df, friendship_old_df, checkins_df, deepwalk_S_id_vec_list, node2vec_S_id_vec_list, \
           LINE_S_id_vec_list_new, HeterHyper_id_vec_list, HeterHyper_all_id_vec_list, bipG_t, bipG_v, bipG_c

# GAT训练函数
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

# 定义单个epoch验证过程
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

# 计算余弦相似度
def cosinematrix(A):
    prod = torch.mm(A, A.t())  # 分子
    norm = torch.norm(A, p=2, dim=1).unsqueeze(0)  # 分母
    cos = prod.div(torch.mm(norm.t(), norm))
    for i in range(len(cos)):
        cos[i][i] = 0
    return cos

# 计算sigmoid
def countSigmoid(x_list):
    sig_x = []
    for x in x_list:
        sig_x.append(1 / (1 + math.exp(-x)))
    return sig_x

# 用来给数据分训练集验证集(测试集用新的社交链接）
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
    # 解码时需要带有tvc的图
    data.train_pos_edge_index_uWithTvc = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_uWithTvc = to_undirected(data.train_pos_edge_index_uWithTvc)
    data.train_pos_edge_index_uWithTvc = torch.cat([data.train_pos_edge_index_uWithTvc,
                                                    torch.stack([row_tvc, col_tvc], dim=0)], 1)

    data.train_pos_edge_index_tvc = torch.stack([row_tvc, col_tvc], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    data.train_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)  # 训练负采样的样本 是全部的负样本

    # neg_adj_mask[neg_row, neg_col] = 0
    # data.train_neg_adj_mask = neg_adj_mask

    # 原函数中对val的采样方式
    neg_adj_mask = torch.ones(num_u, num_u, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    row, col = neg_row[:n_v * val_neg_ratio], neg_col[:n_v * val_neg_ratio]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    """
    采样验证集的负样本 按照负样本的候选用户数量进行采样
    """
    #     val_user_list = list(set(data.val_pos_edge_index[0]) | set(data.val_pos_edge_index[1]))  # 先得到test中的用户list
    #     n_v_e = n_v * 4

    #     # 只有val中的节点未连接的为1
    #     neg_adj_mask = torch.zeros(num_u, num_u, dtype=torch.uint8)
    #     neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    #     for ii in range(len(val_user_list)):
    #         for jj in range(len(val_user_list)):
    #             if (jj <= ii): continue
    #             neg_adj_mask[val_user_list[ii]][val_user_list[jj]] = 1
    #             neg_adj_mask[val_user_list[jj]][val_user_list[ii]] = 1
    #     row, col = data.val_pos_edge_index
    #     neg_adj_mask[row, col] = 0  # val中有连接的为0
    #     neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    #     perm = torch.randperm(neg_row.size(0))
    #     neg_row, neg_col = neg_row[perm], neg_col[perm]

    #     # 随机采样，大概是每个用户取test_neg_candidates_num(50)个候选
    #     # row = neg_row[:n_v * val_neg_num]
    #     # col = neg_col[:n_v * val_neg_num]
    #     row = neg_row[:n_v]
    #     col = neg_col[:n_v]
    #     data.val_neg_edge_index = torch.stack([row, col], dim=0)
    #     print(num_u, n_v, n_v_e)
    return data

# 计算count_o2n_Acc_Recall中需要的link_new_labels_M，目的是为了保证计算各方法的结果时负采样一致
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

# 计算accuracy，recall计算,考虑旧的社交链接问题
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

    #     # 在全局中进行预测
    #     for link_u_p, link_u_new_l in zip(prob_adj, link_new_labels_M):
    #         link_u_l = list(link_u_new_l)
    #         num_1 = link_u_l.count(1)  # 计算一下该用户应该有的链接数
    #         if num_1 == 0:  # 该用户没有新的社交链接 则跳过
    #             continue
    #         # 只对正样本和抽取的负样本进行预测
    #         sort_enu = sorted(enumerate(link_u_p.detach().numpy()), key=lambda link_u_p: link_u_p[1], reverse=True)
    #         sort_index = [x[0] for x in sort_enu]  # 得到从大到小排序后的坐标

    #         tmpK = 0
    #         N_u_true = 0
    #         for index in sort_index:
    #             if link_u_new_l[index] == 2: #旧的社交链接不参与预测
    #                 continue
    #             if link_u_new_l[index] == 1:
    #                 N_u_true += 1
    #             tmpK += 1
    #             if tmpK == K:
    #                 break

    #         P_K_list.append(N_u_true / K)
    #         R_K_list.append(N_u_true / num_1)

    P_ = sum(P_K_list) / len(P_K_list)
    R_ = sum(R_K_list) / len(R_K_list)
    if P_ + R_ != 0:
        F1_ = 2 * (P_ * R_) / (P_ + R_)
    else:
        F1_ = 0

    return P_, R_, F1_

# 绘制曲线的函数
def drawLossVal(x, y, labelName, picName, x_lable, y_lable):
    fig, ax_loss = plt.subplots(figsize=(6, 3))
    # 绘制曲线
    ax_loss.plot(x, y, "r-", linewidth=2.0, label=labelName)
    # 添加样式
    ax_loss.set_title(picName, fontdict={"fontsize": 15})  # 标题
    ax_loss.set_xlabel(x_lable)  # 添加横轴标签
    ax_loss.set_ylabel(y_lable)  # 添加纵轴标签
    ax_loss.legend(loc="best", fontsize=16)  # 展示图例
    plt.grid(True)

# SHGAE的主函数
def main_of_SHGAE(num_u, num_epoch, HeterHyper_all_id_vec_list, friendship_old_df, link_new_labels_M, bipG_t, bipG_v, bipG_c):
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
        edge_index_uid_tvc[0].append(min(edge_t[0], edge_t[1]) - 1)
        edge_index_uid_tvc[1].append(max(edge_t[0], edge_t[1]) - 1)
    for edge_v in bipG_v.edges():
        edge_index_uid_tvc[0].append(min(edge_v[0], edge_v[1]) - 1)
        edge_index_uid_tvc[1].append(max(edge_v[0], edge_v[1]) - 1)
    for edge_c in bipG_c.edges():
        edge_index_uid_tvc[0].append(min(edge_c[0], edge_c[1]) - 1)
        edge_index_uid_tvc[1].append(max(edge_c[0], edge_c[1]) - 1)
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
    model = GATNet(num_node_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    x_epoch = []
    y_loss = []
    y_val_auc = []
    P_K_list = []
    num_fit = 10
    fit_x = np.array([i + 1 for i in range(num_fit)])
    for epoch in range(1, num_epoch + 1):
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

    # print("result：")
    # print("loss:", y_loss[-1])

    zzz = model.encode(data.x, data.train_pos_edge_index_uWithTvc)
    zzz = zzz[:num_u]
    prob_adj = cosinematrix(zzz)

    K = 10
    P_K, R_K, F1_ = count_o2n_Acc_Recall(K, prob_adj.cpu(), link_new_labels_M)
    print("精确度：", P_K)
    print("召回率：", R_K)
    print("F1_：", F1_)
    return P_K, R_K, F1_, prob_adj

# SHGAE_Ablation2
def main_of_SHGAE_Ablation2(num_u, num_epoch, HeterHyper_id_vec_list, friendship_old_df, link_new_labels_M):
    # 构建训练的data
    x_curLink = torch.tensor(np.array(HeterHyper_id_vec_list), dtype=torch.float32)  # 先转array再转tensor更快
    edge_index_uid = [[], []]
    for uid1, uid2 in zip(friendship_old_df['uid1'], friendship_old_df['uid2']):
        edge_index_uid[0].append(uid1)
        edge_index_uid[1].append(uid2)
    edge_index_uid = torch.tensor(edge_index_uid, dtype=torch.long)

    # 把tvc的边放进去
    edge_index_uid_tvc = [[], []]
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
    model = GATNet(num_node_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    x_epoch = []
    y_loss = []
    y_val_auc = []
    num_fit = 5
    fit_x = np.array([i + 1 for i in range(num_fit)])
    for epoch in range(1, num_epoch + 1):
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

    K = 10
    P_K, R_K, F1_ = count_o2n_Acc_Recall(K, prob_adj.cpu(), link_new_labels_M)
    print("精确度：", P_K)
    print("召回率：", R_K)
    print("F1_：", F1_)
    return P_K, R_K, F1_, prob_adj

# SHGAE_Ablation3
def main_of_SHGAE_Ablation3(num_u, num_epoch, HeterHyper_all_id_vec_list, friendship_old_df, link_new_labels_M, bipG_t, bipG_v, bipG_c):
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
        edge_index_uid_tvc[0].append(edge_t[0] - 1)
        edge_index_uid_tvc[1].append(edge_t[1] - 1)
        edge_index_uid_tvc[0].append(edge_t[1] - 1)
        edge_index_uid_tvc[1].append(edge_t[0] - 1)
    for edge_v in bipG_v.edges():
        edge_index_uid_tvc[0].append(edge_v[0] - 1)
        edge_index_uid_tvc[1].append(edge_v[1] - 1)
        edge_index_uid_tvc[0].append(edge_v[1] - 1)
        edge_index_uid_tvc[1].append(edge_v[0] - 1)
    for edge_c in bipG_c.edges():
        edge_index_uid_tvc[0].append(edge_c[0] - 1)
        edge_index_uid_tvc[1].append(edge_c[1] - 1)
        edge_index_uid_tvc[0].append(edge_c[1] - 1)
        edge_index_uid_tvc[1].append(edge_c[0] - 1)
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
    model = GATNet(num_node_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    x_epoch = []
    y_loss = []
    y_val_auc = []
    num_fit = 5
    fit_x = np.array([i + 1 for i in range(num_fit)])
    for epoch in range(1, num_epoch + 1):
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

    K = 10
    P_K, R_K, F1_ = count_o2n_Acc_Recall(K, prob_adj.cpu(), link_new_labels_M)
    print("精确度：", P_K)
    print("召回率：", R_K)
    print("F1_：", F1_)
    return P_K, R_K, F1_, prob_adj

# 计算一下超二部图,deepwalk,异构超图结果并输出 K为topK预测
# def main_of_baseline(K, deepwalk_S_id_vec_list, deepwalk_SM_id_vec_list,
#                                                       node2vec_S_id_vec_list, node2vec_SM_id_vec_list,
#                                                       LINE_S_id_vec_list, LINE_SM_id_vec_list,
#                                                       HeterHyper_id_vec_list, link_new_labels_M):

def main_of_baseline(K, deepwalk_S_id_vec_list, node2vec_S_id_vec_list,
                         LINE_S_id_vec_list, HeterHyper_id_vec_list, link_new_labels_M):
    result_baseline = {}
    prob_adj_set = {}

    # deepwalk结果
    print("-" * 10, "deepwalk_S", "-" * 10)
    prob_adj = cosinematrix(torch.tensor(np.array(deepwalk_S_id_vec_list)))
    prob_adj_set['deepwalk_S'] = prob_adj.clone().detach().requires_grad_(True) # 复制一份，以免+1000保留了
    P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    print("deepwalk_S_精确度：", P_K)
    print("deepwalk_S_召回率：", R_K)
    print("deepwalk_S_F1：", F1)
    result_baseline['deepwalk_S'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(round(F1, 5)) + '|'

    # print("-" * 10, "deepwalk_SM", "-" * 10)
    # prob_adj = cosinematrix(torch.tensor(np.array(deepwalk_SM_id_vec_list)))
    # prob_adj_set['deepwalk_SM'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    # P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    # print("deepwalk_SM_精确度：", P_K)
    # print("deepwalk_SM_召回率：", R_K)
    # print("deepwalk_SM_F1：", F1)
    # result_baseline['deepwalk_SM'] = '|' + str(round(P_K,5)) + '|' + str(round(R_K,5)) + '|' + str(round(F1,5)) + '|'

    # noede2vec结果
    print("-" * 10, "node2vec_S", "-" * 10)
    prob_adj = cosinematrix(torch.tensor(np.array(node2vec_S_id_vec_list)))
    prob_adj_set['node2vec_S'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    print("node2vec_S_精确度：", P_K)
    print("node2vec_S_召回率：", R_K)
    print("node2vec_S_F1：", F1)
    result_baseline['node2vec_S'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(round(F1, 5)) + '|'

    # print("-" * 10, "node2vec_SM", "-" * 10)
    # prob_adj = cosinematrix(torch.tensor(np.array(node2vec_SM_id_vec_list)))
    # prob_adj_set['node2vec_SM'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    # P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    # print("node2vec_SM_精确度：", P_K)
    # print("node2vec_SM_召回率：", R_K)
    # print("node2vec_SM_F1：", F1)
    # result_baseline['node2vec_SM'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(round(F1, 5)) + '|'

    # LINE结果
    print("-" * 10, "LINE_S", "-" * 10)
    prob_adj = cosinematrix(torch.tensor(np.array(LINE_S_id_vec_list)))
    prob_adj_set['LINE_S'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    print("LINE_S_精确度：", P_K)
    print("LINE_S_召回率：", R_K)
    print("LINE_S_F1：", F1)
    result_baseline['LINE_S'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(round(F1, 5)) + '|'

    # print("-" * 10, "LINE_SM", "-" * 10)
    # prob_adj = cosinematrix(torch.tensor(np.array(LINE_SM_id_vec_list)))
    # prob_adj_set['LINE_SM'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    # P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    # print("LINE_SM_精确度：", P_K)
    # print("LINE_SM_召回率：", R_K)
    # print("LINE_SM_F1：", F1)
    # result_baseline['LINE_SM'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(round(F1, 5)) + '|'

    # 异构超图结果
    print("-" * 10, "异构超图结果", "-" * 10)
    prob_adj = cosinematrix(torch.tensor(np.array(HeterHyper_id_vec_list)))
    prob_adj_set['HeterHyperWalk'] = prob_adj.clone().detach().requires_grad_(True)  # 复制一份，以免+1000保留了
    P_K, R_K, F1 = count_o2n_Acc_Recall(K, prob_adj, link_new_labels_M)
    print("异构超图_精确度：", P_K)
    print("异构超图_召回率：", R_K)
    print("异构超图_F1：", F1)
    result_baseline['HeterHyperWalk'] = '|' + str(round(P_K,5)) + '|' + str(round(R_K,5)) + '|' + str(round(F1,5)) + '|'
    return result_baseline, prob_adj_set

if __name__ == '__main__':
    cities = ["Istanbul", "Jakarta", "KualaLampur", "NYC", "SaoPaulo", "TKY"]
    # cities = ["NYC", "SaoPaulo"]

    result_markdown = {}
    # prob_adj_all = {}
    for city in cities:
        print("*" * 20, "开始处理" + city + "的数据", "*" * 20)
        # 读取数据
        print("-" * 10, "读取" + city + "原数据", "-" * 10)
        # 读取的是已经重新编排ID的数据，且id从0开始
        # friendship_new_df, friendship_old_df, checkins_df, deepwalk_S_id_vec_list, deepwalk_SM_id_vec_list, \
        # node2vec_S_id_vec_list, node2vec_SM_id_vec_list, LINE_S_id_vec_list, LINE_SM_id_vec_list, \
        # HeterHyper_id_vec_list, HeterHyper_all_id_vec_list, bipG_t, bipG_v, bipG_c = get_data(city)

        friendship_new_df, friendship_old_df, checkins_df, deepwalk_S_id_vec_list, node2vec_S_id_vec_list, \
        LINE_S_id_vec_list, HeterHyper_id_vec_list, HeterHyper_all_id_vec_list, bipG_t, bipG_v, bipG_c = get_data(city)

        num_u = len(deepwalk_S_id_vec_list)
        neg_num = 50
        link_new_labels_M = count_link_lables(num_u, friendship_new_df, friendship_old_df, neg_num)

        """
        # baseline方法
        # """
        print("-" * 10, "计算baseline方法结果", "-" * 10)
        # result_city, prob_adj_city = main_of_baseline(10, deepwalk_S_id_vec_list, deepwalk_SM_id_vec_list,
        #                                               node2vec_S_id_vec_list, node2vec_SM_id_vec_list,
        #                                               LINE_S_id_vec_list, LINE_SM_id_vec_list,
        #                                               HeterHyper_id_vec_list, link_new_labels_M)  # For Markdown

        result_city, prob_adj_city = main_of_baseline(10, deepwalk_S_id_vec_list,node2vec_S_id_vec_list,
                                                      LINE_S_id_vec_list, HeterHyper_id_vec_list, link_new_labels_M)  # For Markdown

        """
        SHGAE 的主函数
        """
        print("-" * 10, "计算SHGAE", "-" * 10)
        num_epoch = 3000  # 学习epoch数
        P_K, R_K, F1, prob_adj = main_of_SHGAE(num_u, num_epoch, HeterHyper_all_id_vec_list, friendship_old_df,
                                               link_new_labels_M, bipG_t, bipG_v, bipG_c)
        result_city['SHGAE'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(
            round(F1, 5)) + '|'  # For Markdown
        prob_adj_city['SHGAE'] = prob_adj.cpu()  # 存的是GPU里的那份

        """
        SHGAE-消融实验3
        消融3：GAT中tvc-user消息双向传递。（论证语义信息的消息传递模块有效）
        """
        print("-" * 10, "计算SHGAE_Ablation3", "-" * 10)
        num_epoch = 3000  # 学习epoch数
        P_K, R_K, F1, prob_adj = main_of_SHGAE_Ablation3(num_u, num_epoch, HeterHyper_all_id_vec_list,
                                                         friendship_old_df,
                                                         link_new_labels_M, bipG_t, bipG_v, bipG_c)
        result_city['SHGAE_Ablation3'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(
            round(F1, 5)) + '|'  # For Markdown
        prob_adj_city['SHGAE_Ablation3'] = prob_adj.cpu()  # 存的是GPU里的那份

        """
        SHGAE-消融实验1
        消融1： 用随机值 替换掉 Hypergraph Jump Walk的初始特征看模型框架结果如何。（论证HJW模块有效）
        """
        print("-" * 10, "计算SHGAE_Ablation1", "-" * 10)
        num_epoch = 3000  # 学习epoch数
        Int_vec = np.random.rand(len(HeterHyper_all_id_vec_list),len(HeterHyper_all_id_vec_list[0]))  # 随机值作为GAT初始输入
        P_K, R_K, F1, prob_adj = main_of_SHGAE(num_u, num_epoch, Int_vec, friendship_old_df,
                                                link_new_labels_M, bipG_t, bipG_v, bipG_c)
        result_city['SHGAE_Ablation1'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(
            round(F1, 5)) + '|'  # For Markdown
        prob_adj_city['SHGAE_Ablation1'] = prob_adj.cpu()  # 存的是GPU里的那份

        """
        SHGAE-消融实验2
        消融2：GAT只用社交网络部分的图去学习。（论证超图GAT有效）
        """
        print("-" * 10, "计算SHGAE_Ablation2", "-" * 10)
        num_epoch = 3000  # 学习epoch数
        P_K, R_K, F1, prob_adj = main_of_SHGAE_Ablation2(num_u, num_epoch, HeterHyper_all_id_vec_list[:num_u], friendship_old_df,
                                                link_new_labels_M)
        result_city['SHGAE_Ablation2'] = '|' + str(round(P_K, 5)) + '|' + str(round(R_K, 5)) + '|' + str(
            round(F1, 5)) + '|'  # For Markdown
        prob_adj_city['SHGAE_Ablation2'] = prob_adj.cpu()  # 存的是GPU里的那份

        # 将结果的prob_adj_all保存下来
        file = open('data_proces/' + city + '_prob_adj.pickle', 'wb')
        pickle.dump(prob_adj_city, file)
        file.close()

        # 保存markdown结果
        result_markdown[city] = result_city
        # prob_adj_all[city] = prob_adj_city

    # 输出markDown的结果格式
    print("*" * 20, "markDown结果", "*" * 20)
    print('|Dataset| Method| P@10| R@10| F1|\n|-|-|-|-|-|')
    method_list = result_markdown[cities[0]].keys()
    for city in cities:
        city_name = city
        for method in method_list:
            print('|' + city_name + '|' + method + result_markdown[city][method])
            city_name = ''