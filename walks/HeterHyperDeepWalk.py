"""
异构超图游走策类
"""
import random

class HeterHyperDeepWalk:
    def __init__(self, args, num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c):
        self.num_u = num_u
        self.num_t = num_t
        self.num_v = num_v
        self.num_c = num_c
        self.num_node = num_u + num_t + num_v + num_c
        self.social_G = social_G
        self.bipG = [bipG_t, bipG_v, bipG_c]
        self.args = args # r:每个节点游走次数  l:每次游走长度  alpha:游走时访问超边概率

        # 初始化 按权重在超边上选择节点的list
        self.sampleHyperEdge = [0 for _ in range(num_t)] + [1 for _ in range(num_v)] + [2 for _ in range(num_c)]


    def HBdeepWalk(self):
        walks = []
        for id in range(1,self.num_node + 1): # 对每个node
            for _ in range(self.args['r']): # 每个node游走个数
                walks.append(self.one_walk(id))
        return walks

    def one_walk(self,id):
        walk = [id]
        while len(walk) < self.args['l']:
            curr_id = walk[-1]
            if curr_id > self.num_u: # 当前在超边上
                # 超边向user游走
                choice_tvc = self.sampleHyperEdge[curr_id - self.num_u - 1]
                nbrs_curr = sorted(self.bipG[choice_tvc].neighbors(curr_id))
                nbrs_weight = []
                for nbr in nbrs_curr:
                    nbrs_weight.append(self.bipG[choice_tvc][curr_id][nbr]['weight'])
                walk.append(random.choices(nbrs_curr, nbrs_weight)[0])  # 按权重随机抽一个
            else:  # 在社交用户节点上
                if random.random() < self.args['alpha']:  # 按概率游走超边
                    # user向超边游走
                    choice_tvc = random.choice(self.sampleHyperEdge)  # 选择tvc中的一个二部图
                    if curr_id not in self.bipG[choice_tvc].nodes:
                        # print("节点usr", curr_id, "没有check")
                        # 该点没有签到时，重新进行游走，直到选择社交边
                        break
                    nbrs_curr = sorted(self.bipG[choice_tvc].neighbors(curr_id))
                    nbrs_weight = []
                    for nbr in nbrs_curr:
                        nbrs_weight.append(self.bipG[choice_tvc][curr_id][nbr]['weight'])
                    walk.append(random.choices(nbrs_curr, nbrs_weight)[0])  # 按权重随机抽一个

                else:  # 游走社交网络边
                    nbrs_curr = sorted(self.social_G.neighbors(curr_id))
                    # 在社交边上游走不用带权
                    # nbrs_weight = []  # 带权重选
                    # for nbr in nbrs_curr:
                    #     nbrs_weight.append(self.social_G[curr_id][nbr]['weight'])
                    # walk.append(random.choice(nbrs_curr, nbrs_weight)[0])
                    walk.append(random.choice(nbrs_curr))
        return walk

