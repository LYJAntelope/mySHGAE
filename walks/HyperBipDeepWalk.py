"""
超二部图游走类
"""
import random

class HyperBipDeepWalk:
    def __init__(self, args, num_u, num_t, num_v, num_c, bipG_t, bipG_v, bipG_c):
        self.num_u = num_u
        self.num_t = num_t
        self.num_v = num_v
        self.num_c = num_c
        self.num_node = num_u + num_t + num_v + num_c
        self.bipG = [bipG_t, bipG_v, bipG_c]
        self.args = args

        # 初始化 按权重在超边上选择节点的list
        self.sampleHyperEdge = [0 for _ in range(num_t)] + [1 for _ in range(num_v)] + [2 for _ in range(num_c)]
        self.set_tvc_nodes = set(bipG_t.nodes) | set(bipG_v.nodes) | set(bipG_c.nodes)

    def HBdeepWalk(self):
        walks = []

        for id in self.set_tvc_nodes: # 对每个node
            for _ in range(self.args['r']): # 每个node游走个数
                walks.append(self.one_walk(id))
        return walks

    def one_walk(self,id):
        walk = [id]
        while len(walk) < self.args['l']:
            curr_id = walk[-1]
            if curr_id <= self.num_u:
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
            else:
                # 超边向user游走
                choice_tvc = self.sampleHyperEdge[curr_id - self.num_u - 1]
                nbrs_curr = sorted(self.bipG[choice_tvc].neighbors(curr_id))
                nbrs_weight = []
                for nbr in nbrs_curr:
                    nbrs_weight.append(self.bipG[choice_tvc][curr_id][nbr]['weight'])
                walk.append(random.choices(nbrs_curr, nbrs_weight)[0])  # 按权重随机抽一个

        return walk

