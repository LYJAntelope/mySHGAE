"""
DeepWalk  带权
"""
import random

class DeepWalk:
    def __init__(self, args, bipG):
        self.bipG = bipG
        self.args = args

    def HBdeepWalk(self):
        walks = []
        for node in self.bipG.nodes: # 对每个user
            for _ in range(self.args['r']): # 每个user游走个数
                walks.append(self.one_walk(node))
        return walks

    def one_walk(self,node):
        walk = [node]
        while len(walk) < self.args['l']:
            curr_node = walk[-1]
            nbrs_curr = sorted(self.bipG.neighbors(curr_node))
            if len(nbrs_curr) == 0:
                print("出错！！：节点usr",curr_node,"没有邻居")
                break
            nbrs_weight = []
            for nbr in nbrs_curr:
                nbrs_weight.append(self.bipG[curr_node][nbr]['weight'])
            walk.append(random.choices(nbrs_curr,nbrs_weight)[0]) # 按权重随机抽一个
        return walk

