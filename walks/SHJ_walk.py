"""
异构超图游走策类
"""
import random

class SHJ_walk:
    def __init__(self, args, num_u, num_t, num_v, num_c, social_G, bipG_t, bipG_v, bipG_c):
        self.num_u = num_u
        self.num_t = num_t
        self.num_v = num_v
        self.num_c = num_c
        self.num_node = num_u + num_t + num_v + num_c
        self.social_G = social_G
        self.bipG = [bipG_t, bipG_v, bipG_c]
        self.args = args
        self.sampleHyperEdge = [0 for _ in range(num_t)] + [1 for _ in range(num_v)] + [2 for _ in range(num_c)]

    def HBdeepWalk(self):
        walks = []
        for id in range(self.num_node):
            for _ in range(self.args['r']):
                walks.append(self.one_walk(id))
        return walks

    def one_walk(self,id):
        walk = [id]
        while len(walk) < self.args['l']:
            curr_id = walk[-1]
            if curr_id >= self.num_u:
                choice_tvc = self.sampleHyperEdge[curr_id - self.num_u]
                nbrs_curr = sorted(self.bipG[choice_tvc].neighbors(curr_id))
                nbrs_weight = []
                for nbr in nbrs_curr:
                    nbrs_weight.append(self.bipG[choice_tvc][curr_id][nbr]['weight'])
                walk.append(random.choices(nbrs_curr, nbrs_weight)[0])
            else:
                if random.random() < self.args['alpha']:
                    choice_tvc = random.choice(self.sampleHyperEdge)
                    if curr_id not in self.bipG[choice_tvc].nodes:
                        # user has no check-in
                        break
                    nbrs_curr = sorted(self.bipG[choice_tvc].neighbors(curr_id))
                    nbrs_weight = []
                    for nbr in nbrs_curr:
                        nbrs_weight.append(self.bipG[choice_tvc][curr_id][nbr]['weight'])
                    walk.append(random.choices(nbrs_curr, nbrs_weight)[0])

                else:
                    nbrs_curr = sorted(self.social_G.neighbors(curr_id))
                    walk.append(random.choice(nbrs_curr))
        return walk

