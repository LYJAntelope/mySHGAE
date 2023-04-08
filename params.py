"""
参数文件
"""
# parameter of SHJ-walk
args = {'r': 10, 'l': 80, 'alpha': 0.2}

# parameter of skip-gram
kwargs = {}
kwargs["min_count"] = kwargs.get("min_count", 0)
kwargs["vector_size"] = 64
kwargs["sg"] = 1  # skip gram
kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
kwargs["workers"] = 3
kwargs["window"] = 5
kwargs["epochs"] = 5

# TOP-K
top_K = 10
# Number of negative sampling candidates
negative_candidates = 50
# GAT_dimension
GAT_dimension = 64
# GAT_head
head_Num = 5