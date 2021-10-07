import numpy as np

# paths
data_save_path = "data/init_graphs.p"


# datasets
minigc_size = 500
min_graph_nodes = 10
max_graph_nodes = 15
datasets = [
    # "minigc",
    # "imdb",
    "ptc_fm",
]


# GCN Params
gcn_learning_rate = 0.01
gcn_batch_size = 256
gcn_workers = 1
gcn_additional_epochs = 5
num_graph_epochs = 80
gcn_in_dim = 1
gcn_hidden = 256

# RL Params
attacking_budget = [0.01, 0.02, 0.05]
num_epochs = 100
num_runs = 5
gamma = 0.99
eps = np.finfo(np.float32).eps.item()
rl_learning_rate = 0.01
