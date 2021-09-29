# paths
action_hist_path = "data/action_hist.npy"
data_save_path = "data/init_graphs.p"
logging_path = "data/logs.txt"


# Minigc params
train_size = 150
test_size = 30
min_graph_nodes = 10
max_graph_nodes = 15

# GCN Params
gcn_learning_rate = 0.01
gcn_batch_size = 64
gcn_workers = 10
gcn_additional_epochs = 1
num_graph_epochs = 70

# RL Params
poison_points = int(train_size * 0.1)  # attacking budget
num_epochs = 10
num_runs = 1
