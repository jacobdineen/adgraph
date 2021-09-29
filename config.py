# paths
action_hist_path = "data/action_hist.npy"
data_save_path = "data/init_graphs.p"
logging_path = "data/logs.txt"

# hyperparams
train_size = 150
test_size = 30
min_graph_nodes = 10
max_graph_nodes = 15
poison_points = int(train_size * 0.1)  # attacking budget
num_epochs = 10
num_graph_epochs = 70
num_runs = 1
