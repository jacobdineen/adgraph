# Thesis_Aug# HRL

#### Clone this repository and run the below within the directory:

```python
conda env create environment.yml
```

#### Set hyperparameters in config.py:

```python
# paths
action_hist_path = "data/action_hist.npy"
data_save_path = "data/init_graphs.p"


# datasets
minigc_size = 500
min_graph_nodes = 10
max_graph_nodes = 15
datasets = ["minigc", "imdb", "kki", "letter_med"]


# GCN Params
gcn_learning_rate = 0.01
gcn_batch_size = 124
gcn_workers = 1
gcn_additional_epochs = 1
num_graph_epochs = 70
gcn_in_dim = 1
gcn_hidden = 256

# RL Params
attacking_budget = [0.01, 0.02, 0.05, 0.10]  # attacking budget
num_epochs = 100
num_runs = 5
gamma = 0.99
eps = np.finfo(np.float32).eps.item()
rl_learning_rate = 0.01
```

#### Run train.py to run the full suite of experiments

```python
python train.py
```