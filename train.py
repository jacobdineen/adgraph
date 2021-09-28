import dgl
import numpy as np

from graph_class import runthrough, poison_test
from policy_gradient import Policy, select_action, perform_action, graphs_to_state
from graph_insertion import Actions as A
from data import get_dataset
from dgl.data import MiniGCDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import time
from itertools import count
import copy
import pickle

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

# TODO


def trn(
    trainset, testset, num_epochs: int, graph_epochs: int, points: int, run_num: int
):
    """run poison attack and rl algo

    Parameters
    ----------
    train : tensor (graph,label) pairs
        unperturbed trainset
    test : tensor (graph,label) pairs
        unperturbed testset
    num_epochs : int
        number of iterations to run
    graph_epochs : int
        number of iterations to train graph neural net
    points : int
        number of poison points

    Returns
    -------
    ret_info, reward_arr, max_val, max_labels
        [description]
    """
    # logging data
    logging = open("data/logs.txt", "a")
    action_hist = np.zeros([num_epochs, len(train) * 1])  # [[],[]]
    # Perform a pass of training/testing given datasets
    # return trained graph neural network & baseline testset accuracy
    # serialize the original graphs to reset env at each episode
    pickle.dump(trainset.graphs, open("data/save.p", "wb"))

    max_labels = trainset.labels
    g_model, init_reward = runthrough(
        trainset=trainset, testset=testset, epochs=graph_epochs
    )

    for i_episode in count(1):
        # after each attempt of poisoning via label, revert to original labels
        # unpickle original graph structure
        trainset.graphs = pickle.load(open("data/save.p", "rb"))
        state, ep_reward, rand_reward = graphs_to_state(trainset.graphs), 0, 0
        # # For first poisoning point, reward is how much better ("worse") than baseline acc.
        prev_acc = init_reward

        for t in range(1, points + 1):  # Assume 18 Poisoning points
            # fetch action | state
            action = select_action(state, policy)
            action_hist[i_episode - 1][action] += 1
            # move to s' given a in s
            state, trainset, graph, action = perform_action(
                trainset, action, state, num_classes
            )

            # get acc on poison test with modified data
            curr_acc = poison_test(model=g_model, trainset=trainset, testset=testset)

            reward = prev_acc - curr_acc
            prev_acc = curr_acc
            policy.rewards.append(reward)
            ep_reward += reward

        # perform policy updates/optimization
        policy.finish_episode(optimizer, eps)
        print("RL/RUN/EP/ACC/Reward:", run_num, i_episode, curr_acc, ep_reward)
        logging.write(
            f"RL, {run_num}, {i_episode}, {init_reward}, {curr_acc}, {ep_reward}" "\n"
        )

        if i_episode == num_epochs:
            logging.close()
            break
    return action_hist


def trn_random(
    train, test, num_epochs: int, graph_epochs: int, points: int, run_num: int
):
    logging = open("logs.txt", "a")
    action_hist = np.zeros([num_epochs, len(train) * 1])  # [[],[]]
    trainset, testset = train, test
    pickle.dump(trainset.graphs, open("data/save.p", "wb"))

    max_labels = trainset.labels
    g_model, init_reward = runthrough(
        trainset=trainset, testset=testset, epochs=graph_epochs
    )

    for i_episode in count(1):
        trainset.graphs = pickle.load(open("data/save.p", "rb"))
        state, ep_reward, rand_reward = graphs_to_state(trainset.graphs), 0, 0
        prev_acc = init_reward

        for t in range(1, points + 1):  # Assume 18 Poisoning points
            action = np.random.choice(len(train))
            action_hist[i_episode - 1][action] += 1
            state, trainset, graph, action = perform_action(
                trainset, action, state, num_classes
            )

            curr_acc = poison_test(model=g_model, trainset=trainset, testset=testset)

            reward = prev_acc - curr_acc
            prev_acc = curr_acc
            policy.rewards.append(reward)
            ep_reward += reward

        print("RANDOM/RUN/EP/ACC/Reward:", run_num, i_episode, curr_acc, ep_reward)
        logging.write(
            f"Random, {run_num}, {i_episode}, {init_reward}, {curr_acc}, {ep_reward}"
            "\n"
        )
        if i_episode == num_epochs:
            logging.close()
            break
    return action_hist


if __name__ == "__main__":
    torch.manual_seed(42)
    train_size = 150
    test_size = 30
    min_graph_nodes = 10
    max_graph_nodes = 15
    poison_points = int(train_size * 0.5)  # attacking budget
    num_epochs = 5
    num_graph_epochs = 70
    num_runs = 1
    print(device)

    action_histories = []
    for i in range(num_runs):
        print(f"run: {i}/{num_runs}")
        # assert poison_points < min_graph_nodes, 'poison points required to be > min graph nodes'
        # ------------------------
        # fetch datasets
        # ------------------------
        trainset = MiniGCDataset(train_size, min_graph_nodes, max_graph_nodes)
        testset = MiniGCDataset(test_size, min_graph_nodes, max_graph_nodes)
        num_classes = len(np.unique(trainset.labels))
        # ------------------------
        # Instantiate RL Algorithm
        # ------------------------
        policy = Policy(
            in_dim=len(graphs_to_state(trainset.graphs)),  # num graphs
            hidden_dim=156,  # num hidden neurons
            out_dim=len(trainset) * 1,  # num possible actions
            dropout=0.6,
        )

        optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        eps = np.finfo(np.float32).eps.item()

        # ------------------------
        # Run Poison Attack
        # ------------------------

        action_histories.append(
            trn_random(
                train=trainset,
                test=testset,
                num_epochs=num_epochs,
                graph_epochs=num_graph_epochs,
                points=poison_points,
                run_num=i,
            )
        )

        action_histories.append(
            trn(
                train=trainset,
                test=testset,
                num_epochs=num_epochs,
                graph_epochs=num_graph_epochs,
                points=poison_points,
                run_num=i,
            )
        )

    with open("data/action_hist.npy", "wb") as f:
        np.save(f, action_histories)

    # 800 policy iter, 60 GNN iter, 18 Poisoning Points 320/80
    # 2: 1000 policy iter, 60 GNN iter, 20 Poisoning Points- NEW: Reward ONLY at end of episode 320/80
    # 3 400/50/18 - REVERT: Reward after every poison point, however reward rel. to last poisoned point
#     np.save("info.npy", info)
#     np.save("reward.npy", reward)
#     np.save("max_labels.npy", max_labels)
