import numpy as np
import config

from graph_classifier import runthrough, poison_test
from rl.policy_gradient import Policy, select_action, perform_action, graphs_to_state
from data import get_dataset

import torch
import torch.optim as optim

import pickle
from rl.action_space import Actions as A
import logging


torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# TODO


# def write_iter_logs(logging, run_num, i_episode, init_reward, curr_acc, ep_reward):
#     if i_episode == config.num_epochs:
#         print("Max Number of iterations reached. Terminating.")
#         logging.close()
#     print("RUN/EP/ACC/Reward:", run_num, i_episode, curr_acc, ep_reward)
#     logging.write(
#         f"RL, {run_num}, {i_episode}, {init_reward}, {curr_acc}, {ep_reward}" "\n"
#     )


def set_policy(trainset, A):
    # Instantiate RL Algorithm
    # ------------------------
    policy = Policy(
        in_dim=len(graphs_to_state(trainset.graphs)),  # num graphs
        hidden_dim=256,  # num hidden neurons
        out_dim=len(trainset) * len(A.action_space),  # num possible actions
        dropout=0.5,
    )

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()
    return policy, optimizer, eps


def trn(
    trainset,
    testset,
    num_epochs: int,
    graph_epochs: int,
    points: int,
    run_num: int,
    num_classes: int,
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
    # logging = open("data/logs.txt", "a")
    pickle.dump(trainset.graphs, open("data/init_graphs.p", "wb"))

    action_hist = {}

    # max_labels = trainset.labels
    g_model, init_reward = runthrough(
        trainset=trainset, testset=testset, epochs=graph_epochs
    )

    policy, optimizer, eps = set_policy(trainset, A)

    for i_episode in range(num_epochs):
        # after each attempt of poisoning via label, revert to original labels
        # unpickle original graph structure
        trainset.graphs = pickle.load(open("data/init_graphs.p", "rb"))
        state, ep_reward, _ = graphs_to_state(trainset.graphs), 0, 0
        # # For first poisoning point, reward is how much better ("worse") than baseline acc.
        prev_acc = init_reward

        episode_actions = []
        for t in range(points):  # Assume 18 Poisoning points
            # fetch action | state
            action = select_action(state, policy)
            episode_actions.append(action)
            # move to s' given a in s
            state, trainset, _, action = perform_action(
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
        logging.info(
            f"rl, {run_num}, {i_episode}, {init_reward}, {curr_acc}, {ep_reward}"
        )
        # write_iter_logs(logging, run_num, i_episode, init_reward, curr_acc, ep_reward)
    return action_hist


def trn_random(
    trainset,
    testset,
    num_epochs: int,
    graph_epochs: int,
    points: int,
    run_num: int,
    num_classes: int,
):
    # logging = open(config.logging_path, "a")
    pickle.dump(trainset.graphs, open(config.data_save_path, "wb"))

    action_hist = {}

    g_model, init_reward = runthrough(
        trainset=trainset, testset=testset, epochs=graph_epochs
    )

    for i_episode in range(num_epochs):
        trainset.graphs = pickle.load(open(config.data_save_path, "rb"))
        state, ep_reward, _ = graphs_to_state(trainset.graphs), 0, 0
        prev_acc = init_reward

        episode_actions = []
        for t in range(points):  # Assume 18 Poisoning points
            action = np.random.choice(len(trainset))
            episode_actions.append(action)
            state, trainset, _, action = perform_action(
                trainset, action, state, num_classes
            )

            curr_acc = poison_test(model=g_model, trainset=trainset, testset=testset)

            reward = prev_acc - curr_acc
            prev_acc = curr_acc
            ep_reward += reward
        action_hist[i_episode] = episode_actions

        logging.info(
            f"rand, {run_num}, {i_episode}, {init_reward}, {curr_acc}, {ep_reward}"
        )
    return action_hist


if __name__ == "__main__":
    logging.basicConfig(filename="data/logging.log", level=logging.INFO)
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    action_histories = {}  # nested dict

    for i in range(config.num_runs):
        logging.info(f"run: {i}/{config.num_runs}")
        trainset, testset, num_classes = get_dataset(config)
        logging.info("RANDOM")
        action_histories["random"] = trn_random(
            trainset=trainset,
            testset=testset,
            num_epochs=config.num_epochs,
            graph_epochs=config.num_graph_epochs,
            points=config.poison_points,
            run_num=i,
            num_classes=num_classes,
        )

        logging.info("RL")
        action_histories["policy"] = trn(
            trainset=trainset,
            testset=testset,
            num_epochs=config.num_epochs,
            graph_epochs=config.num_graph_epochs,
            points=config.poison_points,
            run_num=i,
            num_classes=num_classes,
        )

    with open(config.action_hist_path, "wb") as f:
        np.save(f, action_histories)
