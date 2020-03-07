import pickle
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

from src.our_utils import device, get_home_path, load_and_process_data, get_data_path


def plot_results(splits, result_path):
    dfs = []
    for split_id in splits:
        try:
            Results = pickle.load(open(os.path.join(result_path, '{}.pkl'.format(split_id)), 'rb'))
        except EOFError:
            continue
        df = pd.DataFrame(Results)
        df['train_auc'] = df['AUC'].apply(lambda x: x[0])
        df['test_auc'] = df['AUC'].apply(lambda x: x[1])
        df['train_loss'] = df['loss'].apply(lambda x: x[0])
        df['test_loss'] = df['loss'].apply(lambda x: x[1])
        df['split_id'] = split_id
        dfs.append(df[['train_auc', 'test_auc', 'train_loss', 'test_loss']])

    means = pd.concat([df.reset_index() for df in dfs]).groupby('index').agg(lambda x: (round(np.mean(x), 4)))
    stds = pd.concat([df.reset_index() for df in dfs]).groupby('index').agg(lambda x: (round(np.std(x), 4)))

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    means[['train_auc', 'test_auc']].plot(yerr=stds, ax=axs[0], capsize=4)
    axs[0].grid()
    means[['train_loss', 'test_loss']].plot(yerr=stds, ax=axs[1], capsize=4)
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'learning_curve.png'))


def visualize_attn(model, data_point, node_tokens=None):
    """
    :param model:
    :param data_point: In original format (U, V, Labels)
    :param node_tokens:
    :return:
    """
    size = int((data_point[0][0] > 0).sum())
    size_ = int((data_point[0][1] > 0).sum())
    u_, v_, l_ = zip(*data_point)
    xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
    yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
    output, weights = model(xx, yy)
    self1 = weights['attn_value']['self'][0][:, :size, :size]
    self1_ = weights['attn_value']['self'][1][:, :size_, :size_]
    self2 = weights['attn_value']['self'][2][:, :size, :size]
    self2_ = weights['attn_value']['self'][3][:, :size_, :size_]
    cross = weights['attn_value']['cross'][0][:, :size, :size_]
    cross_ = weights['attn_value']['cross'][1][:, :size_, :size]

    self1_final = torch.cat([torch.cat([self1, torch.zeros_like(cross)], dim=2),
                             torch.cat([torch.zeros_like(cross_), self1_], dim=2)], dim=1)
    self2_final = torch.cat([torch.cat([self2, torch.zeros_like(cross)], dim=2),
                             torch.cat([torch.zeros_like(cross_), self2_], dim=2)], dim=1)
    cross_final = torch.cat([torch.cat([torch.zeros_like(self1), cross], dim=2),
                             torch.cat([cross_, torch.zeros_like(self1_)], dim=2)], dim=1)
    # call_html()
    if not node_tokens:
        node_tokens = list(map(lambda x: "U_{}".format(x), map(int, u_[0])))[:size] + \
                      list(map(lambda x: "V_{}".format(x), map(int, v_[0])))[:size_]
    attention = [self1_final.unsqueeze(0), cross_final.unsqueeze(0), self2_final.unsqueeze(0)]
    # head_view(attention, node_tokens)
    return attention, node_tokens


if __name__ == '__main__':
    data_name = 'sample_mag_acm'
    split_id = 0
    home_path = get_home_path()
    data_path = get_data_path()
    pickle_path = os.path.join(data_path, 'processed', data_name, '{}.pkl'.format(split_id))
    model_path = os.path.join(home_path, 'results/{}/model_{}.mdl'.format(data_name, split_id))
    model = torch.load(model_path)

    _, test_data, _, _, _, _ = load_and_process_data(pickle_path)
    data_point = [test_data[0]]

    attention, tokens = visualize_attn(model, data_point)
    # call_html()
    # head_view(attention, tokens)
