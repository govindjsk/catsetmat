import argparse
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import sys
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from tqdm.autonotebook import tqdm
from src.our_modules import device, Classifier
from src.our_utils import obtain_node_embeddings, process_node_emb, get_home_path, mkdir_p, load_and_process_data, \
    get_data_path

sys.path.append(get_home_path())
from lib.hypersagnn.main import parse_args as parse_embedding_args


def set_torch_environment():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="CATSETMAT: Main module")

    parser.add_argument('--data_name', type=str, default='sample_mag_acm')
    parser.add_argument('--num_splits', type=int, default=15,
                        help='Number of train-test-splits / negative-samplings. Default is 15.')
    parser.add_argument('--start_split', type=int, default=0,
                        help='Start id of splits; splits go from start_split to start_split+num_splits. Default is 0.')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs. Default is 200.')
    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch size. Default is 300.')
    args = parser.parse_args()
    return args


def process_args(args):
    data_name = args.data_name

    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    return data_name, splits, num_epochs, batch_size


def train(model, data, globaliter=0):
    globaliter += 1
    model.train()
    u_, v_, l_ = zip(*data)
    xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
    yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
    output, weights = model(xx, yy)
    loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    label = output.squeeze(-1)
    auc = roc_auc_score(l_, label.cpu().detach().numpy())
    return loss.item(), auc, weights


def test(model, data):
    model.eval()
    u_, v_, l_ = zip(*data)
    xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
    yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
    output, weights = model(xx, yy)
    loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
    label = output.squeeze(-1)
    auc = roc_auc_score(l_, label.cpu().detach().numpy())
    return loss.item(), auc, weights


def read_cache_node_embeddings(args, node_list_set, train_set, data_name, set_name, split_id, base_path, silent=True):
    file_name = os.path.join(base_path, 'walks/embeddings/{}/{}_wv_{}__{}{}{}.npy'.format(
        data_name, data_name, args.dimensions, data_name, split_id, set_name))
    try:
        if not silent:
            print('Reading embeddings from cache ({})...'.format(file_name))
        A = np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        print('Cache not found. Generating...')
        A = obtain_node_embeddings(args, node_list_set, train_set, data_name, set_name, split_id, base_path)
    node_embedding = process_node_emb(A, node_list_set, args)
    return node_embedding



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


def perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path, num_epochs, batch_size):
    global criterion, optimizer
    pickled_path = os.path.join(data_path, 'processed', data_name, '{}.pkl'.format(split_id))
    train_data, test_data, U_t, V_t, node_list_U, node_list_V = load_and_process_data(pickled_path)
    base_path = home_path
    node_embedding_U = read_cache_node_embeddings(emb_args, node_list_U, U_t, data_name, 'U', split_id, base_path)
    node_embedding_V = read_cache_node_embeddings(emb_args, node_list_V, V_t, data_name, 'V', split_id, base_path)

    model = Classifier(n_head=8,
                       d_model=64,
                       d_k=16,
                       d_v=16,
                       node_embedding1=node_embedding_U,
                       node_embedding2=node_embedding_V,
                       diag_mask=False,
                       bottle_neck=64).to(device).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    loss = []
    auc = []
    t = tqdm(range(num_epochs), 'AUC')
    for i in t:
        train_data1 = shuffle(train_data)
        break_condition = False
        batch_losses = []
        batch_aucs = []
        j = 0
        while not break_condition:
            if j + batch_size < len(train_data1):
                batch_data = train_data1[j:j + batch_size]
                j += batch_size
            else:
                batch_data = train_data1[j:]
                break_condition = True
            batch_loss, batch_auc, weights = train(model, batch_data)
            batch_losses.append(batch_loss)
            batch_aucs.append(batch_auc)
        train_loss = np.mean(batch_losses)
        train_auc = np.mean(batch_aucs)
        test_loss, test_auc, test_weights = test(model, test_data)
        loss.append((train_loss, test_loss))
        auc.append((train_auc, test_auc))
        t.set_description("AUC train: {}, test: {}".format(round(train_auc, 4), round(test_auc, 4)))
        t.refresh()
        # print('({}/{})'.format(round(train_auc, 4), round(test_auc, 4)), end=' ')
        # print("epoch {} :train loss {} and auc {}: test loss {} and auc {} : ".format(i, train_loss_, train_auc,
        #                                                                               test_loss_, test_auc))
    Results = {"AUC": auc, "loss": loss}
    pickle.dump(Results, open(os.path.join(result_path, '{}.pkl'.format(split_id)), 'wb'))
    torch.save(model, os.path.join(result_path, 'model_{}.mdl'.format(split_id)))
    return model


def main():
    set_torch_environment()
    data_name, splits, num_epochs, batch_size = process_args(parse_args())
    emb_args = parse_embedding_args()
    home_path = get_home_path()
    data_path = get_data_path()
    result_path = os.path.join(home_path, 'results', data_name)
    mkdir_p(result_path)
    for split_id in tqdm(splits, 'Split #'):
        perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path, num_epochs, batch_size)
    plot_results(splits, result_path)


if __name__ == '__main__':
    main()
