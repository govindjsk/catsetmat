import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from tqdm.autonotebook import tqdm
from src.our_modules import device, Classifier
from src.our_utils import obtain_node_embeddings, process_node_emb, get_home_path, mkdir_p, load_and_process_data, \
    get_data_path
from src.results_analyzer import plot_results
from lib.hypersagnn.main import parse_args as parse_embedding_args
from lib.fspool.main import EMB_LAYER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--dim', type=int, default=64,
                        help='Embedding dimension for node2vec. Default is 64.')
    parser.add_argument('--model_name', type=str, default='catsetmat')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs. Default is 200.')
    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch size. Default is 100.')
    parser.add_argument('--model_save_split_id', type=int, default=0,
                        help='Split id for which model is to be saved. Default is 0.')
    args = parser.parse_args()
    return args


def process_args(args):
    data_name = args.data_name

    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    return data_name, splits, num_epochs, batch_size, args.model_save_split_id, args.dim, args.model_name


def data_modify(data):
    u_, v_, l_ = zip(*data)
    npoints_u = [len(x[x > 0].tolist()) for x in u_]
    npoints_v = [len(x[x > 0].tolist()) for x in v_]
    mask_u = torch.cat([(x > 0).float().view(1, x.shape[0], 1) for x in u_], dim=0)
    mask_v = torch.cat([(x > 0).float().view(1, x.shape[0], 1) for x in v_], dim=0)

    return u_, npoints_u, v_, npoints_v, mask_u, mask_v, l_


def train(model, data, globaliter=0, model_name='catsetmat'):
    globaliter += 1
    model.train()

    # FSPOOL:
    if model_name == 'fspool':
        U, n_points_U, V, n_points_V, mask_U, mask_V, l_ = data_modify(data)
        U = torch.cat(U, dim=0).view(len(U), U[0].shape[0]).to(device)
        V = torch.cat(V, dim=0).view(len(V), V[0].shape[0]).to(device)
        gold = torch.Tensor(l_).view(-1, 1).to(device)
        inputs = (U, V, torch.from_numpy(np.array((list(map(int, n_points_U))))).to(device),
                  torch.from_numpy(np.array(list(map(int, n_points_V)))).to(device), mask_U.to(device),
                  mask_V.to(device))
        label = model(inputs)
        loss = nn.BCEWithLogitsLoss()(label, gold).to(device)

    # CATSETMAT:
    if model_name == 'catsetmat':
        u_, v_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
        output, weights = model(xx, yy)
        loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
        label = output.squeeze(-1)
        del xx, yy, weights
        torch.cuda.empty_cache()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    auc = roc_auc_score(l_, label.cpu().detach().numpy())
    return loss.item(), auc, None


def test(model, data, model_name='catsetmat'):
    model.eval()

    # FSPOOL:
    if model_name == 'fspool':
        U, n_points_U, V, n_points_V, mask_U, mask_V, l_ = data_modify(data)
        U = torch.cat(U, dim=0).view(len(U), U[0].shape[0]).to(device)
        V = torch.cat(V, dim=0).view(len(V), V[0].shape[0]).to(device)
        inputs = (U, V, torch.from_numpy(np.array((list(map(int, n_points_U))))).to(device),
                  torch.from_numpy(np.array(list(map(int, n_points_V)))).to(device), mask_U.to(device),
                  mask_V.to(device))
        gold = torch.Tensor(l_).view(-1, 1).to(device)
        label = model(inputs)
        loss = nn.BCEWithLogitsLoss()(label, gold).to(device)

    # CATSETMAT:
    if model_name == 'catsetmat':
        u_, v_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
        output, weights = model(xx, yy)
        loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
        label = output.squeeze(-1)

    auc = roc_auc_score(l_, label.cpu().detach().numpy())
    return loss.item(), auc, None


def read_cache_node_embeddings(args, node_list_set, train_set, data_name, set_name, split_id, base_path, silent=True):
    file_name = os.path.join(base_path, 'walks/embeddings/{}/{}_wv_{}__{}{}{}.npy'.format(
        data_name, data_name, args.dimensions, data_name, split_id, set_name))
    try:
        if not silent:
            print('Reading embeddings from cache ({})...'.format(file_name))
        A = np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        print('Cache not found. Generating...')
        A = obtain_node_embeddings(args, node_list_set, train_set, data_name, set_name, split_id, base_path,
                                   silent=silent)
    node_embedding = process_node_emb(A, node_list_set, args)
    return node_embedding


def perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path, num_epochs, batch_size,
                       model_save_split_id, model_name):
    global criterion, optimizer
    pickled_path = os.path.join(data_path, 'processed', data_name, '{}.pkl'.format(split_id))
    train_data, test_data, U_t, V_t, node_list_U, node_list_V = load_and_process_data(pickled_path)
    base_path = home_path
    node_embedding_U = read_cache_node_embeddings(emb_args, node_list_U, U_t, data_name, 'U', split_id, base_path)
    node_embedding_V = read_cache_node_embeddings(emb_args, node_list_V, V_t, data_name, 'V', split_id, base_path)

    # FSPOOL:
    if model_name == 'fspool':
        U, n_points_U, V, n_points_V, mask_U, mask_V, l_ = data_modify(train_data)
        hidden_dim = 128
        latent_dim = emb_args.dimensions
        model = EMB_LAYER(node_embedding_U, node_embedding_V, 0, latent_dim + 1,
                          latent_dim, hidden_dim,
                          set_size_U=max(n_points_U),
                          set_size_V=max(n_points_V),
                          skip=False, relaxed=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1E-6)

    # CATSETMAT:
    if model_name == 'catsetmat':
        latent_dim = emb_args.dimensions
        model = Classifier(n_head=8,
                           d_model=latent_dim,
                           d_k=int(latent_dim / 4) if latent_dim >= 4 else 1,
                           d_v=int(latent_dim / 4) if latent_dim >= 4 else 1,
                           node_embedding1=node_embedding_U,
                           node_embedding2=node_embedding_V,
                           diag_mask=False,
                           bottle_neck=latent_dim).to(device).to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
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
            batch_loss, batch_auc, weights = train(model, batch_data, model_name=model_name)
            batch_losses.append(batch_loss)
            batch_aucs.append(batch_auc)
        train_loss = np.mean(batch_losses)
        train_auc = np.mean(batch_aucs)
        test_loss, test_auc, test_weights = test(model, test_data, model_name=model_name)
        loss.append((train_loss, test_loss))
        auc.append((train_auc, test_auc))
        t.set_description("AUC train: {}, test: {}".format(round(train_auc, 4), round(test_auc, 4)))
        t.refresh()
    results = {"AUC": auc, "loss": loss}
    pickle.dump(results, open(os.path.join(result_path, '{}_{}.pkl'.format(model_name, split_id)), 'wb'))
    if split_id == model_save_split_id:
        torch.save(model, os.path.join(result_path, 'model_{}_{}.mdl'.format(model_name, split_id)))
    return model


def main():
    set_torch_environment()
    data_name, splits, num_epochs, batch_size, model_save_split_id, dim, model_name = process_args(parse_args())
    emb_args = parse_embedding_args()
    emb_args.dimensions = dim
    home_path = get_home_path()
    data_path = get_data_path()
    result_path = os.path.join(home_path, 'results', data_name, 'res')
    mkdir_p(result_path)
    for i, split_id in enumerate(splits):
        print('------- SPLIT#{} ({} of {}) -------'.format(split_id, i, len(splits)))
        perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path, num_epochs, batch_size,
                           model_save_split_id, model_name)
    plot_results(splits, result_path, model_name)


if __name__ == '__main__':
    main()
