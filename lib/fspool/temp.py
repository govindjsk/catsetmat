'''
# home_path = '/content/drive/My Drive/projects/textual_analysis_email/'

home_path = '/home/jupyter/project/textual_analysis_email'

# sample_path = os.path.join(home_path, 'sample_data')

data_params = {'home_path': home_path,
               'r_label_file': 'id_p_map.txt',
               'u_label_file': 'id_a_map.txt',
               'v_label_file': 'id_k_map.txt',
               'r_u_list_file': 'p_a_list_train.txt',
               'r_v_list_file': 'p_k_list_train.txt',
               'emb_pkl_file': 'nodevectors.pkl'}
# methods = [commonneigh, admic_adar, jaccard]
# method_name_map = dict(zip(methods, ['CN', 'AA', 'JC']))
num_iter = 2


pos_A, pos_B = load_bipartite_hypergraph(data_params)
G, obs_pos, unobs_data, V_offset = data_process(pos_A, pos_B, neg_pos_ratio = 1, unobs_ratio=0.5)

max_id=max(list(G.nodes))
max_id


embedding_map=model(G,P=0.25,Q=0.25,WALK_LENGTH=100,WINDOW_SIZE=10)

emb_map=[]

for i in range(0,max_id):
  emb_map.append(embedding_map.get_vector(str(i)))

emb_map.append(np.zeros(128,dtype='float'))

def mapping(data,max_id):


    pairs, labels = zip(*data)
    U, V = zip(*pairs)

    n_points_U = np.array([len(x) for x in U])
    n_points_V = np.array([len(x) for x in V])
    cardinality_U = max(n_points_U)
    cardinality_V = max(n_points_V)


    U = [x + [max_id]*(cardinality_U - len(x)) for x in U]
    V = [x + [max_id]*(cardinality_V - len(x)) for x in V]
    return U, V, n_points_U, n_points_V, cardinality_U, cardinality_V, labels


weight=torch.from_numpy(np.matrix(emb_map)).type(torch.FloatTensor)


# weight=torch.randn((weight.size(0),weight.size(1))).type(torch.FloatTensor)
# weight[weight.size(0)-1][:]=0
# weight[weight.size(0)-1]



hidden_dim = 256
latent_dim = 32


U, V, n_points_U, n_points_V, cardinality_U, cardinality_V, labels = mapping(unobs_data, max_id)
unobs_data=list(zip(U, V, n_points_U, n_points_V, labels))
train_data,test_data=train_test_split(unobs_data,test_size=0.2)
U, V, n_points_U, n_points_V, labels=zip(*train_data)

U=torch.from_numpy(np.array(U)).cuda()
V=torch.from_numpy(np.array(V)).cuda()



tU, tV, tn_points_U, tn_points_V, tlabels=zip(*test_data)
tU=torch.from_numpy(np.array(tU)).cuda()
tV=torch.from_numpy(np.array(tV)).cuda()



# tpoints_U, tpoints_V, tn_U, tn_V, tc_U, tc_V, t_labels = mapping(embedding_map,test_data)

# points_U = pad_zeros(torch.Tensor(points_U), max([c_U,tc_U]))
# points_V = pad_zeros(torch.Tensor(points_V), max([c_V,tc_V]))

# tpoints_U = pad_zeros(torch.Tensor(tpoints_U), max([c_U,tc_U]))
# tpoints_V = pad_zeros(torch.Tensor(tpoints_V), max([c_V,tc_V]))


# net = BLP(input_channels = 129,
#           output_channels = latent_dim,
#           set_size_U = cardinality_U,
#           set_size_V = cardinality_V,
#           dim = hidden_dim,
#           skip = False,
#           relaxed = False)


net=EMB_LAYER(weight,max_id,129,
              latent_dim,hidden_dim,
              set_size_U=int(cardinality_U),
              set_size_V = int(cardinality_V),
              skip=False,relaxed=False).cuda()

# # optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1E-6)
# optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, weight_decay=1E-6)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=0.0001, weight_decay=1E-6)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1E-6)

mask_U = torch.from_numpy(np.array([[1]*n_points_U[i] + [0]*(cardinality_U-n_points_U[i]) for i in range(len(U))])).type(torch.FloatTensor).view(U.shape[0],cardinality_U,1).cuda()
mask_V= torch.from_numpy(np.array([[1]*n_points_V[i] + [0]*(cardinality_V-n_points_V[i]) for i in range(len(V))])).type(torch.FloatTensor).view(V.shape[0],cardinality_V,1).cuda()
tmask_U = torch.from_numpy(np.array([[1]*tn_points_U[i] + [0]*(cardinality_U-tn_points_U[i]) for i in range(len(tU))])).type(torch.FloatTensor).view(tU.shape[0],cardinality_U,1).cuda()
tmask_V= torch.from_numpy(np.array([[1]*tn_points_V[i] + [0]*(cardinality_V-tn_points_V[i]) for i in range(len(tV))])).type(torch.FloatTensor).view(tV.shape[0],cardinality_V,1).cuda()

from tqdm import tqdm_notebook
import pickle
losses = []
test_losses = []
gold = torch.Tensor(labels).view(-1, 1).cuda()
tgold = torch.Tensor(tlabels).view(-1, 1).cuda()

n_epoch =300
inputs = (U,V,  torch.from_numpy(np.array((list(map(int,n_points_U))))).cuda(), torch.from_numpy(np.array(list(map(int,n_points_V)))).cuda(),mask_U,mask_V)
t_input =(tU, tV, torch.from_numpy(np.array(list(map(int,tn_points_U)))).cuda(), torch.from_numpy(np.array(list(map(int, tn_points_V)))).cuda(),tmask_U,tmask_V)
aucs=[]


for _ in tqdm_notebook(range(n_epoch)):

    net.train()

    out = net(inputs)

    torch.save(net.state_dict, 'fs_pool_authors_keywords.model')

    loss = nn.BCEWithLogitsLoss()(out, gold)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print("after",_)
    losses.append(loss.detach().item())
    net.eval()
    test_out = net(t_input)


    #aucs.append((roc_auc_score(gold,out),roc_auc_score(tgold,test_out)))


    test_loss = nn.BCEWithLogitsLoss()(test_out, tgold)
    # print("before",_)
    test_losses.append(test_loss.detach().item())
    aucs.append((_,roc_auc_score(gold.clone().cpu().detach().numpy(),nn.Sigmoid()(out).cpu().detach().numpy()),
                 roc_auc_score(tgold.clone().cpu().detach().numpy(),nn.Sigmoid()(test_out).cpu().detach().numpy())))
    
    pickle.dump({'train_losses': losses, 'test_losses': test_losses}, open('fs_pool_authors_keywords.losses.pkl', 'wb'))

import pickle
loss_dict = pickle.load(open('fs_pool_authors_keywords.losses.pkl', 'rb'))
model_dict = pickle.load(open('fs_pool_authors_keywords.model', 'rb'))

losses = loss_dict['train_losses']
test_losses = loss_dict['test_losses']

from matplotlib import pyplot as plt
n_epoch1 = len(losses)
plt.plot(range(n_epoch1), losses, label='train')
plt.plot(range(n_epoch1), test_losses, label='test')
plt.grid()
plt.legend()
plt.show()


# mse, cha, acc = torch.FloatTensor([-1, -1, -1])
# if not args.classify:
#     mse = (pred - points).pow(2).mean()
#     cha = chamfer_loss(pred, points)
#     if args.loss == 'direct':
#         loss = mse
#     elif args.loss == 'chamfer':
#         loss = cha
#     elif args.loss == 'hungarian':
#         loss = hungarian_loss(pred, points)
#     else:
#         raise NotImplementedError
# else:
#     loss = F.cross_entropy(pred, labels)
#     acc = (pred.max(dim=1)[1] == labels).float().mean()

# if train:
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# tracked_mse = tracker.update('{}_mse'.format(prefix), mse.item())
# tracked_cha = tracker.update('{}_cha'.format(prefix), cha.item())
# tracked_loss = tracker.update('{}_loss'.format(prefix), loss.item())
# tracked_acc = tracker.update('{}_acc'.format(prefix), acc.item())

# fmt = '{:.5f}'.format
# loader.set_postfix(
#     mse=fmt(tracked_mse),
#     cha=fmt(tracked_cha),
#     loss=fmt(tracked_loss),
#     acc=fmt(tracked_acc),
# )

# if args.show and not train:
#     #scatter(input_points, n_points, marker='o', transpose=args.mnist)
#     scatter(pred, n_points, marker='x', transpose=args.mnist)
#     plt.axes().set_aspect('equal', 'datalim')
#     plt.show()
'''


import argparse
from time import sleep
import pdb
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
# from src.our_modules import device, Classifier
from src.our_utils import obtain_node_embeddings, process_node_emb, get_home_path, mkdir_p, load_and_process_data, \
    get_data_path
from src.results_analyzer import plot_results
from src import train_test_sampler
from src import embedding_storer

sys.path.append(get_home_path())
from lib.hypersagnn.main import parse_args as parse_embedding_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_torch_environment():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_args():
    parser = argparse.ArgumentParser(description="CATSETMAT: Main module")

    parser.add_argument('--data_name', type=str, default='sample_mag_acm')
    parser.add_argument('--num_splits', type=int, default=5,
                        help='Number of train-test-splits / negative-samplings. Default is 15.')
    parser.add_argument('--start_split', type=int, default=0,
                        help='Start id of splits; splits go from start_split to start_split+num_splits. Default is 0.')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs. Default is 200.')
    parser.add_argument('--batch_size', type=int, default=300,
                        help='Batch size. Default is 100.')
    parser.add_argument('--model_save_split_id', type=int, default=0,
                        help='Split id for which model is to be saved. Default is 0.')
    args = parser.parse_args('')
    return args


def process_args(args):
    data_name = args.data_name

    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    return data_name, splits, num_epochs, batch_size, args.model_save_split_id

def data_modify(data):
    u_, v_, l_ = zip(*data)
    npoints_u=[len(x[x>0].tolist())for x in u_]
    npoints_v=[len(x[x>0].tolist())for x in v_]
    # pdb.set_trace()
    mask_u=torch.cat([(x>0).float().view(1,x.shape[0],1) for x in u_ ],dim=0)
    mask_v=torch.cat([(x>0).float().view(1,x.shape[0],1) for x in v_ ],dim=0)

    return u_,npoints_u,v_,npoints_v,mask_u,mask_v,l_

   
def train(model, data, globaliter=0):
    globaliter += 1
    model.train()
    U,n_points_U,V,n_points_V,mask_U,mask_V,l_=data_modify(data)
    U=torch.cat(U,dim=0).view(len(U),U[0].shape[0]).to(device)
    V=torch.cat(V,dim=0).view(len(V),V[0].shape[0]).to(device)
    gold = torch.Tensor(l_).view(-1, 1).to(device)
    inputs = (U,V,  torch.from_numpy(np.array((list(map(int,n_points_U))))).to(device), torch.from_numpy(np.array(list(map(int,n_points_V)))).to(device),mask_U.to(device),mask_V.to(device))
    # u_, v_, l_ = zip(*data)
    # xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
    # yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)

    # pdb.set_trace()
    out = model(inputs)

    # torch.save(net.state_dict, 'fs_pool_authors_keywords.model')

    loss = nn.BCEWithLogitsLoss()(out, gold).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # output, weights = model(xx, yy)
    # loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # label = output.squeeze(-1)
    # # pdb.set_trace()
    # del xx,yy,weights
    # torch.cuda.empty_cache()
    auc = roc_auc_score(l_ , out.cpu().detach().numpy())
    return loss.item(), auc,None


def test(model, data):
    model.eval()
    # u_, v_, l_ = zip(*data)

    U,n_points_U,V,n_points_V,mask_U,mask_V,l_=data_modify(data)
    U=torch.cat(U,dim=0).view(len(U),U[0].shape[0]).to(device)
    V=torch.cat(V,dim=0).view(len(V),V[0].shape[0]).to(device)
    inputs = (U,V,  torch.from_numpy(np.array((list(map(int,n_points_U))))).to(device), torch.from_numpy(np.array(list(map(int,n_points_V)))).to(device),mask_U.to(device),mask_V.to(device))

    gold = torch.Tensor(l_).view(-1, 1).to(device)
    # inputs = (U,V,  torch.from_numpy(np.array((list(map(int,n_points_U))))), torch.from_numpy(np.array(list(map(int,n_points_V)))).cuda(),mask_U.cuda(),mask_V.cuda())
    # u_, v_, l_ = zip(*data)
    # xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
    # yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
    out = model(inputs)

    # torch.save(net.state_dict, 'fs_pool_authors_keywords.model')

    loss = nn.BCEWithLogitsLoss()(out, gold).to(device)

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # output, weights = model(xx, yy)
    # loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # label = output.squeeze(-1)
    # # pdb.set_trace()
    # del xx,yy,weights
    # torch.cuda.empty_cache()
    auc = roc_auc_score(l_ , out.cpu().detach().numpy())
    

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
        A = obtain_node_embeddings(args, node_list_set, train_set, data_name, set_name, split_id, base_path, silent=silent)
    node_embedding = process_node_emb(A, node_list_set, args)
    return node_embedding


def perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path, num_epochs, batch_size, model_save_split_id):
    global criterion, optimizer
    pickled_path = os.path.join(data_path, 'processed', data_name, '{}.pkl'.format(split_id))
    train_data, test_data, U_t, V_t, node_list_U, node_list_V = load_and_process_data(pickled_path)
    base_path = home_path

    U,n_points_U,V,n_points_V,mask_U,mask_V,l_=data_modify(train_data)

    node_embedding_U = read_cache_node_embeddings(emb_args, node_list_U, U_t, data_name, 'U', split_id, base_path)
    node_embedding_V = read_cache_node_embeddings(emb_args, node_list_V, V_t, data_name, 'V', split_id, base_path)

    # pdb.set_trace()
    hidden_dim = 128
    # latent_dim = 64
    latent_dim = emb_args.dimensions



    model = EMB_LAYER(node_embedding_U,node_embedding_V,0,latent_dim+1,
              latent_dim,hidden_dim,
              set_size_U=max(n_points_U),
              set_size_V = max(n_points_V),
              skip=False,relaxed=False).to(device)

    # print(model)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    # criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1E-6)
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
    if split_id == model_save_split_id:
        torch.save(model, os.path.join(result_path, 'model_{}.mdl'.format(split_id)))
    return model
