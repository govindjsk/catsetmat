import os
import pickle
import torch
import sys
import torch.nn as nn
from sklearn.metrics import roc_auc_score, pairwise
from sklearn.utils import shuffle
from tqdm.autonotebook import tqdm

from src.link_predictor import predict_links, get_auc_scores
from src.hypersagnn_modules import Classifier as Classifier_hypersagnn
from src.our_modules import device, Classifier
from src.our_utils import obtain_node_embeddings, process_node_emb, \
    get_home_path, load_and_process_data, w2v_model
from src.node2vec import *

sys.path.append(get_home_path())
from lib.fspool.main import EMB_LAYER
from src.graphconstructor import read_graph, read_graph_cross


def data_modify(data):
    u_, v_, l_ = zip(*data)
    npoints_u = [len(x[x > 0].tolist()) for x in u_]
    npoints_v = [len(x[x > 0].tolist()) for x in v_]
    # pdb.set_trace()
    mask_u = torch.cat([(x > 0).float().view(1, x.shape[0], 1) for x in u_], dim=0)
    mask_v = torch.cat([(x > 0).float().view(1, x.shape[0], 1) for x in v_], dim=0)
    return u_, npoints_u, v_, npoints_v, mask_u, mask_v, l_


def hyp(data, max_node_u):
    u_, v_, l_ = zip(*data)
    v_new = []
    u_new = []
    for x in v_:
        x = (x[x > 0] + max_node_u + 1)
        x = (x - 1).tolist()
        v_new.append(x)
    for i in range(len(v_new)):
        x = (u_[i][u_[i] > 0])
        x = (x - 1).tolist()
        u_new.append(x)
    return u_new, v_new, l_


def hyp_hypersagnn(data, max_node_u):
    u_, v_, l_ = zip(*data)
    print(max_node_u)
    v_new = []
    u_new = []

    for x in v_:
        x = (x[x > 0] + max_node_u + 1)
        x = (x - 1).tolist()
        v_new.append(x)

    for i in range(len(v_new)):
        x = (u_[i][u_[i] > 0])
        x = (x - 1).tolist()
        x = x + v_new[i]
        u_new.append(x)

    return u_new, l_


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
    if model_name.startswith('catsetmat'):
        u_, v_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
        output, weights = model(xx, yy)
        loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
        label = output.squeeze(-1)
        # del xx,yy,weights
        # torch.cuda.empty_cache()

    # HYPERSAGNN:
    if model_name.startswith('hypersagnn'):
        u_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        output, weights = model(xx, xx)
        loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
        label = output.squeeze(-1)
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
    if model_name.startswith('catsetmat'):
        u_, v_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        yy = torch.cat(v_, dim=0).view(len(v_), v_[0].shape[0]).to(device)
        output, weights = model(xx, yy)
        loss = criterion(output, torch.from_numpy(np.array(l_)).float().to(device))
        label = output.squeeze(-1)
    # HYPERSAGNN:
    if model_name.startswith('hypersagnn'):
        u_, l_ = zip(*data)
        xx = torch.cat(u_, dim=0).view(len(u_), u_[0].shape[0]).to(device)
        output, weights = model(xx, xx)
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


def built_hyper_edges(train_data, test_data, max_node_u):
    u_train, l_train = hyp_hypersagnn(train_data, max_node_u)
    u_test, l_test = hyp_hypersagnn(test_data, max_node_u)

    len_tr = max([len(x) for x in u_train])
    len_t = max([len(x) for x in u_test])
    max_ = max([len_tr, len_t])
    U_ = []
    U_T = []
    for i in range(len(u_train)):
        y = [x + 1 for x in u_train[i]]
        y = [0] * (max_ - len(y)) + y
        U_.append(torch.from_numpy(np.array(y)).long())
    for i in range(len(u_test)):
        y = [x + 1 for x in u_test[i]]
        y = [0] * (max_ - len(y)) + y
        U_T.append(torch.from_numpy(np.array(y)).long())
    train_data = list(zip(U_, l_train))
    test_data = list(zip(U_T, l_test))
    return train_data, test_data, u_train, l_train


def perform_experiment(emb_args, home_path, data_path, data_name, split_id, result_path,
                       num_epochs, batch_size, model_save_split_id, model_name, lr):
    global criterion, optimizer
    pickled_path = os.path.join(data_path, 'processed', data_name, '{}.pkl'.format(split_id))
    train_data, test_data, U_t, V_t, node_list_U, node_list_V = load_and_process_data(pickled_path)
    base_path = home_path
    # print(pickled_path)
    # pdb.set_trace()
    node_embedding_U = read_cache_node_embeddings(emb_args, node_list_U, U_t, data_name, 'U', split_id, base_path)
    node_embedding_V = read_cache_node_embeddings(emb_args, node_list_V, V_t, data_name, 'V', split_id, base_path)
    if model_name == 'n2v':
        max_node_u = max(node_list_U)
        u_train, v_train, l_train = hyp(train_data, max_node_u)
        u_test, v_test, l_test = hyp(test_data, max_node_u)
        node_list_v = [(x + max_node_u + 1) for x in node_list_V]
        node_list = node_list_U + node_list_v
        index = [idx for idx, val in enumerate(l_train) if val != 0]
        U_train = [u_train[x] for x in index]
        V_train = [v_train[x] for x in index]

        # for only cross graph from bipartite hypergraph
        g_train_c = read_graph_cross(node_list, U_train, V_train)

        # for full graph form bipartite hypergraph
        for i in range(len(U_train)):
            U_train[i] += V_train[i]
        g_train_f = read_graph(node_list, U_train)
        g_n2v_c = Graph(g_train_c, 0, emb_args.p, emb_args.q)
        g_n2v_f = Graph(g_train_f, 0, emb_args.p, emb_args.q)
        g_n2v_c.preprocess_transition_probs()
        g_n2v_f.preprocess_transition_probs()
        walks_c = g_n2v_c.simulate_walks(emb_args.num_walks, emb_args.walk_length)
        walks_f = g_n2v_f.simulate_walks(emb_args.num_walks, emb_args.walk_length)
        A_C = np.array(w2v_model(walks_c, emb_args, node_list, True))
        A_F = np.array(w2v_model(walks_f, emb_args, node_list, True))
        result_c = pairwise.cosine_similarity(A_C[node_list_U], A_C[node_list_v], dense_output=True)
        result_f = pairwise.cosine_similarity(A_F[node_list_U], A_F[node_list_v], dense_output=True)
        auc_result_minc = []
        auc_result_minf = []
        auc_result_meanc = []
        auc_result_meanf = []

        for i in range(len(u_test)):
            probf = []
            probc = []
            for j in u_test[i]:
                for k in v_test[i]:
                    probc.append(result_c[j, k - max_node_u - 1])
                    probf.append(result_f[j, k - max_node_u - 1])
            auc_result_minc.append(min(probc))
            auc_result_meanc.append(sum(probc) / len(probc))
            auc_result_minf.append(min(probf))
            auc_result_meanf.append(sum(probf) / len(probf))
        # t.set_description("AUC test:min_cross {} , mean_cross {} ,min_full {} ,mean_full {}".\
        #                       format(round(roc_auc_score(l_test,auc_result_minc), 4),
        #                              round(roc_auc_score(l_test,auc_result_meanc), 4),
        #                              round(roc_auc_score(l_test,auc_result_minf), 4),
        #                              round(roc_auc_score(l_test,auc_result_meanf), 4)))
        auc = {"min_cross": round(roc_auc_score(l_test, auc_result_minc), 4),
               "mean_cross": round(roc_auc_score(l_test, auc_result_meanc), 4),
               "min_full": round(roc_auc_score(l_test, auc_result_minf), 4),
               "mean_full": round(roc_auc_score(l_test, auc_result_meanf), 4)}
        print('AUC', auc)
        loss = None
        model = [A_C, A_F]
    elif model_name == 'lp':
        train_scores_df, test_scores_df = predict_links(train_data, test_data, U_t, V_t, node_list_U, node_list_V)
        train_auc_scores = get_auc_scores(train_scores_df)
        test_auc_scores = get_auc_scores(test_scores_df)
        auc = test_auc_scores
        print('AUC', auc)
        loss = None
        model = [train_auc_scores, test_auc_scores]
    else:
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
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1E-6)

        # CATSETMAT:
        if model_name.startswith('catsetmat'):
            if '-' not in model_name:
                model_name = 'catsetmat-x'
            latent_dim = emb_args.dimensions
            # print("catset",latent_dim,lr)
            model_type = model_name.split('-')[-1]
            model = Classifier(n_head=8,
                               d_model=latent_dim,
                               d_k=int(latent_dim / 4) if latent_dim >= 4 else 1,
                               d_v=int(latent_dim / 4) if latent_dim >= 4 else 1,
                               node_embedding1=node_embedding_U,
                               node_embedding2=node_embedding_V,
                               diag_mask=False,
                               bottle_neck=latent_dim,
                               cross_attn_type=model_type).to(device).to(device)
            criterion = nn.BCELoss().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-6)

        # HYPER-SAGNN:
        if model_name.startswith('hypersagnn'):
            if '-' not in model_name:
                model_name = 'hypersagnn-sum'
            hypersagnn_mode = model_name.split('-')[-1]
            max_node_u = max(node_list_U)
            node_list_v = [(x + max_node_u + 1) for x in node_list_V]
            train_data, test_data, u_train, l_train = built_hyper_edges(train_data, test_data, max_node_u)
            node_list = node_list_U + node_list_v
            index = [idx for idx, val in enumerate(l_train) if val != 0]
            U_train = [np.array(u_train[x], dtype=int) for x in index]

            node_embedding = read_cache_node_embeddings(emb_args, node_list, np.array(U_train), data_name, 'hyp_U',
                                                        split_id,
                                                        base_path)
            latent_dim = emb_args.dimensions
            model = Classifier_hypersagnn(n_head=8,
                                          d_model=latent_dim,
                                          d_k=int(latent_dim / 4) if latent_dim >= 4 else 1,
                                          d_v=int(latent_dim / 4) if latent_dim >= 4 else 1,
                                          node_embedding1=node_embedding,
                                          diag_mask=False,
                                          bottle_neck=latent_dim,
                                          hypersagnn_mode=hypersagnn_mode).to(device).to(device)
            criterion = nn.BCELoss().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-6)
        # pytorch_total_params = sum(p.numel() for p in model.parameters())
        # print(pytorch_total_params)
        loss = []
        auc = []
        t = tqdm(range(num_epochs), 'Split id {} '.format(split_id))
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
            t.set_description("Split id {}; AUC train: {}, test: {}".format(split_id,
                                                                            round(train_auc, 4),
                                                                            round(test_auc, 4)))
            t.refresh()
            # print('({}/{})'.format(round(train_auc, 4), round(test_auc, 4)), end=' ')
            # print("epoch {} :train loss {} and auc {}: test loss {} and auc {} : ".format(i, train_loss_, train_auc,
            #                                                                               test_loss_, test_auc))
    Results = {"AUC": auc, "loss": loss}
    pickle.dump(Results, open(os.path.join(result_path, '{}_{}.pkl'.format(model_name, split_id)), 'wb'))
    # pickle.dump([], open(os.path.join(result_path,
    #                                   '{}_{}{}.pkl'.format(model_name,
    #                                                        split_id,str(pytorch_total_params))), 'wb'))
    if split_id == model_save_split_id:
        torch.save(model, os.path.join(result_path, 'model_{}_{}.mdl'.format(model_name, split_id)))
    return Results
