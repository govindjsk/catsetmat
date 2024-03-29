import errno
import multiprocessing
import numpy as np
import os
import pickle
import time
import torch.nn as nn
import torch
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler


def get_home_path():
    # return '/content/drive/My Drive/projects/textual_analysis_email/catsetmat'
    # return "/content/drive/My Drive/textual_analysis_email/catsetmat"
    # return '/home/govinds/repos/catsetmat'
    # return "/content/drive/My Drive/repos/govind_swyam/catsetmat"
    # return "/home2/e1-313-15477/govind/repos/catsetmat"
    # return "/home/swyamsingh/repos/catsetmat"
    return "C:\\cygwin64\\home\\Nidhi\\repos\\catsetmat"


sys.path.append(get_home_path())

from lib.hypersagnn.Modules import Wrap_Embedding
from lib.hypersagnn.random_walk_hyper import random_walk_hyper
from lib.hypersagnn.utils import walkpath2str
from src.our_modules import device


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pad_zeros_np(x, max_size):
    # Adding 1 as well to differentiate between padded zeros and others.
    return np.concatenate((x + 1, np.zeros((max_size - x.shape[0],), dtype=int)))


def load_and_process_data(pickled_path):
    data = pickle.load(open(pickled_path, 'rb'))
    train_data, test_data, max_he_U, max_he_V, node_list_U, node_list_V = \
        [data[x] for x in ["train_data", "test_data", "max_length_u",
                           "max_length_v", 'node_list_U', 'node_list_V']]
    U_train_hes, V_train_hes, label_train = zip(*train_data)
    U_test_hes, V_test_hes, label_test = zip(*test_data)
    U_train_hes_tensor = []
    V_train_hes_tensor = []
    for i in range(len(U_train_hes)):
        U_train_hes_tensor.append(torch.from_numpy(pad_zeros_np(U_train_hes[i], max_he_U)).long())
        V_train_hes_tensor.append(torch.from_numpy(pad_zeros_np(V_train_hes[i], max_he_V)).long())

    U_test_hes_tensor = []
    V_test_hes_tensor = []
    for i in range(len(U_test_hes)):
        U_test_hes_tensor.append(torch.from_numpy(pad_zeros_np(U_test_hes[i], max_he_U)).long())
        V_test_hes_tensor.append(torch.from_numpy(pad_zeros_np(V_test_hes[i], max_he_V)).long())

    train_data = list(zip(U_train_hes_tensor, V_train_hes_tensor, label_train))
    test_data = list(zip(U_test_hes_tensor, V_test_hes_tensor, label_test))

    # Converting list of hyperedges to a set to remove redundancy
    U_train_hes = np.array(list(map(lambda x: np.array(list(x)), set(map(frozenset, U_train_hes)))))
    V_train_hes = np.array(list(map(lambda x: np.array(list(x)), set(map(frozenset, V_train_hes)))))
    return train_data, test_data, U_train_hes, V_train_hes, node_list_U, node_list_V


def process_node_emb(A, node_list, args):
    A = StandardScaler().fit_transform(A)
    A = np.concatenate((np.zeros((1, A.shape[-1]), dtype='float32'), A), axis=0)
    A = A.astype('float32')
    A = torch.tensor(A).to(device)
    # print(A.shape)
    node_embedding = Wrap_Embedding(int(len(node_list) + 1), args.dimensions, scale_grad_by_freq=False, padding_idx=0,
                                    sparse=False)
    node_embedding.weight = nn.Parameter(A)
    return node_embedding


def w2v_model(walks, emb_args, node_list, silent=False):
    start = time.time()
    if os.name != 'nt':
        split_num = 20
        pool = ProcessPoolExecutor(max_workers=split_num)
        walks = np.array_split(walks, split_num)
        if not silent:
            print("Start turning path to strs")
        process_list = []
        for walk in walks:
            process_list.append(pool.submit(walkpath2str, walk, silent))
        results = []
        for p in as_completed(process_list):
            results += p.result()
        pool.shutdown(wait=True)
        walks = results
    else:
        walks = walkpath2str(walks, silent)
    if not silent:
        print(
            "Finishing Loading and processing %.2f s" %
            (time.time() - start))
        print("Start Word2vec")
        print("num cpu cores", multiprocessing.cpu_count())
    w2v = Word2Vec(
        walks,
        size=emb_args.dimensions,
        window=emb_args.window_size,
        min_count=0,
        sg=1,
        iter=1,
        workers=multiprocessing.cpu_count())
    wv = w2v.wv
    A = [wv[str(i)] for i in node_list]
    A = np.array(A)
    return A

def obtain_node_embeddings(args, node_list, hyperedges, data_name, set_name, split_id, base_path, silent=False):
    walk_path = random_walk_hyper(args, node_list, hyperedges, data_name, set_name, split_id, base_path, silent)
    walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
    A = w2v_model(walks, args, node_list, silent)
    emb_base = os.path.join(base_path, 'walks/embeddings/{}'.format(
        data_name))
    mkdir_p(emb_base)
    emb_path = os.path.join(emb_base, '{}_wv_{}__{}{}{}.npy'.format(
        data_name, args.dimensions, data_name, split_id, set_name))
    if not silent:
        print('Saving embeddings to {}'.format(emb_path))
    np.save(emb_path, A)
    return A


def get_default_data_params(data_path=None):
    if not data_path:
        data_path = get_data_path()
    data_params = {'raw_data_path': os.path.join(data_path, 'raw'),
                   'r_label_file': 'id_p_map.txt',
                   'u_label_file': 'id_a_map.txt',
                   'v_label_file': 'id_k_map.txt',
                   'r_u_list_file': 'p_a_list_train.txt',
                   'r_v_list_file': 'p_k_list_train.txt',
                   'processed_data_path': os.path.join(data_path, 'processed')}
    return data_params


def get_data_path():
    # return '/content/drive/My Drive/projects/textual_analysis_email/catsetmat/data'
    # return '/home/govinds/repos/catsetmat/data'
    # return '/content/drive/My Drive/textual_analysis_email/catsetmat/data'
    # return "/content/drive/My Drive/repos/govind_swyam/catsetmat/data"
    # return "/home2/e1-313-15477/govind/repos/catsetmat/data"
    # return "/home/swyamsingh/repos/catsetmat/data"
    return "C:\\cygwin64\\home\\Nidhi\\repos\\catsetmat\\data"


def main():
    pass


if __name__ == '__main__':
    main()
