import argparse
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
    return "/content/drive/My Drive/textual_analysis_email/catsetmat"
    # return '/home/govinds/repos/catsetmat'

sys.path.append(get_home_path())

from lib.hypersagnn.Modules import Wrap_Embedding
from lib.hypersagnn.random_walk_hyper import random_walk_hyper
from lib.hypersagnn.utils import walkpath2str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    U_t, V_t, label_t = zip(*train_data)
    U_tes, V_tes, label_tes = zip(*test_data)
    U__ = []
    V__ = []
    for i in range(len(U_t)):
        U__.append(torch.from_numpy(pad_zeros_np(U_t[i], max_he_U)))
        V__.append(torch.from_numpy(pad_zeros_np(V_t[i], max_he_V)))

    U__t = []
    V__t = []
    for i in range(len(U_tes)):
        U__t.append(torch.from_numpy(pad_zeros_np(U_tes[i], max_he_U)))
        V__t.append(torch.from_numpy(pad_zeros_np(V_tes[i], max_he_V)))

    train_data = list(zip(U__, V__, label_t))
    test_data = list(zip(U__t, V__t, label_tes))

    # Converting list of hyperedges to a set to remove redundancy
    U_t = np.array(list(map(lambda x: np.array(list(x)), set(map(frozenset, U_t)))))
    V_t = np.array(list(map(lambda x: np.array(list(x)), set(map(frozenset, V_t)))))
    return train_data, test_data, U_t, V_t, node_list_U, node_list_V


# def parse_args():
#     # Parses the node2vec arguments.
#     parser = argparse.ArgumentParser(description="Run node2vec.")
#
#     parser.add_argument('--data', type=str, default=data_name)
#     parser.add_argument('--TRY', action='store_true')
#     parser.add_argument('--FILTER', action='store_true')
#     parser.add_argument('--grid', type=str, default='')
#     parser.add_argument('--remark', type=str, default='')
#
#     parser.add_argument('--random-walk', action='store_true')
#
#     parser.add_argument('--dimensions', type=int, default=64,
#                         help='Number of dimensions. Default is 64.')
#
#     parser.add_argument('-l', '--walk-length', type=int, default=40,
#                         help='Length of walk per source. Default is 40.')
#
#     parser.add_argument('-r', '--num-walks', type=int, default=10,
#                         help='Number of walks per source. Default is 10.')
#
#     parser.add_argument('-k', '--window-size', type=int, default=10,
#                         help='Context size for optimization. Default is 10.')
#
#     parser.add_argument('-i', '--iter', default=1, type=int,
#                         help='Number of epochs in SGD')
#
#     parser.add_argument('--workers', type=int, default=8,
#                         help='Number of parallel workers. Default is 8.')
#
#     parser.add_argument('--p', type=float, default=2,
#                         help='Return hyperparameter. Default is 1.')
#
#     parser.add_argument('--q', type=float, default=0.25,
#                         help='Inout hyperparameter. Default is 1.')
#
#     parser.add_argument('-a', '--alpha', type=float, default=0.0,
#                         help='The weight of random walk -skip-gram loss. Default is ')
#     parser.add_argument('--rw', type=float, default=0.01,
#                         help='The weight of reconstruction of adjacency matrix loss. Default is ')
#     parser.add_argument('-w', '--walk', type=str, default='',
#                         help='The walk type, empty stands for normal rw')
#     parser.add_argument('-d', '--diag', type=str, default='True',
#                         help='Use the diag mask or not')
#     parser.add_argument(
#         '-f',
#         '--feature',
#         type=str,
#         default='walk',
#         help='Features used in the first step')
#
#     args = parser.parse_args()
#
#     if not args.random_walk:
#         args.model_name = 'model_no_randomwalk'
#         args.epoch = 25
#     else:
#         args.model_name = 'model_{}_'.format(args.data)
#         args.epoch = 25
#     if args.TRY:
#         args.model_name = 'try' + args.model_name
#         if not args.random_walk:
#             args.epoch = 5
#         else:
#             args.epoch = 1
#     # args.epoch = 1
#     args.model_name += args.remark
#     # print(args.model_name)
#
#     args.save_path = os.path.join(
#         '../checkpoints/', args.data, args.model_name)
#     if not os.path.exists(args.save_path):
#         os.makedirs(args.save_path)
#     return args


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


def obtain_node_embeddings(args, node_list, hyperedges, data_name, set_name, split_id, base_path, silent=False):
    walk_path = random_walk_hyper(args, node_list, hyperedges, data_name, set_name, split_id, base_path, silent)
    walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
    start = time.time()
    split_num = 20
    pool = ProcessPoolExecutor(max_workers=split_num)
    process_list = []
    walks = np.array_split(walks, split_num)

    result = []
    if not silent:
        print("Start turning path to strs")
    for walk in walks:
        process_list.append(pool.submit(walkpath2str, walk, silent))

    for p in as_completed(process_list):
        result += p.result()

    pool.shutdown(wait=True)

    walks = result
    if not silent:
        print(
            "Finishing Loading and processing %.2f s" %
            (time.time() - start))
        print("Start Word2vec")
        print("num cpu cores", multiprocessing.cpu_count())
    w2v = Word2Vec(
        walks,
        size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        iter=1,
        workers=multiprocessing.cpu_count())
    wv = w2v.wv
    A = [wv[str(i)] for i in node_list]
    A = np.array(A)
    emb_base = os.path.join(base_path, 'walks/embeddings/{}'.format(
        data_name))
    mkdir_p(emb_base)
    emb_path = os.path.join(emb_base, '{}_wv_{}__{}{}{}.npy'.format(
        data_name, args.dimensions, data_name, split_id, set_name))
    if not silent:
        print('Saving embeddings to {}'.format(emb_path))
    np.save(emb_path, A)
    # np.save("../%s_wv_%d_%s_%s.npy" %
    #         (args.data, args.dimensions, args.walk, set_name), A)
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
    return '/content/drive/My Drive/textual_analysis_email/catsetmat/data'


def main():
    pass


if __name__ == '__main__':
    main()
