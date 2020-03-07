import argparse
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

from .our_utils import mkdir_p, get_default_data_params
from .data_reader import load_bipartite_hypergraph, get_neg_samp


def parse_args():
    parser = argparse.ArgumentParser(description="CATSETMAT: Train-Test Sampler")

    parser.add_argument('--data_name', type=str, default='sample_mag_acm')
    parser.add_argument('--num_splits', type=int, default=15,
                        help='Number of train-test-splits / negative-samplings. Default is 15.')
    parser.add_argument('--start_split', type=int, default=0,
                        help='Start id of splits; splits go from start_split to start_split+num_splits. Default is 0.')
    parser.add_argument('--neg_factor', type=int, default=5,
                        help='Negative sampling factor. Default is 5.')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of samples to use while testing. Default is 0.2')
    args = parser.parse_args()
    return args


def process_args(args):
    data_name = args.data_name
    neg_factor = args.neg_factor
    test_ratio = args.test_ratio

    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    return data_name, splits, neg_factor, test_ratio


def split_data(data, labels, test_ratio=0.2):
    labels = np.array(labels)
    n = len(data) if type(data) == list else data.shape[0]
    assert n == labels.shape[0], 'Inconsistent!'
    pos_ids = list(set(labels.nonzero()[0]))
    neg_ids = list(set(range(n)).difference(pos_ids))
    if type(data) == list:
        pos_data = [data[i] for i in pos_ids]
        neg_data = [data[i] for i in neg_ids]
    else:
        pos_data = data[pos_ids]
        neg_data = data[neg_ids]
    train_pos, test_pos = train_test_split(pos_data, test_size=test_ratio)
    train_neg, test_neg = train_test_split(neg_data, test_size=test_ratio)
    if type(data) == list:
        train_data = train_pos + train_neg
        test_data = test_pos + test_neg
    else:
        train_data = np.concatenate([train_pos, train_neg])
        test_data = np.concatenate([test_pos, test_neg])
    train_labels = np.concatenate([[1] * len(train_pos), [0] * len(train_neg)])
    test_labels = np.concatenate([[1] * len(test_pos), [0] * len(test_neg)])
    return train_data, test_data, train_labels, test_labels


def prepare_data(U, V, neg_U, neg_V):
    node_list_U = []
    U_ = []

    U = U + list(neg_U)
    V = V + list(neg_V)
    for u in (U):
        node_list_U += u
        U_.append(np.asarray(u))
    node_list_U = list(set(node_list_U))

    node_list_V = []
    V_ = []
    for v in V:
        node_list_V += v
        V_.append(np.asarray(v))
    node_list_V = list(set(node_list_V))

    labels = [1] * (len(U) - len(neg_U)) + [0] * len(neg_U)

    max_he_U = max(map(len, U))
    max_he_V = max(map(len, V))

    data = list(zip(U_, V_))
    return data, labels, max_he_U, max_he_V, node_list_U, node_list_V


def main():
    data_name, splits, neg_factor, test_ratio = process_args(parse_args())
    data_params = get_default_data_params()
    data_params['raw_data_path'] = os.path.join(data_params['raw_data_path'], data_name)
    data_params['processed_data_path'] = os.path.join(data_params['processed_data_path'], data_name)
    U, V = load_bipartite_hypergraph(data_params)

    for iteration in tqdm(splits, 'Creating splits'):
        neg_U, neg_V = get_neg_samp(U, V, num_neg=len(U) * neg_factor)
        data, labels, max_he_U, max_he_V, node_list_U, node_list_V = prepare_data(U, V, neg_U, neg_V)

        train_data, test_data, train_labels, test_labels = split_data(data, labels, test_ratio=test_ratio)
        train_data = [(x[0], x[1], l) for x, l in zip(train_data, train_labels)]
        test_data = [(x[0], x[1], l) for x, l in zip(test_data, test_labels)]

        data = {"train_data": train_data,
                "test_data": test_data,
                "max_length_u": max_he_U,
                "max_length_v": max_he_V,
                'node_list_U': node_list_U,
                'node_list_V': node_list_V}
        mkdir_p(data_params['processed_data_path'])
        pickle.dump(data, open(os.path.join(data_params['processed_data_path'],
                                            '{}.pkl'.format(iteration)), 'wb'))


if __name__ == '__main__':
    main()
