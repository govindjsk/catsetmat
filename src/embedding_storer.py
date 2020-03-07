import argparse
import os
import sys
from tqdm.autonotebook import tqdm
from src.our_utils import obtain_node_embeddings, get_home_path, get_default_data_params, load_and_process_data
sys.path.append(get_home_path())
from lib.hypersagnn.main import parse_args as parse_embedding_args


def parse_args():
    parser = argparse.ArgumentParser(description="CATSETMAT: Embedding Storer")

    parser.add_argument('--data_name', type=str, default='sample_mag_acm')
    parser.add_argument('--num_splits', type=int, default=15,
                        help='Number of train-test-splits / negative-samplings. Default is 15.')
    parser.add_argument('--start_split', type=int, default=0,
                        help='Start id of splits; splits go from start_split to start_split+num_splits. Default is 0.')
    args = parser.parse_args()
    return args


def process_args(args):
    data_name = args.data_name
    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    return data_name, splits


def main():
    data_name, splits = process_args(parse_args())
    data_params = get_default_data_params()
    data_params['raw_data_path'] = os.path.join(data_params['raw_data_path'], data_name)
    data_params['processed_data_path'] = os.path.join(data_params['processed_data_path'], data_name)
    home_path = get_home_path()
    base_path = home_path
    for split_id in tqdm(splits, 'Pre-storing embeddings'):
        emb_args = parse_embedding_args()
        pickled_path = os.path.join(data_params['processed_data_path'], '{}.pkl'.format(split_id))
        train_data, test_data, U_t, V_t, node_list_U, node_list_V = load_and_process_data(pickled_path)
        # This automatically stores embeddings to an external path
        _ = obtain_node_embeddings(emb_args, node_list_U, U_t, data_name, 'U', split_id, base_path, True)
        _ = obtain_node_embeddings(emb_args, node_list_V, V_t, data_name, 'V', split_id, base_path, True)


if __name__ == '__main__':
    main()
