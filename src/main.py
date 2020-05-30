import argparse
import os
import torch
import sys
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
from src.our_utils import get_home_path, mkdir_p, get_data_path
from src.results_analyzer import plot_results_by_max
sys.path.append(get_home_path())
from lib.hypersagnn.main import parse_args as parse_embedding_args
from src.experimenter import perform_experiment
from src.our_modules import device
from multiprocessing import set_start_method

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
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
    args = parser.parse_args()
    return args


def process_args(args):
    data_name = args.data_name
    num_splits = args.num_splits
    start_split = args.start_split
    splits = range(start_split, start_split + num_splits)
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr=args.lr
    return data_name, splits, num_epochs, batch_size, args.model_save_split_id, args.dim, args.model_name,lr


def main():
    parallel_version = False
    set_torch_environment()
    data_name, splits, num_epochs, batch_size, model_save_split_id, dim, model_name, lr = process_args(parse_args())
    emb_args = parse_embedding_args()
    emb_args.dimensions = dim
    home_path = get_home_path()
    data_path = get_data_path()

    result_path = os.path.join(home_path, 'results', data_name, '_tuning_dim'+str(dim)+'_learning_rate'+str(lr))
    # result_path = os.path.join(home_path, 'results', data_name, 'res')

    mkdir_p(result_path)
    if parallel_version:
        num_splits = len(splits)
        max_workers = min(num_splits, multiprocessing.cpu_count())
        pool = ProcessPoolExecutor(max_workers=max_workers)
        process_list = []
        for split_id in splits:
            process_list.append(pool.submit(perform_experiment, emb_args, home_path, data_path, data_name,
                                            split_id, result_path, num_epochs, batch_size,
                                            model_save_split_id, model_name, lr))
            print('{} of {} processes scheduled'.format(len(process_list), num_splits))
        results_list = []
        for p in as_completed(process_list):
            results_list.append(p.result())
            print('{} of {} processes completed'.format(len(results_list), len(process_list)))
        pool.shutdown(wait=True)
    else:
        results_list = []
        for i, split_id in enumerate(splits):
            print('------- SPLIT#{} ({} of {}) -------'.format(split_id, i, len(splits)))
            results = perform_experiment(emb_args, home_path, data_path, data_name,
                                         split_id, result_path, num_epochs, batch_size,
                                         model_save_split_id, model_name, lr)
            results_list.append(results)
    # plot_results(splits, result_path, model_name)
    plot_results_by_max(splits,result_path,model_name,dim,lr)


if __name__ == '__main__':
    main()
