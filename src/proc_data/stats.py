import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
from math import ceil
import multiprocessing
from pathlib import Path
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

from src.util.util import split_range_into_k_segments


def parallel_read_to_shared_dict(shared_dict, lock, directory, filenames):
    """
    """
    temp = list()
    for filename in filenames:
        file_path = Path(f'{directory}/{filename}')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)  # Read the .pkl file
        temp.append(data)
    
    pattern = r"(\d+)_(\w+)\.pkl"
    with lock:
        for filename, data in zip(filenames, temp):
            match = re.match(pattern, filename)
            if match:
                global_index, label = match.groups()
                shared_dict[(int(global_index), int(label))] = data
    
    del temp
    gc.collect()


def read_to_shared_dict(shared_dict, directory, filenames, max_workers, batch_size):
    """
    """
    lock = multiprocessing.Lock()  # Lock for synchronization

    num_batches = ceil(len(filenames) / batch_size)

    print(f"Reading {len(filenames)} files in {num_batches} batches")

    batches = split_range_into_k_segments(start=0, end=len(filenames), k=num_batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(parallel_read_to_shared_dict, shared_dict, lock, directory, filenames[batch_start:batch_end]) for batch_start, batch_end in batches]


def flip_rate(dataset_name, model_name, threshold_str):
    """
    """
    with open(f'data/{dataset_name}/{model_name}_base_pred.pkl', 'rb') as f:
        base_y_pred = pickle.load(f)

    with open(f'data/{dataset_name}/rules_{threshold_str}.pkl', 'rb') as f:
        rules_dict = pickle.load(f)
    
    rules_list = list(rules_dict.items())

    filenames = [f'{global_index}_{label}.pkl' for (global_index, label), (rule_w_target, support) in rules_list]
    directory = f'data/{dataset_name}/saved_models/{model_name}/metadata'
    
    with multiprocessing.Manager() as manager:
        metadata = manager.dict()

        read_to_shared_dict(metadata, directory, filenames, max_workers=8, batch_size=1000)
        print(f'{len(metadata)} files read into shared dictionary')

        flips = []
        for (global_index, label), (_, _, new_y_pred) in metadata.items():    
            _flips = (base_y_pred != new_y_pred).astype(int)
            flips.append(_flips.reshape(-1, 1))

        flips = np.concatenate(flips, axis=1)
        sum_flips = flips.sum(axis=1)
        no_flips = (sum_flips == 0)
        num_no_flips = no_flips.sum()

        print(f'Flip Rate: 1 - {num_no_flips}/{len(sum_flips)} = {1 - num_no_flips/len(sum_flips)}')


def model_similarity(dataset_name, model_name, threshold_str):
    """
    """
    with open(f'data/{dataset_name}/{model_name}_base_pred.pkl', 'rb') as f:
        base_y_pred = pickle.load(f)

    highest_model_sim = {i:None for i in range(len(base_y_pred))}

    with open(f'data/{dataset_name}/rules_{threshold_str}.pkl', 'rb') as f:
        rules_dict = pickle.load(f)
    
    rules_list = list(rules_dict.items())

    filenames = [f'{global_index}_{label}.pkl' for (global_index, label), (rule_w_target, support) in rules_list]
    directory = f'data/{dataset_name}/saved_models/{model_name}/metadata'
    
    with multiprocessing.Manager() as manager:
        metadata = manager.dict()

        read_to_shared_dict(metadata, directory, filenames, max_workers=8, batch_size=1000)
        print(f'{len(metadata)} files read into shared dictionary')

        for (global_index, label), (_, model_similarity, new_y_pred) in metadata.items():

            flips = (base_y_pred != new_y_pred).astype(int)
            flips_where = np.where(flips)[0]
            for j in flips_where:
                if highest_model_sim[j] is None or model_similarity > highest_model_sim[j][1]:
                    highest_model_sim[j] = ((global_index, label), model_similarity)
    

    temp_model_sims = list()
    for j in highest_model_sim.keys():
        if highest_model_sim[j] is not None:
            temp_model_sims.append(highest_model_sim[j][1])
    
    print(f'Avg Model Similarity: {np.mean(temp_model_sims)}')
    print(f'Std dev Model Similarity: {np.std(temp_model_sims)}')
    print(f'Quantiles Model Similarity: {np.quantile(temp_model_sims, [0.25, 0.5, 0.75])}')

    plt.hist(temp_model_sims, bins=np.linspace(0, 1.0, 101), edgecolor='black')
    plt.title(f' {dataset_name} {model_name} model similarities: {len(temp_model_sims)} points')
    plt.xlabel('Model Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'plots/{dataset_name}_{model_name}_model_similarities_{threshold_str}.png')
    plt.clf()


def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description="Script to process a given Process ID (PID).")
    
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--threshold_str", type=str, help="Threshold string for rules")

    return parser.parse_args()


if __name__ == '__main__':
    """
    time python -m src.proc_data.stats --dataset compas --model logreg --threshold_str freq_1000_supp_0.3
    """
    args = parse_args()

    flip_rate(args.dataset, args.model, args.threshold_str)
    model_similarity(args.dataset, args.model, args.threshold_str)
        
    

    