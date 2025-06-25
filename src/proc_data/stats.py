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
import heapq

np.random.seed(42)

from src.proc_data.util import load_dataset
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


def top_k_model_similarity(dataset_name, model_name, threshold_str, k):
    """
    """
    with open(f'data/{dataset_name}/{model_name}_base_pred.pkl', 'rb') as f:
        base_y_pred = pickle.load(f)

    topk_model_sims = {i:[] for i in range(len(base_y_pred))}

    with open(f'data/{dataset_name}/rules_{threshold_str}.pkl', 'rb') as f:
        rules_dict = pickle.load(f)
    
    rules_list = list(rules_dict.items())

    filenames = [f'{global_index}_{label}.pkl' for (global_index, label), (rule_w_target, support) in rules_list]
    directory = f'data/{dataset_name}/saved_models/{model_name}/metadata'
    
    with multiprocessing.Manager() as manager:
        metadata = manager.dict()

        read_to_shared_dict(metadata, directory, filenames, max_workers=8, batch_size=1000)
        print(f'{len(metadata)} files read into shared dictionary')

        for (global_index, label), (accuracy, model_similarity, new_y_pred) in metadata.items():
            _, support = rules_dict[(global_index, label)]

            flips = (base_y_pred != new_y_pred).astype(int)
            flips_where = np.where(flips)[0]
            for j in flips_where:
                min_heap = topk_model_sims[j]
                entry = (model_similarity, accuracy, support)
                if len(min_heap) < k:
                    heapq.heappush(min_heap, entry)
                else:
                    heapq.heappushpop(min_heap, entry)

    for i in topk_model_sims.keys():
        topk_model_sims[i] = sorted(topk_model_sims[i], reverse=True)

    temp_model_sims = list()
    for j in topk_model_sims.keys():
        temp_model_sims.extend([entry[0] for entry in topk_model_sims[j]])
    
    temp_accuracies = list()
    for j in topk_model_sims.keys():
        temp_accuracies.extend([entry[1] for entry in topk_model_sims[j]])
    
    temp_supports = list()
    for j in topk_model_sims.keys():
        temp_supports.extend([entry[2] for entry in topk_model_sims[j]])

    print(f'Avg Model Similarity: {np.mean(temp_model_sims)}')
    print(f'Std dev Model Similarity: {np.std(temp_model_sims)}')
    print(f'Quantiles Model Similarity: {np.quantile(temp_model_sims, [0.25, 0.5, 0.75])}')

    if dataset_name == 'adult':
        _dataset_name = "Adult Income"
    elif dataset_name == 'so':
        _dataset_name = "Stackoverflow Survey"
    elif dataset_name == 'compas':
        _dataset_name = "Compas"

    if model_name == 'linsvc':
        _model_name = "SVM"
    elif model_name == 'logreg':
        _model_name = "Logistic Regression"
    elif model_name == 'nn':
        _model_name = "Neural Network"
    elif model_name == 'adaboost':
        _model_name = "Adaboost"
    elif model_name == 'rf':
        _model_name = "Random Forest"

    plt.figure(figsize=(5, 3))
    plt.hist(temp_model_sims, bins=np.linspace(0.6, 1.0, 30), color='skyblue', edgecolor='black')
    plt.title(f'{_dataset_name} | {_model_name}', fontsize=16)
    plt.xlabel('Model Similarity', fontsize=14)
    plt.xticks(fontsize=12)
    if dataset_name == 'adult':
        plt.ylabel('Frequency', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'plots/{dataset_name}_{model_name}_topk_model_similarities_{threshold_str}.png', bbox_inches='tight', dpi=400)
    plt.clf()

    df, _, target, _, test_indices_no_duplicate = load_dataset(dataset_name)    
    accuracy = (base_y_pred == df[target].values[test_indices_no_duplicate]).astype(int)

    plt.figure(figsize=(5, 3))
    plt.hist(temp_accuracies, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{_dataset_name} | {_model_name}', fontsize=16)
    plt.xlabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=12)
    if dataset_name == 'adult':
        plt.ylabel('Frequency', fontsize=14)
    plt.yticks(fontsize=12)
    plt.axvline(x=accuracy.mean(), color='red', linestyle='dashed', linewidth=2, label='Base Model Accuracy')
    plt.legend(fontsize=12)
    plt.savefig(f'plots/{dataset_name}_{model_name}_topk_accuracy_{threshold_str}.png', bbox_inches='tight', dpi=400)
    plt.clf()

    plt.figure(figsize=(5, 3))
    plt.hist(temp_supports, bins=np.linspace(0.0, 0.3, 30), color='skyblue', edgecolor='black')
    plt.title(f'{_dataset_name} | {_model_name}', fontsize=16)
    plt.xlabel('Support', fontsize=14)
    plt.xticks(fontsize=12)
    if dataset_name == 'adult':
        plt.ylabel('Frequency', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'plots/{dataset_name}_{model_name}_topk_support_{threshold_str}.png', bbox_inches='tight', dpi=400)
    plt.clf()

    temp_flip_rates = [(1-x) for x in temp_model_sims]

    plt.figure(figsize=(5, 3))
    plt.scatter(temp_supports, temp_flip_rates, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'{_dataset_name} | {_model_name}', fontsize=16)
    plt.xlabel('Support', fontsize=14)
    plt.xticks(fontsize=12)
    if dataset_name == 'adult':
        plt.ylabel('Flip rate', fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'plots/{dataset_name}_{model_name}_topk_scatter_{threshold_str}.png', bbox_inches='tight', dpi=400)
    plt.clf()


def plot_model_similarities(dataset_name, model_name, threshold_str):
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

        model_sims = []
        for (global_index, label), (_, model_similarity, new_y_pred) in metadata.items():
            model_sims.append(model_similarity)
    
    plt.hist(model_sims, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{dataset_name} {model_name} model similarities')
    plt.xlabel('Model Similarity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'plots/{dataset_name}_{model_name}_all_model_similarities_{threshold_str}.png', bbox_inches='tight', dpi=400)
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
    time python -m src.proc_data.stats --dataset compas --model nn --threshold_str freq_1000_supp_0.3
    """
    # args = parse_args()

    # flip_rate(args.dataset, args.model, args.threshold_str)
    # model_similarity(args.dataset, args.model, args.threshold_str)
    # plot_model_similarities(args.dataset, args.model, args.threshold_str)
    
    dataset_names = ["adult", "so", "compas"]
    # dataset_names = ["adult"]  # For testing, only use adult dataset
    model_names = ["linsvc", "logreg", "nn", "adaboost", "rf"]

    # dataset_names = ["compas"]  # For testing, only use compas dataset
    # model_names = ["nn"]  # For testing, only use neural network model

    for dataset_name in dataset_names:
        if dataset_name == "adult":
            threshold_str = "freq_2000_supp_0.3"
        else:
            threshold_str = "freq_1000_supp_0.3"
        for model_name in model_names:
            print(f"Processing {dataset_name} {model_name} with threshold {threshold_str}")
            # flip_rate(dataset_name, model_name, threshold_str)
            # model_similarity(dataset_name, model_name, threshold_str)
            # plot_model_similarities(dataset_name, model_name, threshold_str)

            top_k_model_similarity(dataset_name, model_name, threshold_str, k=5)
    

    