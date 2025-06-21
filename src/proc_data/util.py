from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import gc
from math import ceil
import re
import numpy as np
from multiprocessing import shared_memory as sm
from src.util.util import split_range_into_k_segments
import pandas as pd
import pickle
from collections import defaultdict
from src.util.util import X_y_split
import multiprocessing

np.random.seed(42)


def build_rules_cg(row, profiles, dataset_name, index, muli_value, train_set_raw, multi_val_col_metadata):
    """
    """
    rules = list() # list[(tuple(tuple(tuple(str, str)))]
    if not muli_value:
        for profile in profiles: # profiles in a list[tuple[tuple[str]]]
            rule = tuple([tuple([(c, row.iloc[0][c]) for c in conjunction]) for conjunction in profile]) # tuple(tuple(tuple(str, str)))
            rules.append(rule)
    else:
        multi_val_col = multi_val_col_metadata.keys()
        for profile in profiles: # profiles in a list[tuple[tuple[str]]]
            rule = list() # list(tuple(tuple(str, str)))
            for conjunction in profile: # conjunction in a tuple[str]
                temp = list() # list(tuple(str, str))
                for c in conjunction: # c in a str
                    if c not in multi_val_col:
                        temp.append((c, row.iloc[0][c]))
                    else:
                        vals = train_set_raw.iloc[index][c].split(';')
                        for val in vals:
                            temp.append((f'{c}_{multi_val_col_metadata[c][val]}', 1))
                rule.append(tuple(temp))
            rules.append(tuple(rule))

    with open(f'data/{dataset_name}/rules/{index}.pkl', 'wb') as f:
        pickle.dump(rules, f)
    

def build_rules_for_train_set_cg(train_set, profiles, dataset_name, muli_value=False, train_set_raw=None, multi_val_col_metadata=None):
    """
    """
    with ProcessPoolExecutor(56) as executor:
        futures = {executor.submit(build_rules_cg, train_set.iloc[[i]], profiles, dataset_name, i, muli_value, train_set_raw, multi_val_col_metadata): i for i in range(len(train_set))}
        wait(futures)

    #     sorted_futures = sorted(futures.items(), key=lambda x: x[1])
    #     sorted_results = [future.result() for future, i in sorted_futures]
    # return sorted_results

    # return list(map(lambda i: build_rules_cg(row=train_set.iloc[[i]], profiles=profiles), range(len(train_set))))


def parallel_sort_rules_cg(dataset_name, index):
    """
    """
    with open(f'data/{dataset_name}/rules/{index}.pkl', 'rb') as f:
        rules = pickle.load(f)
    
    sorted_rules = list() # list(tuple(tuple(tuple(str, str)))))
    for rule in rules:
        # Sorting the conjunctions. Disjunctions are already sorted during the profile generation.
        temp_rule = [sorted(conjunction, key=lambda x: x[0]) for conjunction in rule]

        def first_string_key(inner_list):
            return tuple(x[0] for x in inner_list)
        
        sorted_rule = sorted(temp_rule, key=first_string_key)
        sorted_rule = tuple([tuple(conjunction) for conjunction in sorted_rule])  # Convert back to tuple for immutability
        sorted_rules.append(sorted_rule)
        # print(sorted_rule)
    
    with open(f'data/{dataset_name}/sorted_rules/{index}.pkl', 'wb') as f:
        pickle.dump(sorted_rules, f)


def sort_rules_cg(dataset_name, train_set):
    """
    """
    with ProcessPoolExecutor(56) as executor:
        futures = {executor.submit(parallel_sort_rules_cg, dataset_name, i): i for i in range(len(train_set))}
        wait(futures)


def read_rules_cg(dataset_name, index):
    """
    TODO: Not used anymore.
    """
    file_path = f'data/{dataset_name}/sorted_rules/{index}.pkl'
    try:
        with open(file_path, 'rb') as f:
            sorted_rules = pickle.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sorted_rules = None
    return sorted_rules


def parallel_read_rules_cg(dataset_name, train_set):
    """
    TODO: Not used anymore.
    """
    with ThreadPoolExecutor(5000) as executor:
        futures = {executor.submit(read_rules_cg, dataset_name, i): i for i in range(len(train_set))}

        sorted_futures = sorted(futures.items(), key=lambda x: x[1])

        sorted_results = [future.result() for future, i in sorted_futures]

    return sorted_results


def parallel_read_to_shared_dict(dataset_name, shared_dict, lock, batch_start, batch_end):
    """
    TODO: Not used anymore.
    """
    temp = list()
    for i in range(batch_start, batch_end):
        file_path = f'data/{dataset_name}/sorted_rules/{i}.pkl'
        try:
            with open(file_path, 'rb') as f:
                sorted_rules = pickle.load(f)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            sorted_rules = None
    
        temp.append(sorted_rules)

    with lock:
        for i in range(batch_start, batch_end):
            sorted_rules = temp[i - batch_start]
            if sorted_rules is not None:
                for rule in sorted_rules:
                    if rule not in shared_dict:
                        shared_dict[rule] = list()
                    shared_dict[rule].append(i)
    
    print(f"Batch {batch_start}-{batch_end} processed.")
    del temp
    gc.collect()  # Force garbage collection to free memory



def read_to_shared_dict(dataset_name, train_set, shared_dict, batch_size=100):
    """
    TODO: Not used anymore.
    """
    lock = multiprocessing.Lock()  # Lock for synchronization

    num_batches = ceil(len(train_set) / batch_size)

    print(f"Reading {len(train_set)} files in {num_batches} batches")

    batches = split_range_into_k_segments(start=0, end=len(train_set), k=num_batches)

    with ThreadPoolExecutor(max_workers=400) as executor:
         [executor.submit(parallel_read_to_shared_dict, dataset_name, shared_dict, lock, batch_start, batch_end) for batch_start, batch_end in batches]


def parallel_hierarchical_read(dataset_name, batch_start, batch_end):
    """
    """
    local_dict = defaultdict(list)
    for i in range(batch_start, batch_end):
        file_path = f'data/{dataset_name}/sorted_rules/{i}.pkl'
        try:
            with open(file_path, 'rb') as f:
                sorted_rules = pickle.load(f)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            sorted_rules = None
    
        if sorted_rules is not None:
            for rule in sorted_rules:
                local_dict[rule].append(i)

    with open(f'data/{dataset_name}/local_dict/{batch_start}_{batch_end}.pkl', 'wb') as f:
        pickle.dump(local_dict, f)
    
    print(f"Batch {batch_start}-{batch_end} processed and saved.")


def hierarchical_read_to_shared_dict(dataset_name, train_set, num_batches=50):
    """
    """
    batches = split_range_into_k_segments(start=0, end=len(train_set), k=num_batches)

    print(f"Hierarchical reading {len(train_set)} files in {num_batches} batches")
    with ProcessPoolExecutor(10) as executor:
        futures = [executor.submit(parallel_hierarchical_read, dataset_name, batch_start, batch_end) for batch_start, batch_end in batches]
        
        wait(futures)
    
    print("Hierarchical reading complete. Merging results")

    global_dict = defaultdict(list)    
    for batch_start, batch_end in batches:
        print(f"Post Processing batch {batch_start}-{batch_end}")
        with open(f'data/{dataset_name}/local_dict/{batch_start}_{batch_end}.pkl', 'rb') as f:
            local_dict = pickle.load(f)
        
        for rule, indices in local_dict.items():
            global_dict[rule].extend(indices)

    with open(f'data/{dataset_name}/global_dict.pkl', 'wb') as f:
        pickle.dump(global_dict, f)


def rule_to_predicate_cg(rule_w_target, one_hot=False, multi_valued=False):
    """
    
    """
    regex = rf"([^_]+)_(\d+)" # use this regex to match one-hot encoded features for multi-valued features
    rule_wo_target = rule_w_target[:-1]  # Exclude the target value
    target_clause = rule_w_target[-1]  # The target value

    if one_hot:
        if multi_valued:
           rule_wo_target_converted = ' or '.join([' and '.join([f'`{feat}`=={val}' if re.match(regex, feat) else f'`{feat}_{val}`==1' for feat, val in conjunction]) for conjunction in rule_wo_target])
           return  f'({rule_wo_target_converted}) and `{target_clause[0]}`=={target_clause[1]}'
        else:
            rule_wo_target_converted = ' or '.join([' and '.join([f'`{feat}_{val}`==1' for feat, val in conjunction]) for conjunction in rule_wo_target])
            return f'({rule_wo_target_converted}) and `{target_clause[0]}`=={target_clause[1]}'
    else:
        rule_wo_target_converted = ' or '.join([' and '.join([f'`{feat}`=={val}' for feat, val in conjunction]) for conjunction in rule_wo_target])
        return f'({rule_wo_target_converted}) and `{target_clause[0]}`=={target_clause[1]}'


def parallel_compute_support(train_set, target, labels, rules_list, start, end):
    """
    """
    arr = list()
    for i in range(start, end):
        index, rule_wo_target = rules_list[i]
        
        for label in labels:
            rule_w_target = rule_wo_target + ((target,label),)
            predicate = rule_to_predicate_cg(rule_w_target)
            temp_df = train_set.query(f'not({predicate})')
            temp_target_vals = temp_df[target].unique()
            is_single_target = len(temp_target_vals) == 1
            support = (len(train_set) - len(temp_df)) / len(train_set)
            arr.append((index, label, support, is_single_target))
    return arr


def filter_by_support(dataset_name, train_set, target, freq_threshold, support_threshold):
    """
    """
    with open(f'data/{dataset_name}/rules_freq_{freq_threshold}.pkl', 'rb') as f:
        freq_rules_dict = pickle.load(f)
    
    rules_list = list(freq_rules_dict.items())
    labels = train_set[target].unique()

    batches = split_range_into_k_segments(start=0, end=len(rules_list), k=56)
    
    with ProcessPoolExecutor(56) as executor:
        futures = {executor.submit(parallel_compute_support, train_set, target, labels, rules_list, batch_start, batch_end): i for i, (batch_start, batch_end) in enumerate(batches)}
        wait(futures)

        sorted_futures = sorted(futures.items(), key=lambda x: x[1])
        sorted_results = [future.result() for future, i in sorted_futures]
        result = list()
        for res in sorted_results:
            result.extend(res)

        filtered_rules_dict = {}
        for index, label, support, is_single_target in result:
            if support <= support_threshold and not is_single_target:
                filtered_rules_dict[(index, label)] = (freq_rules_dict[index] + ((target,label),), support)

        return filtered_rules_dict


def parallel_compute_ATE(train_set, target, rules_list, start, end):
    """
    """
    arr = list()
    for i in range(start, end):
        (index, label), (rule_w_target, support) = rules_list[i]
        predicate = rule_to_predicate_cg(rule_w_target)
        temp_df = train_set.query(f'not({predicate})')
        temp_target_vals = temp_df[target].unique()
        del_ATE = temp_df[target].mean() - train_set[target].mean()
        arr.append((index, label, del_ATE))
    return arr


def compute_ATE(dataset_name, train_set, target, freq_threshold, support_threshold):
    """
    """
    with open(f'data/{dataset_name}/rules_freq_{freq_threshold}_supp_{support_threshold}.pkl', 'rb') as f:
        filtered_rules_dict = pickle.load(f)
    
    rules_list = list(filtered_rules_dict.items())

    batches = split_range_into_k_segments(start=0, end=len(rules_list), k=56)
    
    with ProcessPoolExecutor(56) as executor:
        futures = {executor.submit(parallel_compute_ATE, train_set, target, rules_list, batch_start, batch_end): i for i, (batch_start, batch_end) in enumerate(batches)}
        wait(futures)

        sorted_futures = sorted(futures.items(), key=lambda x: x[1])
        sorted_results = [future.result() for future, i in sorted_futures]
        result = list()
        for res in sorted_results:
            result.extend(res)

        new_rules_dict = {}
        for index, label, del_ATE in result:
            new_rules_dict[(index, label)] = (filtered_rules_dict[(index, label)][0], filtered_rules_dict[(index, label)][1], del_ATE)

        return new_rules_dict


def equality_test(shm_name, shape, dtype, args):
    """
    """
    existing_shm = sm.SharedMemory(name=shm_name)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    arr = [(i, j, (shared_data[i] == shared_data[j]).all()) for i, j in args]
    existing_shm.close()
    return arr


def find_duplicate_testrows(num_processor, num_batches, dataset_name, df, target=None):
    """
    """
    if target is not None:
        df = df.drop(labels=[target], axis=1)

    # Convert DataFrame to a NumPy array (shared memory works with NumPy)
    data = df.to_numpy()

    # Create shared memory block
    shm = sm.SharedMemory(create=True, size=data.nbytes)

    # Create a NumPy array backed by shared memory
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data  # Copy data to shared memory

    print('Creating task list...')

    equality_matrix = np.zeros((len(df), len(df)), dtype=bool)
    tasks = [(i, j) for i in range(len(df)) for j in range(i+1, len(df))]

    print('Comparing rows...')
    
    batches = split_range_into_k_segments(start=0, end=len(tasks), k=num_batches)
    k=0
    for batch_start, batch_end in batches:
        print(f"iteration: {k+1}")
        k += 1

        segments = split_range_into_k_segments(start=batch_start, end=batch_end, k=num_processor)
        with ProcessPoolExecutor(num_processor) as executor:
            futures = [executor.submit(equality_test, 
                                    shm.name,
                                    data.shape, 
                                    data.dtype, 
                                    tasks[start:end]) 
                                    for start, end in segments]
        
            wait(futures)
            
            for future in as_completed(futures):
                arr = future.result()
                for i, j, equality in arr:
                    equality_matrix[i, j] = equality
    
    shm.close()
    shm.unlink()
    print('Comparison complete...')

    similar = np.arange(len(df))

    for i in range(len(df)):
        for j in range(i+1, len(df)):
            if similar[j] != j:
                continue

            if equality_matrix[i, j]:
                similar[j] = i
                print(i, j, df.index[i], df.index[j])
    
    with open(f'data/{dataset_name}/similar_testrows.npy' if target is None else f'data/{dataset_name}/similar_testrows_wo_target.npy', 'wb') as f:
        np.save(f, similar)


def remove_duplicates(num_processor, num_batches, dataset_name, df, target):
    """
    """
    with open(f'data/{dataset_name}/test_set_indices.pkl', 'rb') as f:
        test_set_indices = pickle.load(f)

    test_set = df.loc[test_set_indices]
    
    find_duplicate_testrows(num_processor=num_processor, num_batches=num_batches, dataset_name=dataset_name, df=test_set, target=target)

    with open(f'data/{dataset_name}/similar_testrows_wo_target.npy', 'rb') as f:
        similar_wo_target = np.load(f)

    # Group the indices of similar rows
    groups = defaultdict(set)
    for i in range(len(similar_wo_target)):
        groups[similar_wo_target[i]].add(i)

    # Keep the most common label in each group
    final_selection = list()
    for key in groups.keys():
        vals = list(groups[key])
        labels, indices, counts = np.unique(test_set.iloc[vals][target], return_index=True, return_counts=True)
        common_iloc_index = vals[indices[np.argmax(counts)]]
        final_selection.append(common_iloc_index)

    final_df = test_set.iloc[final_selection]

    print(f'test_set: {len(test_set)}, final_df: {len(final_df)}')

    with open(f'data/{dataset_name}/test_indices_no_duplicate.pkl', 'wb') as f:
        pickle.dump(final_df.index.to_list(), f)


def load_dataset(dataset_name):
    """
    """
    if dataset_name == 'adult':
        from src.proc_data.proc_adult import load_adult
        load_data = load_adult
    elif dataset_name == 'so':
        from src.proc_data.proc_so import load_so
        load_data = load_so
    elif dataset_name == 'compas':
        from src.proc_data.proc_compas import load_compas
        load_data = load_compas
    
    df, df_1hot, target, train_set_indices, test_indices_no_duplicate = load_data()
    return df, df_1hot, target, train_set_indices, test_indices_no_duplicate


def get_valid_eval_indices(dataset_name, fold):
    """
    """
    if dataset_name == 'adult':
        from src.proc_data.proc_adult import get_valid_eval_indices as get_indices
    elif dataset_name == 'so':
        from src.proc_data.proc_so import get_valid_eval_indices as get_indices
    elif dataset_name == 'compas':
        from src.proc_data.proc_compas import get_valid_eval_indices as get_indices
    
    valid_indices, eval_indices = get_indices(fold)
    return valid_indices, eval_indices


def get_base_model(dataset_name, model_name, device=None):
    """
    """
    if dataset_name == 'adult':
        from src.proc_data.proc_adult import load_adult, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_adult

    elif dataset_name == 'so':
        from src.proc_data.proc_so import load_so, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_so

    elif dataset_name == 'compas':
        from src.proc_data.proc_compas import load_compas, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_compas
    
    df, df_1hot, target, train_set_indices, test_indices_no_duplicate = load_data()
    train_set, test_set = df.loc[train_set_indices], df.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = df_1hot.loc[train_set_indices], df_1hot.loc[test_indices_no_duplicate]

    if model_name == 'nn':
        from src.proc_data.nn_util import getNNClassifier, fitNNClassifier, predictNNClassifier

        device = 'cpu' if device is None else device

        # Train the original model
        X_train, y_train = X_y_split(df=train_set_1hot, target=target)

        nn_hypers = getNNHyperparameters()
        hidden_dims, learning_rate, nn_batch_size, num_epochs = nn_hypers['hidden_dims'], nn_hypers['learning_rate'], nn_hypers['nn_batch_size'], nn_hypers['num_epochs']

        clf, criterion, optimizer = getNNClassifier(X_dim=X_train.shape[1], hidden_dims=hidden_dims, y_dim=1, device=device, learning_rate=learning_rate)
        fitNNClassifier(X=X_train, y=y_train[target], model=clf, criterion=criterion, optimizer=optimizer, device=device, batch_size=nn_batch_size, num_epochs=num_epochs)

        getClassifier = [nn_hypers, device, getNNClassifier, fitNNClassifier, predictNNClassifier]
        
        X_test, y_test = X_y_split(df=test_set_1hot, target=target)
        base_y_pred = predictNNClassifier(X=X_test, model=clf, device=device, batch_size=nn_batch_size)

    else:
        if model_name == 'rf':
            getClassifier = getRFClassifier
            _train_set, _test_set = train_set, test_set
        
        elif model_name == 'adaboost':
            getClassifier = getAdaBoostClassifier
            _train_set, _test_set = train_set, test_set
        
        elif model_name == 'logreg':
            getClassifier = getLogRegClassifier
            _train_set, _test_set = train_set_1hot, test_set_1hot
        
        elif model_name == 'linsvc':
            getClassifier = getSVMClassifier
            _train_set, _test_set = train_set_1hot, test_set_1hot
    
        # Train the original model
        X_train, y_train = X_y_split(df=_train_set, target=target)
        clf = getClassifier()
        clf.fit(X_train, y_train[target])

        X_test, y_test = X_y_split(df=_test_set, target=target)
        base_y_pred = clf.predict(X_test)

    return getClassifier, clf, base_y_pred


def get_base_predictions(dataset_name, model_name):
    """
    """
    with open(f'data/{dataset_name}/{model_name}_base_pred.pkl', 'rb') as f:
        base_y_pred = pickle.load(f)

    return base_y_pred


if __name__ == '__main__':
    # dataset_name = 'so'
    dataset_name = 'adult'
    with open(f'data/{dataset_name}/test_indices_no_duplicate.pkl', 'rb') as f:
        test_indices_no_duplicate = pickle.load(f)

    # df = pd.read_csv(f'data/{dataset_name}/so_enc_final.csv')
    df = pd.read_csv(f'data/{dataset_name}/adult_encoded.csv')

    df_no_duplicate = df.loc[test_indices_no_duplicate]
    
    # df_no_duplicate.to_csv(f'data/{dataset_name}/so_enc_final_no_duplicate.csv', index=False)
    df_no_duplicate.to_csv(f'data/{dataset_name}/adult_encoded_final_no_duplicate.csv', index=False)