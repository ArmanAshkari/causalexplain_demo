import argparse
import pathlib
import pickle
import time
import numpy as np
import os
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor, wait
from src.util.util import X_y_split, split_range_into_k_segments
from src.taskmonitor.util import TRACKER_DIR, init_tracker_file, write_tracker_file, write_complete_work
from src.util.directories import SCRATCH_DIR
from src.proc_data.util import rule_to_predicate_cg

np.random.seed(42)


def parallel_retrain_taskmonitor(taskid, jobid, train_set, test_set, target, getClassifier, base_clf, indices, this_rules, model_name, one_hot, multi_valued, save_dir):
    """
    """
    X_test, y_test = X_y_split(df=test_set, target=target)
    
    if model_name == 'nn':
        nn_hypers, device, getNNClassifier, fitNNClassifier, predictNNClassifier = getClassifier
        hidden_dims, learning_rate, nn_batch_size, num_epochs = nn_hypers['hidden_dims'], nn_hypers['learning_rate'], nn_hypers['nn_batch_size'], nn_hypers['num_epochs']
        base_y_pred = predictNNClassifier(X=X_test, model=base_clf, device=device, batch_size=nn_batch_size)
    else:
        base_y_pred = base_clf.predict(X_test)
    
    for i, this_index in enumerate(indices):
        init_tracker_file(taskid=taskid, jobid=jobid, index=this_index)

        (global_index, label), (rule_w_target, support) = this_rules[i]

        if os.path.exists(f'{save_dir}/metadata/{global_index}_{label}.pkl'):
            print(f"File {save_dir}/metadata/{global_index}_{label}.pkl exists. Skipping.")
            write_tracker_file(taskid=taskid, jobid=jobid, index=this_index, msg=f"File {save_dir}/metadata/{global_index}_{label}.pkl exists. Skipping.")
            write_complete_work(taskid=taskid, jobid=jobid, index=this_index)
            continue

        predicate = rule_to_predicate_cg(rule_w_target, one_hot, multi_valued)
        print(this_index, predicate)
        
        new_train_set = train_set.query(f'not({predicate})')
        dropped_indices = set(train_set.index) - set(new_train_set.index)
        drop_set_size = len(dropped_indices)
        print(this_index, f"Drop Set Size: {drop_set_size} | support: {support}=?{len(dropped_indices)/len(train_set)}")

        X_train, y_train = X_y_split(new_train_set, target)
        
        if model_name == 'nn':
            new_clf, criterion, optimizer = getNNClassifier(X_dim=X_train.shape[1], hidden_dims=hidden_dims, y_dim=1, device=device, learning_rate=learning_rate) 
            fitNNClassifier(X=X_train, y=y_train[target], model=new_clf, criterion=criterion, optimizer=optimizer, device=device, batch_size=nn_batch_size, num_epochs=num_epochs)
            new_y_pred = predictNNClassifier(X=X_test, model=new_clf, device=device, batch_size=nn_batch_size)
        else:
            new_clf = getClassifier()
            new_clf.fit(X_train, y_train[target])
            new_y_pred = new_clf.predict(X_test)
    
        accuracy = accuracy_score(y_true=y_test, y_pred=new_y_pred)
        model_similarity = accuracy_score(y_true=base_y_pred, y_pred=new_y_pred)
        
        if model_name != 'rf':
            with open(f'{save_dir}/models/{global_index}_{label}.pkl', 'wb') as f:
                pickle.dump(new_clf, f)

        metadata = (accuracy, model_similarity, new_y_pred)
        with open(f'{save_dir}/metadata/{global_index}_{label}.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(this_index, f"Accuracy: {accuracy}, Model Similarity: {model_similarity}, Drop Set Size: {drop_set_size}")
        write_tracker_file(taskid=taskid, jobid=jobid, index=this_index, msg=f"Accuracy: {accuracy}, Model Similarity: {model_similarity}, Drop Set Size: {drop_set_size}")

        write_complete_work(taskid=taskid, jobid=jobid, index=this_index)


def retrain_taskmonitor(taskid, jobid, train_set, test_set, target, getClassifier, base_clf, num_processor, batch_start, batch_size, indices, dataset_name, model_name, one_hot, multi_valued, threshold_str):
    """
    """
    print('retrain', taskid, jobid, batch_start, batch_size, dataset_name, model_name, threshold_str)
    
    rules_dir = f'{SCRATCH_DIR}/data/{dataset_name}/rules_{threshold_str}.pkl'
    with open(rules_dir, 'rb') as f:
        rules_dict = pickle.load(f)
    rules = list(rules_dict.items())

    this_indices = indices[batch_start: (batch_start+batch_size)]
    segements = split_range_into_k_segments(start=0, end=len(this_indices), k=num_processor)

    save_dir = f'{SCRATCH_DIR}/data/{dataset_name}/saved_models/{model_name}'
    pathlib.Path(f'{save_dir}/models').mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{save_dir}/metadata').mkdir(parents=True, exist_ok=True)

    print(f"Launching parallel retrain task monitor with {num_processor} processors.")

    with ProcessPoolExecutor(num_processor) as executor:
        futures = [executor.submit(parallel_retrain_taskmonitor,
                                taskid,
                                jobid,
                                train_set,
                                test_set,
                                target,
                                getClassifier,
                                base_clf,
                                this_indices[start:end],
                                [rules[i] for i in this_indices[start:end]],
                                model_name,
                                one_hot,
                                multi_valued,
                                save_dir)
                                for start, end in segements]

        # wait for all tasks to complete
        wait(futures)
        print("All threads completed. Clean exit.")


def retrain(args):
    """
    """
    taskid = args.taskid
    jobid = args.jobid
    num_processor = 2 * args.num_processor
    batch_start = args.batch_start
    batch_size = args.batch_size
    dataset_name = args.dataset
    model_name = args.model
    threshold_str = args.threshold_str

    with open(f'{TRACKER_DIR}/indices/{taskid}-{jobid}.pkl', 'rb') as f:
        indices= pickle.load(f)
    
    if args.dataset == 'adult':
        from src.proc_data.proc_adult import load_adult, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_adult
        multi_valued = False
    elif args.dataset == 'so':
        from src.proc_data.proc_so import load_so, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_so
        multi_valued = True
    elif args.dataset == 'compas':
        from src.proc_data.proc_compas import load_compas, getRFClassifier, getAdaBoostClassifier, getLogRegClassifier, getSVMClassifier, getNNHyperparameters
        load_data = load_compas
        multi_valued = False
    
    df, df_1hot, target, train_set_indices, test_indices_no_duplicate = load_data()
    train_set, test_set = df.loc[train_set_indices], df.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = df_1hot.loc[train_set_indices], df_1hot.loc[test_indices_no_duplicate]

    if model_name == 'nn':
        from src.proc_data.nn_util import getNNClassifier, fitNNClassifier, predictNNClassifier, get_device

        # device = get_device()
        device = 'cpu'
        print(f"Using device: {device}")

        # Train the original model
        X_train, y_train = X_y_split(df=train_set_1hot, target=target)

        nn_hypers = getNNHyperparameters()
        hidden_dims, learning_rate, nn_batch_size, num_epochs = nn_hypers['hidden_dims'], nn_hypers['learning_rate'], nn_hypers['nn_batch_size'], nn_hypers['num_epochs']

        clf, criterion, optimizer = getNNClassifier(X_dim=X_train.shape[1], hidden_dims=hidden_dims, y_dim=1, device=device, learning_rate=learning_rate)
        fitNNClassifier(X=X_train, y=y_train[target], model=clf, criterion=criterion, optimizer=optimizer, device=device, batch_size=nn_batch_size, num_epochs=num_epochs)

        getClassifier = [nn_hypers, device, getNNClassifier, fitNNClassifier, predictNNClassifier]

        retrain_taskmonitor(taskid=taskid, jobid=jobid, train_set=train_set_1hot, test_set=test_set_1hot, target=target, getClassifier=getClassifier, base_clf=clf, num_processor=num_processor, batch_start=batch_start, batch_size=batch_size, indices=indices, dataset_name=dataset_name, model_name=model_name, one_hot=True, multi_valued=multi_valued, threshold_str=threshold_str)
    
    else:
        if model_name == 'rf':
            getClassifier = getRFClassifier
            one_hot = False
            _train_set, _test_set = train_set, test_set
        elif model_name == 'adaboost':
            getClassifier = getAdaBoostClassifier
            one_hot = False
            _train_set, _test_set = train_set, test_set
        elif model_name == 'logreg':
            getClassifier = getLogRegClassifier
            one_hot = True
            _train_set, _test_set = train_set_1hot, test_set_1hot
        elif model_name == 'linsvc':
            getClassifier = getSVMClassifier
            one_hot = True
            _train_set, _test_set = train_set_1hot, test_set_1hot
    
        # Train the original model
        X_train, y_train = X_y_split(df=_train_set, target=target)
        clf = getClassifier()
        clf.fit(X_train, y_train[target])

        retrain_taskmonitor(taskid=taskid, jobid=jobid, train_set=_train_set, test_set=_test_set, target=target, getClassifier=getClassifier, base_clf=clf, num_processor=num_processor, batch_start=batch_start, batch_size=batch_size, indices=indices, dataset_name=dataset_name, model_name=model_name, one_hot=one_hot, multi_valued=multi_valued, threshold_str=threshold_str)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the experiment configuration.")
    
    parser.add_argument("taskid", type=int, help="Task ID")
    parser.add_argument("jobid", type=int, help="Job ID")
    parser.add_argument("num_processor", type=int, help="Number of processors")
    parser.add_argument("batch_start", type=int, help="Batch start")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument('--model', type=str, help="Name of the model")
    parser.add_argument('--threshold_str', type=str, help="Threshold string")
    
    return parser.parse_args()


if __name__ == '__main__':
    time.sleep(5)

    args = parse_args()
    
    print('main_script', args)
    
    retrain(args)

