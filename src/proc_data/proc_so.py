from collections import defaultdict
import glob
import multiprocessing
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from .util import get_base_model, remove_duplicates
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .util import build_rules_for_train_set_cg, sort_rules_cg, read_to_shared_dict, hierarchical_read_to_shared_dict, rule_to_predicate_cg, filter_by_support, compute_ATE


np.random.seed(42)


def preprocess_so():
    """
    """
    # Load the dataset
    cols = ['Age', 'Gender', 'RaceEthnicity', 'Country', 'FormalEducation', 'UndergradMajor', 'DevType', 'HoursComputer', 'Dependents', 'ConvertedSalary']
    target = 'ConvertedSalary'
    
    so = pd.read_csv(f'data/so/survey_results_public.csv', usecols=cols, keep_default_na=False)

    # Checkpoint: Convert missing values in the target column to NAN
    so[target] = so[target].replace({'NA': np.nan}).astype(float)
    so.to_csv(f'data/so/survey_results.csv', index=False)
    
    # Load Checkpoint
    so = pd.read_csv(f'data/so/survey_results.csv', keep_default_na=False)

    # Process multi valued columns
    multi_val_cols = ['Gender', 'RaceEthnicity', 'DevType']
    so_multi_val = so[multi_val_cols].copy()
    multi_val_col_metadata = dict()
    arr_multi_val_cols_to_set = [list() for i in range(len(so))]

    for col in multi_val_cols:
        all_values = set(';'.join(so[col]).split(';'))
        val_map = {val: index for index, val in enumerate(all_values)}
        multi_val_col_metadata[col] = val_map
        so_multi_val[list(map(lambda x: f'{col}_{val_map[x]}', all_values))] = 0

        col_vals = so[col].str.split(';')
        for i, vals in enumerate(col_vals):
            arr_multi_val_cols_to_set[i].extend(list(map(lambda x: f'{col}_{val_map[x]}', vals)))
    
    for i, cols_to_set in enumerate(arr_multi_val_cols_to_set):
        so_multi_val.loc[i, cols_to_set] = 1
    
    so_multi_val.drop(multi_val_cols, axis=1, inplace=True)
    
    with open(f'data/so/multi_val_col_metadata.pkl', 'wb') as f:
        pickle.dump(multi_val_col_metadata, f)

    # Encode single valued columns
    single_val_cols = list(set(cols) - (set(multi_val_cols) | set([target])))
    so_single_val = so[single_val_cols].copy()
    le_dict = {}
    for col in single_val_cols:
        le = LabelEncoder()
        so_single_val[col] = le.fit_transform(so_single_val[col])
        le_dict[col] = le
    
    with open(f'data/so/le_dict.pkl', 'wb') as f:
        pickle.dump(le_dict, f)

    # Apply one hot encoding of encoded single valued columns
    enc_1hot = OneHotEncoder(handle_unknown='ignore', dtype=np.int8, sparse_output=False)
    enc_1hot.set_output(transform='pandas')
    so_single_val_1hot = enc_1hot.fit_transform(so_single_val)
    
    with open(f'data/so/enc_1hot.pkl', 'wb') as f:
        pickle.dump(enc_1hot, f)

    # Checkpoint: Concat and save one hot encoded dataset
    so_1hot = pd.concat([so_single_val_1hot, so_multi_val, so[target]], axis=1)
    so_1hot.to_csv(f'data/so/so_1hot.csv', index=False)
    
    # Load Checkpoint
    so_1hot = pd.read_csv(f'data/so/so_1hot.csv')

    # ************** Careful: This will take a long time to recompute **************
    # # Checkpoint: Impute missing values for the target
    # imputer = KNNImputer(n_neighbors=10, weights='distance', n_jobs=-1)
    # imputer.set_output(transform='pandas')
    # so_1hot_imputed = imputer.fit_transform(so_1hot)
    # so_1hot_imputed.to_csv(f'data/so/so_1hot_imputed.csv', index=False)
    
    # Load Checkpoint
    so_1hot_imputed = pd.read_csv(f'data/so/so_1hot_imputed.csv')

    # Checkpoint: Discretize the target to binary variable   
    median_salary = so_1hot_imputed[target].median()
    temp = so_1hot_imputed[target] > median_salary
    so_1hot_imputed[f'{target}_0'] = 1 - temp.astype(int)
    so_1hot_imputed[f'{target}_1'] = temp.astype(int)
    so_1hot_imputed.drop(target, axis=1, inplace=True)
    so_1hot_imputed.to_csv(f'data/so/so_1hot_final.csv', index=False)
    
    # Load Checkpoint
    so_1hot_final = pd.read_csv(f'data/so/so_1hot_final.csv')

    # Checkpoint: Copy the target column to the label encoded dataset
    so_enc = pd.concat([so_single_val, so_multi_val, so_1hot_final[f'{target}_1']], axis=1)
    so_enc.rename(columns={f'{target}_1': target}, inplace=True)
    so_enc.to_csv(f'data/so/so_enc_final.csv', index=False)

    # Load Checkpoint
    so_enc = pd.read_csv(f'data/so/so_enc_final.csv')

    # Split the data into training and testing sets and save indices
    train_set, test_set = train_test_split(so_enc, test_size=0.20, random_state=42)
    with open(f'data/so/train_set_indices.pkl', 'wb') as f:
        pickle.dump(train_set.index.to_list(), f)
    with open(f'data/so/test_set_indices.pkl', 'wb') as f:
        pickle.dump(test_set.index.to_list(), f)


def load_so():
    """
    """
    # load preprocessed data
    # so_raw = pd.read_csv(f'data/so/survey_results.csv', keep_default_na=False)
    so = pd.read_csv(f'data/so/so_enc_final.csv')
    so_1hot = pd.read_csv(f'data/so/so_1hot_final.csv')
    target = 'ConvertedSalary'

    # Fix the target feature 
    so_1hot.drop(f'{target}_0', axis=1, inplace=True)
    so_1hot.rename(columns={f'{target}_1': target}, inplace=True)

    # Load indices
    with open(f'data/so/train_set_indices.pkl', 'rb') as f:
        train_set_indices = pickle.load(f)
    with open(f'data/so/test_indices_no_duplicate.pkl', 'rb') as f:
        test_indices_no_duplicate = pickle.load(f)
    
    # return so_raw, so, so_1hot, target, train_set_indices, test_indices_no_duplicate
    return so, so_1hot, target, train_set_indices, test_indices_no_duplicate


def build_rules_from_profiles_for_train_set():
    """
    For so dataset (3250 * 3747), takes approximately 12 mins to compute in notch475 
    """
    # Load data
    so, so_1hot, target, train_set_indices, test_indices_no_duplicate = load_so()

    so_raw = pd.read_csv(f'data/so/survey_results.csv', keep_default_na=False)
    with open(f'data/so/multi_val_col_metadata.pkl', 'rb') as f:
        multi_val_col_metadata = pickle.load(f)
    
    # Split train and test set
    train_set, test_set = so.loc[train_set_indices], so.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = so_1hot.loc[train_set_indices], so_1hot.loc[test_indices_no_duplicate]
    train_set_raw, test_set_raw = so_raw.loc[train_set_indices], so_raw.loc[test_indices_no_duplicate]

    with open(f'data/so/profiles_dis2.pkl', 'rb') as f:
        profiles = pickle.load(f)

    print(len(train_set))
    print(f'Number of profiles: {len(profiles)}')

    build_rules_for_train_set_cg(train_set=train_set, profiles=profiles, dataset_name='so', muli_value=True, train_set_raw=train_set_raw, multi_val_col_metadata=multi_val_col_metadata) #list(list(tuple(tuple(tuple(str,str)))))
    


def split_valid_eval_indices():
    """
    """
    _, _, _, _, test_indices_no_duplicate = load_so()

    folds = 3
    for fold in range(folds):
        rand_test_indices = np.random.permutation(len(test_indices_no_duplicate))
        limit = int(0.7 * len(rand_test_indices))
        valid_indices, eval_indices = rand_test_indices[:limit], rand_test_indices[limit:]
        with open(f'data/so/mlclf_indices_{fold+1}.pkl', 'wb') as f:
            pickle.dump((valid_indices, eval_indices), f)
            

def get_valid_eval_indices(fold):
    """
    """
    with open(f'data/so/mlclf_indices_{fold}.pkl', 'rb') as f:
        valid_indices, eval_indices = pickle.load(f)
    return valid_indices, eval_indices


def getRFClassifier():
    """
    """
    return RandomForestClassifier(n_estimators=100, random_state=42)


def getAdaBoostClassifier():
    """
    """
    base_estimator = DecisionTreeClassifier(max_depth=1)
    return AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME', n_estimators=100, random_state=42)


def getLogRegClassifier():
    """
    """
    return LogisticRegression(solver='liblinear', random_state=42)


def getSVMClassifier():
    """
    """
    return LinearSVC(random_state=42)


def getNNHyperparameters():
    """
    """
    return {'hidden_dims':  [512, 256, 128, 64, 32, 16],
            'learning_rate': 0.001,
            'nn_batch_size': 10000,
            'num_epochs': 30
            }


if __name__ == '__main__':
    """
    time python -m src.proc_data.proc_so
    """
    # preprocess_so()
    # remove_duplicates(num_processor=56, num_batches=5, dataset_name='so', df=pd.read_csv('data/so/so_enc_final.csv'), target='ConvertedSalary')

    so, so_1hot, target, train_set_indices, test_indices_no_duplicate = load_so()

    train_set, test_set = so.loc[train_set_indices], so.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = so_1hot.loc[train_set_indices], so_1hot.loc[test_indices_no_duplicate]

    # print(len(test_indices_no_duplicate))
    # valid_indices, eval_indices = get_valid_eval_indices(1)
    # print(len(valid_indices), len(eval_indices))

    #region
    # dataset_name = 'so'
    # n_trees = "10000"
    # max_depth = "5"
    # sample_size = "0.25"
    # max_features = "0.5"

    # print(f"Reading aggregated rules list for {dataset_name} dataset")

    # from src.util.directories import SCRATCH_DIR

    # DATA_DIR = f'{SCRATCH_DIR}/{dataset_name}'
    # aggr_rules_dir = f'{DATA_DIR}/rand_forest_aggr_rules/ntrees_{n_trees}_maxdepth_{max_depth}_samplesize_{sample_size}_maxfeatures_{max_features}'
    # with open(f'{aggr_rules_dir}/aggregated_rules_list_frequency_0_coverage_0_maxsupport_1.0.pkl', 'rb') as f:
    #     rules_list = pickle.load(f)
    
    # print(f"Number of rules: {len(rules_list)}")
    # print(f"{len(rules_list[0])}")
    # rule, metadata, global_index, label = rules_list[0][0], rules_list[0][1], rules_list[0][2], rules_list[0][3]
    # print(rule, metadata, global_index, label)
    #endregion

    # build_rules_from_profiles_for_train_set()

    # sort_rules_cg(dataset_name='so', train_set=train_set)

    # hierarchical_read_to_shared_dict(dataset_name='so', train_set=train_set)
    
    # with open(f'data/so/global_dict.pkl', 'rb') as f:
    #     global_dict = pickle.load(f)
    # print(f'len(global_dict): {len(global_dict)}')

    # temp = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
    # dict = defaultdict(int)
    # for key in global_dict.keys():
    #     for x in temp:
    #         if len(global_dict[key]) >= x:
    #             dict[x] += 1
    # print(dict)

    # dict = {}
    # min_freq = 1000
    # for i, key in enumerate(global_dict.keys()):
    #     if len(global_dict[key]) > min_freq:
    #         dict[i] = key
    
    # with open(f'data/so/rules_freq_{min_freq}.pkl', 'wb') as f:
    #     pickle.dump(dict, f)

    # filtered_rules_dict = filter_by_support(dataset_name='so', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=0.3)
    # with open(f'data/so/rules_freq_{min_freq}_supp_0.3.pkl', 'wb') as f:
    #     pickle.dump(filtered_rules_dict, f)

    # print(f'len(filtered_rules_dict): {len(filtered_rules_dict)}')

    # with open(f'data/so/rules/0.pkl', 'rb') as f:
    #     rules = pickle.load(f)
    
    # for rule in rules:
    #     labels = train_set[target].unique()
    #     for label in labels:
    #         rule_w_label = rule + ((target,label),)
    #         predicate = rule_to_predicate_cg(rule_w_label, one_hot=True, multi_valued=True)
    #         print(predicate)
    #         temp = train_set_1hot.query(predicate)
    #         temp2 = train_set_1hot.query(f'not({predicate})')
    #         print(f'len(temp): {len(temp)}, len(temp2): {len(temp2)} | len(train_set): {len(train_set_1hot)}')

    # model_name = 'nn'
    # getClassifier, clf, base_y_pred = get_base_model(dataset_name='so', model_name=model_name)
    # with open(f'data/so/{model_name}_base_pred.pkl', 'wb') as f:
    #     pickle.dump(base_y_pred, f)


    min_freq, max_support = 1000, 0.3
    filtered_rules_dict = compute_ATE(dataset_name='so', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=max_support)
    with open(f'data/so/rules_freq_{min_freq}_supp_{max_support}_w_ATE.pkl', 'wb') as f:
        pickle.dump(filtered_rules_dict, f)