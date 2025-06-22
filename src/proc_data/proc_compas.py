from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from .util import remove_duplicates
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .util import build_rules_for_train_set_cg, sort_rules_cg, hierarchical_read_to_shared_dict, filter_by_support, get_base_model, compute_ATE

np.random.seed(42)


def preprocess_compas():
    """
    """
    cols = ['priors_count', 'juv_other_count', 'juv_misd_count', 'juv_fel_count', 'race', 'age_cat', 'sex','score_text', 'is_recid']

    compas = pd.read_csv(f'data/compas/compas-scores-two-years.csv', usecols=cols)
    compas = compas[cols]

    compas['age_cat'] = compas['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2}).astype(int)    
    compas['score_text'] = compas['score_text'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)    
    compas['race'] = compas['race'].map({'Other': 0, 'African-American': 0, 'Hispanic': 0, 'Native American': 0, 'Asian': 0, 'Caucasian': 1}).astype(int)
    compas['sex'] = compas['sex'].map({'Male': 1, 'Female': 0}).astype(int)    
    
    compas.loc[(compas['priors_count'] <= 5), 'priors_count'] = 0
    compas.loc[(compas['priors_count'] > 5) & (compas['priors_count'] <= 15), 'priors_count'] = 1
    compas.loc[(compas['priors_count'] > 15), 'priors_count'] = 2
    
    compas.loc[(compas['juv_fel_count'] == 0), 'juv_fel_count'] = 0
    compas.loc[(compas['juv_fel_count'] == 1), 'juv_fel_count'] = 1
    compas.loc[(compas['juv_fel_count'] > 1), 'juv_fel_count'] = 2
    
    compas.loc[(compas['juv_misd_count'] == 0), 'juv_misd_count'] = 0
    compas.loc[(compas['juv_misd_count'] == 1), 'juv_misd_count'] = 1
    compas.loc[(compas['juv_misd_count'] > 1), 'juv_misd_count'] = 2
    
    compas.loc[(compas['juv_other_count'] == 0), 'juv_other_count'] = 0
    compas.loc[(compas['juv_other_count'] == 1), 'juv_other_count'] = 1
    compas.loc[(compas['juv_other_count'] > 1), 'juv_other_count'] = 2
    
    compas.to_csv(f'data/compas/compas.csv', index=False)

    # Apply one hot encoding
    enc_1hot = OneHotEncoder(handle_unknown='ignore', dtype=np.int8, sparse_output=False)
    enc_1hot.set_output(transform='pandas')
    compas_1hot = enc_1hot.fit_transform(compas)

    # Save the final DataFrame
    compas_1hot.to_csv(f'data/compas/compas_1hot.csv', index=False)

    # Split the data into training and testing sets and save indices
    train_set, test_set = train_test_split(compas, test_size=0.3, random_state=42)
    with open(f'data/compas/train_set_indices.pkl', 'wb') as f:
        pickle.dump(train_set.index.to_list(), f)
    with open(f'data/compas/test_set_indices.pkl', 'wb') as f:
        pickle.dump(test_set.index.to_list(), f)


def load_compas():
    """
    """
    # load preprocessed data
    compas = pd.read_csv(f'data/compas/compas.csv')
    compas_1hot = pd.read_csv(f'data/compas/compas_1hot.csv')
    target = 'is_recid'

    # Fix the target feature 
    compas_1hot.drop(f'{target}_0', axis=1, inplace=True)
    compas_1hot.rename(columns={f'{target}_1': target}, inplace=True)

    # Load indices
    with open(f'data/compas/train_set_indices.pkl', 'rb') as f:
        train_set_indices = pickle.load(f)
    with open(f'data/compas/test_set_indices.pkl', 'rb') as f:
        test_set_indices = pickle.load(f)

    return compas, compas_1hot, target, train_set_indices, test_set_indices


def split_valid_eval_indices():
    """
    """
    _, _, _, _, test_indices = load_compas()

    folds = 3
    for fold in range(folds):
        rand_test_indices = np.random.permutation(len(test_indices))
        limit = int(0.7 * len(rand_test_indices))
        valid_indices, eval_indices = rand_test_indices[:limit], rand_test_indices[limit:]
        with open(f'data/compas/mlclf_indices_{fold+1}.pkl', 'wb') as f:
            pickle.dump((valid_indices, eval_indices), f)


def get_valid_eval_indices(fold):
    """
    """
    with open(f'data/compas/mlclf_indices_{fold}.pkl', 'rb') as f:
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
    return {'hidden_dims': [128, 64, 32, 16],
            'learning_rate': 0.001,
            'nn_batch_size': 1000,
            'num_epochs': 30
            }


if __name__ == '__main__':
    """
    time python -m src.proc_data.proc_compas
    """
    # preprocess_compas()

    # remove_duplicates(num_processor=192, num_batches=5, dataset_name='compas', df=pd.read_csv(f'data/compas/compas.csv'), target='is_recid')

    compas, compas_1hot, target, train_set_indices, test_set_indices = load_compas()
    # # print(test_set_indices)
    # mlclf_train_indices, mlclf_eval_indices = get_valid_eval_indices(fold=1)

    train_set, test_set = compas.loc[train_set_indices], compas.loc[test_set_indices]
    train_set_1hot, test_set_1hot = compas_1hot.loc[train_set_indices], compas_1hot.loc[test_set_indices]

    # mlclf_train_set, mlclf_eval_set = test_set.iloc[mlclf_train_indices], test_set.iloc[mlclf_eval_indices]
    # print(mlclf_train_set)
    # print(mlclf_eval_set)

    # print(len(test_set_indices))

    # with open(f'data/compas/profiles_dis2.pkl', 'rb') as f:
    #     profiles = pickle.load(f)
    # print(f'Loaded {len(profiles)} profiles from disk.')

    # build_rules_for_train_set_cg(train_set=train_set, profiles=profiles, dataset_name='compas')

    # sort_rules_cg(dataset_name='compas', train_set=train_set)

    # hierarchical_read_to_shared_dict(dataset_name='compas', train_set=train_set)

    # with open(f'data/compas/global_dict.pkl', 'rb') as f:
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
    
    # with open(f'data/compas/rules_freq_{min_freq}.pkl', 'wb') as f:
    #     pickle.dump(dict, f)

    
    # filtered_rules_dict = filter_by_support(dataset_name='compas', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=0.3)
    # with open(f'data/compas/rules_freq_{min_freq}_supp_0.3.pkl', 'wb') as f:
    #     pickle.dump(filtered_rules_dict, f)

    # print(f'len(filtered_rules_dict): {len(filtered_rules_dict)}')

    # model_name = 'nn'
    # getClassifier, clf, base_y_pred = get_base_model(dataset_name='compas', model_name=model_name)
    # with open(f'data/compas/{model_name}_base_pred.pkl', 'wb') as f:
    #     pickle.dump(base_y_pred, f)

    min_freq, max_support = 1000, 0.3
    filtered_rules_dict = compute_ATE(dataset_name='compas', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=max_support)
    with open(f'data/compas/rules_freq_{min_freq}_supp_{max_support}_w_ATE.pkl', 'wb') as f:
        pickle.dump(filtered_rules_dict, f)