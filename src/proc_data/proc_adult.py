from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from .util import filter_by_support, remove_duplicates
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from .util import build_rules_for_train_set_cg, sort_rules_cg, parallel_read_rules_cg, read_to_shared_dict, hierarchical_read_to_shared_dict, rule_to_predicate_cg, get_base_model, compute_ATE
import multiprocessing

np.random.seed(42)


def preprocess_adult():
    """
    """
    # Load the dataset
    cols = ['age', 'race', 'gender', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'hours-per-week', 'native-country', 'income']
    adult = pd.read_csv('data/adult/adult.csv', usecols=cols)

    # Discretize age
    brackets = [float('-inf'), 20, 40, 60, float('inf')]
    labels = ['Young', 'YoungAdult', 'MidAged', 'Old']
    adult['age'] = pd.cut(adult['age'], bins=brackets, labels=labels)

    # Discretize hours-per-week
    brackets = [float('-inf'), 20, 40, 60, float('inf')]
    labels = ['Low', 'Medium', 'High', 'Extreme']
    adult['hours-per-week'] = pd.cut(adult['hours-per-week'], bins=brackets, labels=labels)

    # Apply label encoding. Without it Randomforest doesn't work. NaNs also get encoded.
    categ_cols = cols
    le_dict = {}
    for col in categ_cols:
        le = LabelEncoder()
        adult[col] = le.fit_transform(adult[col])
        le_dict[col] = le

    # Apply one hot encoding
    enc_1hot = OneHotEncoder(handle_unknown='ignore', dtype=np.int8, sparse_output=False)
    enc_1hot.set_output(transform='pandas')
    adult_1hot = enc_1hot.fit_transform(adult)

    # Save preprocessed data
    adult.to_csv('data/adult/adult_encoded.csv', index=False)
    adult_1hot.to_csv('data/adult/adult_1hot.csv', index=False)
    with open('data/adult/enc_1hot.pkl', 'wb') as f:
        pickle.dump(enc_1hot, f)
    with open('data/adult/le_dict.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    
    # Split the data into training and testing sets and save indices
    train_set, test_set = train_test_split(adult_1hot, test_size=0.3, random_state=42)
    with open('data/adult/train_set_indices.pkl', 'wb') as f:
        pickle.dump(train_set.index.to_list(), f)
    with open('data/adult/test_set_indices.pkl', 'wb') as f:
        pickle.dump(test_set.index.to_list(), f)


def load_adult():
    """
    """
    # load preprocessed data
    adult = pd.read_csv('data/adult/adult_encoded.csv')
    adult_1hot = pd.read_csv('data/adult/adult_1hot.csv')
    target = 'income'

    # Fix the target feature 
    adult_1hot.drop(f'{target}_0', axis=1, inplace=True)
    adult_1hot.rename(columns={f'{target}_1': target}, inplace=True)

    # Load indices
    with open('data/adult/train_set_indices.pkl', 'rb') as f:
        train_set_indices = pickle.load(f)
    with open(f'data/adult/test_indices_no_duplicate.pkl', 'rb') as f:
        test_indices_no_duplicate = pickle.load(f)
    
    return adult, adult_1hot, target, train_set_indices, test_indices_no_duplicate


def build_rules_from_profiles_for_train_set():
    """
    For adult dataset (3250 * 3747), takes approximately 12 mins to compute in notch475 
    """
    # Load data
    adult, adult_1hot, target, train_set_indices, test_indices_no_duplicate = load_adult()
    
    # Split train and test set
    train_set, test_set = adult.loc[train_set_indices], adult.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = adult_1hot.loc[train_set_indices], adult_1hot.loc[test_indices_no_duplicate]

    with open(f'data/adult/profiles_dis2.pkl', 'rb') as f:
        profiles = pickle.load(f)

    print(len(train_set))

    # build_rules_for_train_set_cg(train_set=train_set, profiles=profiles, dataset_name='adult') #list(list(tuple(tuple(tuple(str,str)))))

    # with open(f'data/adult/rules_trainset.pkl', 'wb') as f:
    #     pickle.dump(rules_trainset, f)

    # print(f'type(rules): {type(rules_trainset)}, len(rules): {len(rules_trainset)}')
    # print(f'type(rules[0]): {type(rules_trainset[0])}, len(rules[0]): {len(rules_trainset[0])}')
    # print(f'type(rules[0][0]): {type(rules_trainset[0][0])}, len(rules[0][0]): {len(rules_trainset[0][0])}')
    # print(f'type(rules[0][0][0]): {type(rules_trainset[0][0][0])}, len(rules[0][0][0]): {len(rules_trainset[0][0][0])}')
    # print(f'type(rules[0][0][0][0]): {type(rules_trainset[0][0][0][0])}, len(rules[0][0][0][0]): {len(rules_trainset[0][0][0][0])}')
    # print(f'type(rules[0][0][0][0][0]): {type(rules_trainset[0][0][0][0][0])}, len(rules[0][0][0][0][0]): {len(rules_trainset[0][0][0][0][0])}')
    
    # print(f'rules[0]: {rules_trainset[0]}')
    # print(f'type(rules[0]): {type(rules_trainset[0])}')
    # print(f'rules[0][0]: {rules_trainset[0][0]}')
    # print(f'type(rules[0][0]): {type(rules_trainset[0][0])}')




def split_valid_eval_indices():
    """
    """
    _, _, _, _, test_indices_no_duplicate = load_adult()

    folds = 3
    for fold in range(folds):
        rand_test_indices = np.random.permutation(len(test_indices_no_duplicate))
        limit = int(0.7 * len(rand_test_indices))
        valid_indices, eval_indices = rand_test_indices[:limit], rand_test_indices[limit:]
        with open(f'data/adult/mlclf_indices_{fold+1}.pkl', 'wb') as f:
            pickle.dump((valid_indices, eval_indices), f)
            

def get_valid_eval_indices(fold):
    """
    """
    with open(f'data/adult/mlclf_indices_{fold}.pkl', 'rb') as f:
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
    return {'hidden_dims': [512, 256, 128, 64, 32, 16],
            'learning_rate': 0.001,
            'nn_batch_size': 10000,
            'num_epochs': 30
            }


if __name__ == '__main__':
    """
    time python -m src.proc_data.proc_adult
    """
    # preprocess_adult()
    # remove_duplicates(num_processor=56, num_batches=5, dataset_name='adult', df=pd.read_csv('data/adult/adult_encoded.csv'), target='income')

    # split_valid_eval_indices()

    adult, adult_1hot, target, train_set_indices, test_indices_no_duplicate = load_adult()

    # valid_indices, eval_indices = get_valid_eval_indices(fold=1)

    train_set, test_set = adult.loc[train_set_indices], adult.loc[test_indices_no_duplicate]
    train_set_1hot, test_set_1hot = adult_1hot.loc[train_set_indices], adult_1hot.loc[test_indices_no_duplicate]

    # print(len(test_set))

    # for x in mlclf_train_indices:
    #     assert x < len(test_set), print(f'x: {x}, len(test_set): {len(test_set)}')
    # for x in mlclf_eval_indices:
    #     assert x < len(test_set), print(f'x: {x}, len(test_set): {len(test_set)}')

    # print(eval_indices)

    # print(len(valid_indices), len(eval_indices))

    # mlclf_train_set, mlclf_eval_set = test_set.iloc[mlclf_train_indices], test_set.iloc[mlclf_eval_indices]
    # print(mlclf_train_set)
    # print(mlclf_eval_set)

    # instantiate_profiles_for_test_set()
    # build_rules_from_profiles_for_train_set()

    # with open(f'data/adult/rules/0.pkl', 'rb') as f:
    #     rules = pickle.load(f)

    # print(f'type(rules): {type(rules)}, len(rules): {len(rules)}')
    # print(f'type(rules[0]): {type(rules[0])}, len(rules[0]): {len(rules[0])}')
    # print(f'type(rules[0][0]): {type(rules[0][0])}, len(rules[0][0]): {len(rules[0][0])}')
    # print(f'type(rules[0][0][0]): {type(rules[0][0][0])}, len(rules[0][0][0]): {len(rules[0][0][0])}')
    # print(f'type(rules[0][0][0][0]): {type(rules[0][0][0][0])}, len(rules[0][0][0][0]): {len(rules[0][0][0][0])}')

    # print(f'value: {rules[1000]}')
    # print(f'value[0]: {rules[1000][0]}')
    # print(f'value[0][0]: {rules[1000][0][0]}')
    # print(f'value[0][0][0]: {rules[1000][0][0][0]}')

    # sort_rules_cg(dataset_name='adult', train_set=train_set)
    # sorted_rules = parallel_read_rules_cg(dataset_name='adult', train_set=train_set)

    # with multiprocessing.Manager() as manager:
    #     rules_dict = manager.dict()

    #     read_to_shared_dict(dataset_name='adult', train_set=train_set, shared_dict=rules_dict)

    #     with open(f'data/adult/rules_dict.pkl', 'wb') as f:
    #         pickle.dump(rules_dict, f)


    # hierarchical_read_to_shared_dict(dataset_name='adult', train_set=train_set)

    # with open(f'data/adult/global_dict.pkl', 'rb') as f:
    #     global_dict = pickle.load(f)
    # print(f'len(global_dict): {len(global_dict)}')

    # with open(f'data/adult/rules_list.pkl', 'wb') as f:
    #     pickle.dump(list(global_dict.keys()), f)

    # with open(f'data/adult/rules_list.pkl', 'rb') as f:
    #     rules_list = pickle.load(f)
    
    # temp = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]
    # dict = defaultdict(int)
    # for key in global_dict.keys():
    #     for x in temp:
    #         if len(global_dict[key]) >= x:
    #             dict[x] += 1
    # print(dict)

    # dict = {}
    # min_freq = 2000
    # for i, key in enumerate(global_dict.keys()):
    #     if len(global_dict[key]) > min_freq:
    #         dict[i] = key
    
    # with open(f'data/adult/rules_freq_{min_freq}.pkl', 'wb') as f:
    #     pickle.dump(dict, f)

    # filtered_rules_dict = filter_by_support(dataset_name='adult', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=0.3)
    # with open(f'data/adult/rules_freq_{min_freq}_supp_0.3.pkl', 'wb') as f:
    #     pickle.dump(filtered_rules_dict, f)

    # print(f'len(filtered_rules_dict): {len(filtered_rules_dict)}')

    # import matplotlib.pyplot as plt

    # bins = np.linspace(0, 50, 50)

    # plt.figure(figsize=(10, 6))
    # plt.hist(temp, bins=bins, color='skyblue', edgecolor='black')
    # plt.title('Histogram of temp values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('data/adult/temp_histogram.png', dpi=300)


    # with open(f'data/adult/rules/0.pkl', 'rb') as f:
    #     rules = pickle.load(f)
    
    # for rule in rules:
    #     labels = train_set[target].unique()
    #     for label in labels:
    #         rule_w_label = rule + ((target,label),)
    #         predicate = rule_to_predicate_cg(rule_w_label, one_hot=True)
    #         print(predicate)
    #         temp = train_set_1hot.query(predicate)
    #         temp2 = train_set_1hot.query(f'not({predicate})')
    #         print(f'len(temp): {len(temp)}, len(temp2): {len(temp2)} | len(train_set): {len(train_set_1hot)}')

    # model_name = 'adaboost'
    # getClassifier, clf, base_y_pred = get_base_model(dataset_name='adult', model_name=model_name)
    # with open(f'data/adult/{model_name}_base_pred.pkl', 'wb') as f:
    #     pickle.dump(base_y_pred, f)

    # min_freq, max_support = 2000, 0.3
    # filtered_rules_dict = compute_ATE(dataset_name='adult', train_set=train_set, target=target, freq_threshold=min_freq, support_threshold=max_support)
    # with open(f'data/adult/rules_freq_{min_freq}_supp_{max_support}_w_ATE.pkl', 'wb') as f:
    #     pickle.dump(filtered_rules_dict, f)


    rules_dir = f'data/adult/rules_freq_2000_supp_0.3_w_ATE.pkl'
    with open(rules_dir, 'rb') as f:
        rules = pickle.load(f)
    
    data = [(rule[-1][1], support, del_ATE) for rule, support, del_ATE in rules.values()]


    # Separate supports and del_ATEs based on the first value (0 or 1) in each tuple
    supports_0, del_ATEs_0 = [], []
    supports_1, del_ATEs_1 = [], []

    for label, support, del_ATE in data:
        if label == 0:
            supports_0.append(support)
            del_ATEs_0.append(del_ATE)
        else:
            supports_1.append(support)
            del_ATEs_1.append(del_ATE)

    # # Example: split a list of tuples into two lists
    # # Suppose data is a list of (support, del_ATE) tuples
    # supports, del_ATEs = zip(*data)
    # supports = list(supports)
    # del_ATEs = list(del_ATEs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(supports_0, del_ATEs_0, c='blue', label='Label 0', alpha=0.6)
    plt.scatter(supports_1, del_ATEs_1, c='red', label='Label 1', alpha=0.6)
    plt.xlabel('Support')
    plt.ylabel('Delta ATE')
    plt.title('Support vs Delta ATE Scatterplot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/adult/support_vs_delta_ATE_scatter.png', dpi=300)
    # plt.show()