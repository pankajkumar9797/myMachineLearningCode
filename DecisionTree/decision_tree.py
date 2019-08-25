import numpy as np
import pandas as pd
from collections import Counter


class Node:
    def __init__(self, n_feature):
        self.children = []
        self.value = n_feature


def read_file(df):
    raise NotImplementedError


def cal_entropy(x):
    return -x*np.log2(x)


def cal_parent_entropy(arr):
    label_items = Counter(arr).values()
    total = sum(Counter(arr).values())

    entropy = 0
    for i, element in enumerate(label_items):
        temp = element/total
        entropy += -temp*np.log2(temp)

    return entropy


def find_best_feature(df, features_list):
    parent = cal_parent_entropy(df['label'].values)

    columns = features_list
    feature_gain = {}
    for feature in columns:
        feature_values = list(set(df[feature].values))
        item_entropy = 0
        for items in feature_values:
            item_example = df.loc[(df[feature] == items) & (df['label']), 'label'].shape[0]
            total_examples = df.loc[(df[feature] == items), 'label'].shape[0]

            fraction_example = item_example/total_examples

            # Calculating the entropy of the if the feature is selected
            items_fraction = df.loc[(df[feature] == items), feature].shape[0]/df.shape[0]
            item_entropy += items_fraction*cal_entropy(fraction_example)

        feature_gain[feature] = parent - item_entropy

    print("Feature values: ", feature_gain.values())
    max_gain = max(feature_gain.values())
    best_feature = ''
    for feature in feature_gain.keys():
        if feature_gain[feature] == max_gain:
            best_feature = feature
            break

    return best_feature


def decision_tree(df, features_list, level):
    # calculate the parent entropy
    best_feature = find_best_feature(df, features_list)
    print("level : {} and best feature: {}".format(level, best_feature))
    level = level + 1
    features_list.remove(best_feature)
    # Now we have the best feature, we need to split the data further on this feature
    for items in set(df[best_feature].values):
        temp = Counter(df.loc[(df[best_feature] == items), 'label'])
        print("items: {} and counter {}".format(items, temp))
        if len(temp.keys()) > 1 and features_list:
            # We divide further
            print("Temp:", temp.values())
            decision_tree(df, features_list, level)
        else:
            print("Decision Tree ends")
            print("else: ", temp.values())


if __name__ == '__main__':
    x1 = [True, False, True, False]
    x2 = [True, True, False, False]
    label = [True, True, True, False]

    df = pd.DataFrame(zip(x1, x2, label), columns=['x1', 'x2', 'label'])
    features = list(df.columns[:-1])

    level = 0
    decision_tree(df, features, level)