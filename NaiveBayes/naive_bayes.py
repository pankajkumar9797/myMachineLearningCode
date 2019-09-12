import numpy as np
import pandas as pd
from pprint import pprint
import os
import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

nltk.download('stopwords')
nltk.download('punkt')


def create_feature_counts_dataframe(df):
    n_cols = df.shape[1]
    col_names = df.columns
    labels = set(df.iloc[:, -1].values)
    Count = {}
    Prob = {}
    for i in range(n_cols-1):
        features = list(set(df.iloc[:, i].values))
        print(features)
        Count[col_names[i]] = {}
        Prob[col_names[i]] = {}
        for f in features:
            Count[col_names[i]][f] = {}
            Prob[col_names[i]][f] = {}
            for l in labels:
                count_f_l = df.loc[(df[col_names[i]] == f) & (df[col_names[-1]] == l), :].shape[0]
                count_l = df.loc[df[col_names[-1]] == l, :].shape[0]
                Count[col_names[i]][f][l] = count_f_l
                Prob[col_names[i]][f][l] = count_f_l/count_l

    return Count, Prob


def classify(row_df):
    raise NotImplementedError


def train_test_split():
    file_paths = []
    folders = os.listdir('20_newsgroups')
    for folder in folders:
        root_path = os.path.join('20_newsgroups', folder)
        files = os.listdir(root_path)
        for file in files:
            file_path = os.path.join('20_newsgroups', folder, file)
            file_paths.append(file_path)

    # Now we have all paths and we need to reshuffle for generating the X_train and the X_test
    # splitting the dataset into 0.8 and 0.2
    random.shuffle(file_paths)
    threshold_len = int(len(file_paths)*0.8)
    X_train_path = file_paths[:threshold_len]
    X_test_path = file_paths[threshold_len:]

    X_train_files = []
    X_train_class = []
    for p in X_train_path:
        temp = p.split('/')
        class_name = temp[1]
        X_train_class.append(class_name)
        with open(p, 'r', errors='ignore') as f:
            data = f.read()
            X_train_files.append(data)
        f.close()

    X_train_df = pd.DataFrame()
    X_train_df['data'] = X_train_files
    X_train_df['target'] = X_train_class

    X_test_files = []
    X_test_class = []
    for p in X_test_path:
        temp = p.split('/')
        class_name = temp[1]
        X_test_files.append(class_name)
        with open(p, 'r', errors='ignore') as f:
            data = f.read()
            X_test_class.append(data)
        f.close()

    X_test_df = pd.DataFrame()
    X_test_df['data'] = X_test_files
    X_test_df['target'] = X_test_class

    vectorizer = TfidfVectorizer(encoding='latin1')
    X_train = vectorizer.fit_transform(X_train_df['data'].values)
    y_train = X_train_df['target'].values

    X_test = vectorizer.transform(X_test_df['data'].values)
    y_test = X_test_df['target'].values

    return X_train, y_train, X_test, y_test


def count_classes(X, y):
    class_word_count = {}
    class_word_count_total = {}
    num_of_classes = list(set(y))
    for cl in num_of_classes:
        indices = np.where(y == cl)[0]
        temp = X[indices, :].sum(axis=0)
        temp = np.array(temp)
        class_word_count[cl] = temp.flatten()
        class_word_count_total[cl] = np.sum(class_word_count[cl])

    return class_word_count, class_word_count_total


def naive_bayes(X_train, X_test):
    raise NotImplementedError


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = train_test_split()
    print(type(X_train))
    print(X_train.shape)
    class_words, total_words = count_classes(X_train, y_train)
    print(total_words)
    for key in class_words.keys():
        temp = class_words[key]