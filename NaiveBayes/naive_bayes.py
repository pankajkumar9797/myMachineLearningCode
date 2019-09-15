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
import string
import re

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

    return X_train_df, X_test_df


def data_to_tfidf_data(X_train_df, X_test_df):

    vectorizer = TfidfVectorizer(encoding='latin1')
    X_train = vectorizer.fit_transform(X_train_df['data'].values)
    y_train = X_train_df['target'].values

    X_test = vectorizer.transform(X_test_df['data'].values)
    y_test = X_test_df['target'].values

    return X_train, y_train, X_test, y_test


def data_to_count_data(X_train_df):
    # count words, exclude stop words, remove words with frequency less than 50,
    # then use lemmatizer to count similar words as a single word.
    stop_words = stopwords.words('english')
    word_class = {}
    num_of_classes = list(set(X_train_df['target'].values))
    vocabulary = []
    y = X_train_df['target'].values
    data_doc_words = {}
    for c in num_of_classes:
        indices = np.where(y == c)[0]
        word_list = []
        word_class[c] = {}
        for i, idx in enumerate(indices):
            data = X_train_df.loc[idx, 'data']
            data = data.strip(string.punctuation).lower()
            data = re.findall('[a-zA-Z]+', data)
            words = [w for w in data if w not in stop_words and len(w) > 3]
            word_list.extend(words)
            data_doc_words[idx] = {}
            word_set, word_c = np.unique(words, return_counts=True)
            for w, count in zip(word_set, word_c):
                data_doc_words[idx][w] = count

        for word in word_list:
            word_class[c][word] = 0

        for word in word_list:
            word_class[c][word] += 1

    # Removing the less frequent words
    for c in num_of_classes:
        for word in list(word_class[c].keys()):
            if word_class[c][word] < 5:
                del word_class[c][word]
            else:
                vocabulary.append(word)

    vocabulary = list(set(vocabulary))
    total_word_count = {}
    for word in vocabulary:
        total_word_count[word] = 0

    for word in vocabulary:
        temp = 0
        for cl in num_of_classes:
            if word in word_class[cl].keys():
                temp += word_class[cl][word]

        total_word_count[word] = temp

    return word_class, total_word_count, data_doc_words


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


def counts_words_with_documents(X_train_df):
    data = X_train_df['data'].values

    raise NotImplementedError


def naive_bayes(class_word_count, class_word_count_total):
    num_of_classes = (class_word_count.keys())
    vocabulary = list(class_word_count_total.keys())
    V = len(vocabulary)
    probability = {}
    for word in vocabulary:
        probability[word] = {}
        for cl in num_of_classes:
            if word in class_word_count[cl].keys():
                temp = (class_word_count[cl][word] + 1)/(class_word_count_total[word] + V)
                temp = -np.log(temp)
                probability[word][cl] = temp
            else:
                probability[word][cl] = 0

    return probability


def probability(word_count_train, total_word_count_train, data_doc_count_train,
                prior_class_prob, test_data, category):
    # temp = np.log(word_count_train[category]) - np.log()

    raise NotImplementedError


def predict_single_data(word_count_train, total_word_count_train, data_doc_count_train, prior_class_prob, test_data):
    classes = word_count_train.keys()
    predicted_label = ''
    max_prob = -1000
    for c in classes:
        class_prob = probability(word_count_train, total_word_count_train, data_doc_count_train,
                                 prior_class_prob, test_data, c)
        predict_probability[c] = class_prob
        if class_prob > max_prob:
            max_prob = class_prob
            predicted_label = c

    return predicted_label


def calculate_prior_probabilities(df):
    total_documents = df.shape[0]
    classes, class_counts = np.unique(df['target'].values(), return_counts=True)
    prior_class_prob = {}
    for cl, counts in zip(classes, class_counts):
        prior_class_prob[cl] = counts/total_documents

    return prior_class_prob


def predict(train, test):
    word_count_train, total_word_count_train, data_doc_count_train = data_to_count_data(train)
    # Calculating the prior probabilities
    prior_class_prob = calculate_prior_probabilities(train)

    X_test = test['data'].values()
    y_predict_list = []
    for xdata in X_test:
        y_predict = predict_single_data(word_count_train, total_word_count_train,
                                        data_doc_count_train, prior_class_prob, xdata)
        y_predict_list.append(y_predict)

    return y_predict_list,


def score(y_predict, y):
    count = 0
    for predict_label, actual_label in zip(y_predict, y):
        if predict_label == actual_label:
            count += 1

    return count/len(y)


if __name__ == '__main__':
    train, test = train_test_split()
    word_count, total_word_count, data_doc_count = data_to_count_data(train)
    probability = naive_bayes(word_count, total_word_count)
    print(probability)
