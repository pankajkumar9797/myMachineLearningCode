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
import time

nltk.download('stopwords')
nltk.download('punkt')


def create_paths_array():
    """
    :return: Vector containing the paths to the documents inside class directory
    """
    file_paths = []
    folders = os.listdir('20_newsgroups')
    for folder in folders:
        root_path = os.path.join('20_newsgroups', folder)
        files = os.listdir(root_path)
        for file in files:
            file_path = os.path.join('20_newsgroups', folder, file)
            file_paths.append(file_path)

    return file_paths


def train_test_split(file_paths, fraction=0.8):
    """
    :param file_paths: Vector containing the paths to the documents inside class directory
    :param fraction: for splitting into the train and the test set, default is (0.80, 0.20) split for train and test
    :return: Dataframe of train and test sets with columns 'data' (text data) and 'target' (document label)
    """
    random.shuffle(file_paths) # Shuffling the array containing the paths to data
    threshold_len = int(len(file_paths)*fraction)
    X_train_path = file_paths[:threshold_len] # Train data set
    X_test_path = file_paths[threshold_len:]  # Test data set

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
        X_test_class.append(class_name)
        with open(p, 'r', errors='ignore') as f:
            data = f.read()
            X_test_files.append(data)
        f.close()

    X_test_df = pd.DataFrame()
    X_test_df['data'] = X_test_files
    X_test_df['target'] = X_test_class

    return X_train_df, X_test_df


class NaiveBayes:
    def __init__(self, X_train_df, X_test_df, alpha=0.1):
        self.X_train = X_train_df['data'].values
        self.X_test = X_test_df['data'].values

        self.y_train = X_train_df['target'].values
        self.y_test = X_test_df['target'].values
        self.class_labels = list(set(X_train_df['target'].values))
        self.alpha = alpha

        self.stop_words = stopwords.words('english')
        self.vocabulary = []
        self.word_class_freq = {}  # Word frequency w.r.t. class
        self.word_total_freq = {}  # Total word frequency in all classes
        self.prior_class_prob = self.calculate_prior_probabilities(X_train_df)

    def fit(self):
        """
        count words, exclude stop words, remove words with frequency less than 50,
        then use lemmatizer to count similar words as a single word.
        :return: frequency of words in respective classes and word frequency in all classes
        """
        word_class = {}
        vocabulary = []
        for c in self.class_labels:
            indices = np.where(self.y_train == c)[0]
            word_list = []
            word_class[c] = {}
            for i, idx in enumerate(indices):
                data = self.X_train[idx]
                data = data.strip(string.punctuation).lower()
                data = re.findall('[a-zA-Z]+', data)
                words = [w for w in data if w not in self.stop_words and len(w) > 3]
                word_list.extend(words)

            for word in word_list:
                word_class[c][word] = 0

            for word in word_list:
                word_class[c][word] += 1

        # Removing the less frequent words
        for c in self.class_labels:
            for word in list(word_class[c].keys()):
                if word_class[c][word] < 5:
                    del word_class[c][word]
                else:
                    vocabulary.append(word)
        """
        Setting up the vocabulary of list of words in documents of all classes
        """
        vocabulary = list(set(vocabulary))
        total_word_count = {}
        for word in vocabulary:
            total_word_count[word] = 0

        for word in vocabulary:
            temp = 0
            for cl in self.class_labels:
                if word in word_class[cl].keys():
                    temp += word_class[cl][word]

            total_word_count[word] = temp

        """
        Setting up the dictionary of words w.r.t. classes, in all classes and 
        also the class attribute of vocabulary 
        """
        self.word_class_freq = word_class
        self.word_total_freq = total_word_count
        self.vocabulary = vocabulary

    def data_to_tfidf_data(self):
        """
        :return: tfidf data for train and test data
        """
        vectorizer = TfidfVectorizer(encoding='latin1')
        X_train_tfidf = vectorizer.fit_transform(self.X_train)
        X_test_tfidf = vectorizer.transform(self.X_test)

        return X_train_tfidf, X_test_tfidf

    def data_to_words(self, X_data):
        """
        :param X_data: text data
        :return: word dictionary corresponding to the input text data
        """
        data = X_data
        data = data.strip(string.punctuation).lower()
        data = re.findall('[a-zA-Z]+', data)  # regular expression for extracting only words
        words = [w for w in data if w not in self.stop_words and len(w) > 3]
        word_set, word_c = np.unique(words, return_counts=True)
        word_count = {}
        for w, count in zip(word_set, word_c):
            word_count[w] = count

        return word_count

    @staticmethod
    def calculate_prior_probabilities(df):
        total_documents = df.shape[0]
        classes, class_counts = np.unique(df['target'].values, return_counts=True)
        prior_class_prob = {}
        for cl, counts in zip(classes, class_counts):
            prior_class_prob[cl] = counts/total_documents

        return prior_class_prob

    def probability(self, total_word_classes, point_test_data, category):
        """
        :param total_word_classes: A dictionary, it contains the total number of words in each class
        :param point_test_data: test text data
        :param category: class label
        :return: probability for a particular class label
        """
        # find words in the test data
        test_doc_word_count = self.data_to_words(point_test_data)
        prob = {}
        V = len(self.word_total_freq.keys())
        for word in test_doc_word_count.keys():
            if word in self.word_class_freq[category].keys():
                prob[word] = np.log((self.word_class_freq[category][word] + self.alpha)
                                    / (total_word_classes[category] + self.alpha*V))
            else:
                prob[word] = np.log(self.alpha/(total_word_classes[category] + self.alpha*V))

        return np.log(self.prior_class_prob[category]) + sum(prob.values())

    def predict_single_data(self, total_word_classes, test_data):
        """
        :param total_word_classes: A dictionary, it contains the total number of words in each class
        :param test_data: test text data
        :return: predicted class label which has the maximum probability
        """
        predicted_label = ''
        max_prob = -100000
        for c in self.class_labels:
            class_prob = self.probability(total_word_classes, test_data, c)
            if class_prob > max_prob:
                max_prob = class_prob
                predicted_label = c

        return predicted_label

    def predict(self):
        """
        :return: Predicted vector with class labels
        """
        total_word_classes = {}
        for c in self.class_labels:
            total_word_classes[c] = sum(self.word_class_freq[c].values())
        y_predict_list = []
        for xdata in self.X_test:
            y_predict = self.predict_single_data(total_word_classes, xdata)
            y_predict_list.append(y_predict)

        return y_predict_list,

    @staticmethod
    def score(y_predict, y):
        count = 0
        for predict_label, actual_label in zip(y_predict, y):
            if predict_label == actual_label:
                count += 1

        return count/len(y)

    @staticmethod
    def confusion_matrix(y_predict, y):
        df_confusion = pd.crosstab(y, y_predict)
        return df_confusion


def run_my_naive_bayes():
    file_paths = create_paths_array()
    train, test = train_test_split(file_paths)
    clf = NaiveBayes(train, test, alpha=1.0)
    clf.fit()
    y_pred = clf.predict()
    accuracy = clf.score(y_pred[0], test['target'].values)

    return accuracy


def sklearn_classify():
    """
    Using the Naive Bayes from Sklearn as a basis for comparing the accuracy of my Naive bayes algorithm
    :return: accuracy of classification
    """

    """
    importing various libraries
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics

    """
    fetching document data set from internet using inbuilt Sklearn function
    """
    news_train = fetch_20newsgroups(subset='train', shuffle=True)
    news_test = fetch_20newsgroups(subset='test', shuffle=True)

    """
    Converting the data set into the Term frequency and inverse document frequency.
    It gives more weight to the words which occur less frequent in class documents
    """
    tfid_vector = TfidfVectorizer()
    X_train_Tfidf = tfid_vector.fit_transform(news_train.data)
    X_test_Tfidf = tfid_vector.transform(news_test.data)

    """
    Using inbuilt sklearn naive bayes function
    """
    clf = MultinomialNB()
    clf.fit(X_train_Tfidf, news_train.target)
    predicted = clf.predict(X_test_Tfidf)
    """
    Calculating the accuracy score
    """
    score = metrics.accuracy_score(predicted, news_test.target)

    return score


if __name__ == '__main__':
    start_time = time.time()
    my_score = run_my_naive_bayes()
    my_naive_bayes_time = time.time()
    print("My program took :", my_naive_bayes_time - start_time)
    sklearn_score = sklearn_classify()
    sklearn_time = time.time()
    print("My Naive Bayes score : {}".format(my_score))
    print("Sklearn Naive Bayes score: {}".format(sklearn_score))
    print("Sklearn took :", sklearn_time - my_naive_bayes_time)