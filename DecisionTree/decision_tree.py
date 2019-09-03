import numpy as np
import pandas as pd
from collections import Counter
from pprint import pprint


class DecisionTreeClassifier:
    """
    This class takes input cleaned data frame df,
    max_depth of the tree(optional)
    """
    def __init__(self, df):
        self.df = df
        self.column_header = df.columns
        self.features_type = self.column_feature_type(df)
        self.counter = 0
        self.max_depth = 5
        self.tree = {}  # Structure of the tree

    def fit(self):
        '''
        This member function fits the training data
        Initializes the tree attribute on the trained data
        '''
        self.tree = self.build_tree(self.df)

    # Method for predicting the test set
    def predict(self, train_df):
        """
        :param train_df:
        :return:
        """
        predictions = []
        for i in range(train_df.shape[0]):
            ans = self.predict_single_example(train_df.loc[i, :])
            predictions.append(ans)

        return predictions

    def predict_single_example(self, x_row):
        question = ''
        if len(self.tree.keys()) > 0:
            question = list(self.tree.keys())[0]
        else:
            print('Tree is not yet buit, run fit function')

        feature_name, comparison_operator, value = question.split(" ")

        # ask question, feature is continuous
        if comparison_operator == "<=":
            if x_row[feature_name] <= float(value):
                ans = tree[question][0]
            else:
                ans = tree[question][1]

        # else the feature is categorical
        else:
            if str(x_row[feature_name]) == value:
                ans = tree[question][0]
            else:
                ans = tree[question][1]

        # base case
        if not isinstance(ans, dict):
            return ans

        # recursive part
        else:
            residual_tree = ans
            return classify_example(x_row, residual_tree)

    '''
    Defining helper functions: Calculate entropy, overall_entropy, is_pure, max_label_class
    column_feature_type, get_potential_split, split_data, find_best_split(based on information gain)
    build_tree   
    '''
    # Defining helper functions
    @staticmethod
    def calculate_entropy(data):
        """
        This function calculates the entropy of the labels in data.
        :param data: 2d array with last column consists of labels
        :return: entropy
        """
        labels = data[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        entropy = sum(p * (-np.log2(p)))
        return entropy

    def overall_entropy(self, true_data_arr, false_data_arr):
        """
        This function calculates the entropy after splitting the data
        :param true_data_arr: True branch array after splitting the data
        :param false_data_arr: False branch array after splitting the data
        :return: overall entropy
        """
        total_size = len(true_data_arr) + len(false_data_arr)
        p_true = len(true_data_arr) / total_size
        p_false = len(false_data_arr) / total_size

        total_entropy = p_true * self.calculate_entropy(true_data_arr) \
                        + p_false * self.calculate_entropy(false_data_arr)
        return total_entropy

    @staticmethod
    def is_pure(df):
        """
        Boolean method for determining of the sample is pure
        :param df: DataFrame
        :return: true is data has only one label, else false
        """
        labels = df.iloc[:, -1]
        unique_labels = np.unique(labels)

        if len(unique_labels) == 1:
            return True
        else:
            return False

    @staticmethod
    def max_label_class(df):
        """
        :param df: DataFrame
        :return: label with the max the count
        """
        labels = df.iloc[:, -1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_idx = counts.argmax()
        max_count_label = unique_labels[max_idx]
        return max_count_label

    @staticmethod
    def column_feature_type(df):
        """
        :param df: DataFrame
        :return: list consisting of types of features, i.e. categorical or numerical
        """
        column_features = []
        for f in df.columns[:-1]:  # assuming labels is the last column
            unique_column_values = np.unique(df[f])
            first_value = unique_column_values[0]
            if (isinstance(first_value, str)) or len(unique_column_values) <= 15:
                column_features.append('categorical')
            else:
                column_features.append('numerical')

        return column_features

    def get_potential_splits(self, df):
        """
        :param df: DataFrame
        :return: Dictionary consisting of columns index and feature items as list
        e.g. {0: array([False,  True]), 1: array([False,  True])}
        here 0 represents the first column with items False and True
        """
        potential_splits = {}
        n_cols = df.shape[1]
        data = df.values
        for column_index in range(n_cols - 1):  # excluding the last column which is the label
            values = data[:, column_index]
            unique_values = np.unique(values)

            type_of_feature = self.features_type[column_index]
            if type_of_feature == "continuous":
                potential_splits[column_index] = []
                for index in range(len(unique_values)):
                    if index != 0:
                        current_value = unique_values[index]
                        previous_value = unique_values[index - 1]
                        potential_split = (current_value + previous_value) / 2

                        potential_splits[column_index].append(potential_split)

            elif len(unique_values) > 1:
                potential_splits[column_index] = unique_values
        return potential_splits

    def split_data(self, df, split_column_idx, split_value):
        """
        This method splits the data based on the column index and the value(
        if category of feature is continuous)

        :param df:  DataFrame
        :param split_column_idx: int, index of the column
        :param split_value: float
        :return: Two 2d arrays after splitting the data
        """
        data = df.values
        split_column_values = data[:, split_column_idx]

        type_of_feature = self.features_type[split_column_idx]
        if type_of_feature == "continuous":
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values > split_value]

        # feature is categorical
        else:
            data_below = data[split_column_values == split_value]
            data_above = data[split_column_values != split_value]

        return data_below, data_above

    def find_best_split(self, df, potential_split_indices):
        """
        This method finds the best split based on th maximum information gain.
        :param df: DataFrame
        :param potential_split_indices: list of integers, Column indices
        :return: index of the column and the value at which column needs to be split.
        """
        overall_entropy = 9999
        best_split_column = None
        best_split_value = None
        for column_index in potential_split_indices:
            for value in potential_split_indices[column_index]:
                data_below, data_above = self.split_data(df, column_index, value)
                current_overall_entropy = self.overall_entropy(data_above, data_below)
                if current_overall_entropy < overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value

    def build_tree(self, df):
        """
        This function builds the tree by splitting the dataframe recursively based on the maximum
        information gain. It uses helper functions defined earlier
        :param df:
        :return:
        """
        print('\n')
        print('Level ', self.counter)
        print('Count of 0(False) = ', df.loc[df.iloc[:, -1] == False, 'label'].shape[0])
        print('Count of 1(True) = ', df.loc[df.iloc[:, -1] == True, 'label'].shape[0])
        initial_entropy = self.calculate_entropy(df.values)
        print('Current Entropy  is = ', initial_entropy)
        if initial_entropy == 0.0:
            print('Reached leaf Node')
        # check if the data is a pure leaf or reached the maximum tree depth
        if (self.is_pure(df)) or (self.counter == self.max_depth):
            classification = self.max_label_class(df)

            return classification

        # recursive part of the function, has not classified remaining data as leaf.
        else:
            self.counter += 1

            # helper functions
            potential_splits = self.get_potential_splits(df)
            split_column, split_value = self.find_best_split(df, potential_splits)
            print('Splitting on feature' + ' ' + self.column_header[split_column] + ' ' + 'with gain ratio  ')
            data_below, data_above = self.split_data(df, split_column, split_value)
            print(initial_entropy - self.overall_entropy(data_below, data_above))
            print('\n')
            # determine question
            feature_name = self.column_header[split_column]
            type_of_feature = self.features_type[split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)

            # feature is categorical
            else:
                question = "{} = {}".format(feature_name, split_value)

            # instantiate the sub-tree
            sub_tree = {question: []}

            # find answers (recursion)
            df_below = pd.DataFrame(data_below, columns=['x1', 'x2', 'label'])
            true_ans = self.build_tree(df_below)

            df_above = pd.DataFrame(data_above, columns=['x1', 'x2', 'label'])
            false_ans = self.build_tree(df_above)

            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base case).
            if true_ans == false_ans:
                sub_tree = true_ans
            else:
                sub_tree[question].append(true_ans)
                sub_tree[question].append(false_ans)

            return sub_tree


if __name__ == '__main__':
    x1 = [True, False, True, False]
    x2 = [True, True, False, False]
    label = [True, True, True, False]

    df = pd.DataFrame(zip(x1, x2, label), columns=['x1', 'x2', 'label'])
    print(df)
    tree = DecisionTreeClassifier(df)
    tree.fit()
