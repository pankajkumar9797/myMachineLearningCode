import numpy as np
import pandas as pd
from collections import Counter
from pprint import pprint


class DecisionTreeClassifier:
    def __init__(self, df):
        self.df = df
        self.column_header = df.columns
        self.features_type = self.column_feature_type(df)
        self.counter = 0
        self.max_depth = 5

    def fit(self):
        tree = self.build_tree(self.df)
        pprint(tree)
        return tree

    def predict(self):
        raise NotImplementedError

    # Defining helper functions
    @staticmethod
    def calculate_entropy(data):
        labels = data[:, -1]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        entropy = sum(p * (-np.log2(p)))
        return entropy

    def overall_entropy(self, true_data_arr, false_data_arr):
        total_size = len(true_data_arr) + len(false_data_arr)
        p_true = len(true_data_arr) / total_size
        p_false = len(false_data_arr) / total_size

        total_entropy = p_true * self.calculate_entropy(true_data_arr) \
                        + p_false * self.calculate_entropy(false_data_arr)
        return total_entropy

    @staticmethod
    def is_pure(df):
        labels = df.iloc[:, -1]
        unique_labels = np.unique(labels)

        if len(unique_labels) == 1:
            return True
        else:
            return False

    @staticmethod
    def max_label_class(df):
        labels = df.iloc[:, -1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_idx = counts.argmax()
        max_count_label = unique_labels[max_idx]
        return max_count_label

    @staticmethod
    def column_feature_type(df):
        column_features = []
        for f in df.columns[:-1]:  # assuming labels is the last column
            unique_column_values = np.unique(df[f])
            first_value = unique_column_values[0]
            if (isinstance(first_value, str)) or len(unique_column_values) <= 15:
                column_features.append('categorical')
            else:
                column_features.append('numerical')

        return column_features

    def split_data(self, df, split_column_idx, split_value):
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

    def get_potential_splits(self, df):
        potential_splits = {}
        n_columns = df.shape[1]
        data = df.values
        for column_index in range(n_columns - 1):  # excluding the last column which is the label
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

    def build_tree(self, df):
        print('\n')
        print('Level ', self.counter)
        print('Count of 0(False) = ', df.loc[df.iloc[:, -1] == False, 'label'].shape[0])
        print('Count of 1(True) = ', df.loc[df.iloc[:, -1] == True, 'label'].shape[0])
        initial_entropy = self.calculate_entropy(df.values)
        print('Current Entropy  is = ', initial_entropy)
        # base cases
        if (self.is_pure(df)) or (self.counter == self.max_depth):
            classification = self.max_label_class(df)

            return classification

        # recursive part
        else:
            self.counter += 1

            # helper functions
            potential_splits = self.get_potential_splits(df)
            print('Potential splits: ', potential_splits)
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

            # instantiate sub-tree
            sub_tree = {question: []}

            # find answers (recursion)
            df_above = pd.DataFrame(data_above, columns=['x1', 'x2', 'label'])
            no_answer = self.build_tree(df_above)
            df_below = pd.DataFrame(data_below, columns=['x1', 'x2', 'label'])
            yes_answer = self.build_tree(df_below)

            # If the answers are the same, then there is no point in asking the qestion.
            # This could happen when the data is classified even though it is not pure
            # yet (min_samples or max_depth base case).
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree


if __name__ == '__main__':
    x1 = [True, False, True, False]
    x2 = [True, True, False, False]
    label = [True, True, True, False]

    df = pd.DataFrame(zip(x1, x2, label), columns=['x1', 'x2', 'label'])
    print(df)
    tree = DecisionTreeClassifier(df)
    tree.fit()
