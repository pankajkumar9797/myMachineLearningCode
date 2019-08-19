__author__ = "Pankaj Kumar"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def standardize_data(X):
    columns = X.columns
    for col in columns:
        X[col] = (X[col].values - np.min(X[col].values)) / (np.max(X[col].values) - np.min(X[col].values))

    return X


class LogisticRegression:
    def __init__(self, file_path, iterations=50000, learning_rate=0.001, batch_size=32):
        train_data = pd.read_csv(file_path)
        train_data.drop('Unnamed: 0', axis=1, inplace=True)
        columns = train_data.columns
        train_data.rename(columns={columns[-1]: 'Y'}, inplace=True)
        for i in range(train_data.shape[1] - 1):
            train_data[i + train_data.shape[1]] = train_data.iloc[:, i] * train_data.iloc[:, i]

        X_initial = pd.DataFrame(np.ones(train_data.shape[0]), columns=['constant'])
        X_initial = X_initial.join(standardize_data(train_data.drop('Y', axis=1)))

        self.X = X_initial.values
        self.y = train_data.loc[:, 'Y'].values
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.slope_theta = np.zeros(self.X.shape[1])
        self.cost_vector = []

    # Calculates the gradient of the batch of examples
    def dcost_dtheta(self, slope_theta, X, y):
        N = X.shape[0]
        # Apply sigmoid to the output of the linear regression
        predictions = np.divide(1.0, 1.0 + np.exp((-1)*X.dot(slope_theta)))  # X.dot(slope_theta)
        diff = y-predictions
        cost = (-1)*(y*np.log(predictions) + (1 - y)*np.log(1.0 - predictions))
        self.cost_vector.append((1/N)*np.sum(cost))
        gradient = (-1/N)*np.transpose(X).dot(diff)
        return gradient  # this will be a vector of gradients

    # Calculates the slope and intercept by iterating over examples
    def fit(self):
        N = self.X.shape[0]
        for it in range(self.iterations):
            self.slope_theta -= self.learning_rate * self.dcost_dtheta(self.slope_theta,
                                                                       self.X,
                                                                       self.y)

    def predict(self, test_file_path):
        test_data = pd.read_csv(test_file_path)
        for i in range(test_data.shape[1]-1):
            test_data[i + test_data.shape[1]] = test_data.iloc[:, i] * test_data.iloc[:, i]

        test_data.drop('Unnamed: 0', axis=1, inplace=True)
        test_data = standardize_data(test_data)
        prediction = self.slope_theta[0] + test_data.values.dot(self.slope_theta[1:])

        probability = np.divide(1.0, 1.0 + np.exp((-1) * prediction))
        final_prediction = (np.where(probability > 0.5, 1, 0))

        return final_prediction

    def intercept(self):
        return self.slope_theta[0]

    def coefficients(self):
        return self.slope_theta[1:]

    def plot_cost(self):
        fig = plt.figure()
        plt.plot(self.cost_vector)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig("Loss.png")


if __name__ == '__main__':
    train_file_path = 'titanic_train.csv'
    test_file = 'titanic_test.csv'

    lg = LogisticRegression(train_file_path)
    lg.fit()
    lg.plot_cost()
    prediction = lg.predict(test_file)

    with open('predictions.csv', 'w') as f:
        for pred in prediction:
            f.write("%3.5f\n" % (pred))

    f.close()