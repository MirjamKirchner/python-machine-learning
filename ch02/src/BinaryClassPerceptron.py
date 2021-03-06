import numpy as np


class BinaryClassPerceptron:
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        """
        An artifical neuron (perceptron) can execute a binary classification task on the input data
        :param eta: the learning rate, typically between 0.0 and 1.0
        :param n_iter: maximum number of passes over the training set
        :param random_state: a seed to initialise a random number generator
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array):
        """
        Fits the weights of the perceptron to minimise the prediction error
        :param X: nxm matrix where rows represent examples, and columns features
        :param y: a nx1 target-matrix (vector)
        :return: None
        """
        # Initialise weights
        rng = np.random.default_rng(self.random_state)  # Initialise random number generator rng
        self.w_ = rng.normal(scale=0.01, size=1 + X.shape[1])
        while np.sum(np.absolute(self.w_)) == 0:  # Make sure weights are not all zero
            self.w_ = rng.normal(scale=0.01, size=1 + X.shape[1])

        # Initialise errors
        self.errors_ = np.array([])

        # Update weights
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) * np.append([1], xi)
                self.w_ += update
                errors += (update != 0).sum()
            self.errors_ = np.append(self.errors_, errors)
        return self

    def net_input(self, X: np.array) -> np.array:
        """
        Computes the net input of the input example
        :param X: nx(m+1) example-matrix
        :return: net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]  # Split up into w_[1:] and w[0] to handle matrix inputs of X

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class (in {positive, negative}) of a given input example
        :param X: 1x(m+1) example-matrix
        :return: 1 for predicting the positive class and -1 otherwise
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    from config import PATH_TO_DATA
    import os

    X = np.load(os.path.join(PATH_TO_DATA, "training_examples_iris.npy"))
    y = np.load(os.path.join(PATH_TO_DATA, "training_target_iris_setosa.npy"))

    bcp = BinaryClassPerceptron()
    bcp.fit(X, y)
    pred = bcp.predict(X)
    print(pred)


