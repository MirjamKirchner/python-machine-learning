import numpy as np
from BinaryClassPerceptron import BinaryClassPerceptron


class AdalinePerceptron(BinaryClassPerceptron):
    """The main difference of the Adaline perceptron to Rosenblatt's perceptron (BinaryClassPerceptron) is that weights
    are updated based on a linear activation rather than a unit step function."""

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        """
        An artifical neuron (perceptron) can execute a binary classification task on the input data
        :param eta: the learning rate, typically between 0.0 and 1.0
        :param n_iter: maximum number of passes over the training set
        :param random_state: a seed to initialise a random number generator
        """
        super().__init__(eta=eta, n_iter=n_iter, random_state=random_state)

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

        # Initialise cost
        self.cost_ = np.array([])

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            errors = y - self.activation(net_input)

            # Update weights
            self.w_[0] += self.eta * np.sum(errors)
            self.w_[1:] += self.eta * np.dot(X.T, errors)

            # Update cost per iteration
            sse = 0.5 * np.sum(errors**2)
            self.cost_ = np.append(self.cost_, sse)

        return self

    def activation(self, X: np.array) -> np.array:
        """
        Compute linear activation
        :param X: nxm matrix where rows represent examples, and columns features
        :return: Linear activation of X using a unit function, i.e. return X
        """
        return X

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class (in {positive, negative}) of a given input example
        :param X: 1x(m+1) example-matrix
        :return: 1 for predicting the positive class and -1 otherwise
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


if __name__ == "__main__":
    from config import PATH_TO_DATA
    from plotting import plot_decision_surface, plot_comparison_of_adaline_learning_rates
    import os

    X = np.load(os.path.join(PATH_TO_DATA, "training_examples_iris.npy"))
    # Standardisation of X
    X_std = X.copy()
    X_std[:, 0] = (X_std[:, 0] - np.mean(X_std[:, 0]))/np.std(X_std[:, 0])
    X_std[:, 1] = (X_std[:, 1] - np.mean(X_std[:, 1])) / np.std(X_std[:, 1])
    y = np.load(os.path.join(PATH_TO_DATA, "training_target_iris_setosa.npy"))

    ap_medium_eta = AdalinePerceptron(eta=0.005)
    ap_medium_eta.fit(X_std, y)
    plot_decision_surface(X_std, y, ap_medium_eta)

    plot_comparison_of_adaline_learning_rates(X_std, y, n_iter=20)





