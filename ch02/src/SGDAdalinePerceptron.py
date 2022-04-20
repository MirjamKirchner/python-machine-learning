from AdalinePerceptron import AdalinePerceptron
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt


class SGDAdalinePerceptron(AdalinePerceptron):
    """
    Although stochastic gradient descent (SGD) can be considered as an approximation of gradient descent, it typically
    reaches convergence much faster because of the more frequent weight updates. Since each gradient is calculated based
    on a single training example, the error surface is much noisier than in gradient descent, which can also have the
    advantage that SGD can escape shallow local minima more readily if we are working with non-linear cost functions
    [...]. To obtain satisfying results via SGD, it is important to present training data in a random order [...].
    """

    # 1. Update weights incrementally for each training set
    # 2. Shuffle training set for each epoch
    # 3. partial_fit function for online learning
    # Functions that can be taken from Adaline: predict, activation, super.__init__
    # Function that cannot be taken from Adaline fit
    # Additional functions _shuffle, _initialise_weights, fit_partially
    def __init__(self, eta: float = 0.01, n_iter: int = 50, shuffle: bool = True, random_state: int = 1):
        """
        An artifical neuron (perceptron) can execute a binary classification task on the input data
        :param eta: the learning rate, typically between 0.0 and 1.0
        :param n_iter: maximum number of passes over the training set
        :param shuffle: a boolean value that if True indicates that the order in which the training examples are
        presented to the neuron is randomised before each epoch
        :param random_state: a seed to initialise a random number generator
        """
        super().__init__(eta=eta, n_iter=n_iter, random_state=random_state)
        self.shuffle = shuffle
        self.w_initialized = False
        self.rng = np.random.default_rng(self.random_state)

    def _initialise_weights(self, X_num_cols: int):
        # Initialise weights
        self.w_ = self.rng.normal(scale=0.01, size=1 + X_num_cols)
        while np.sum(np.absolute(self.w_)) == 0:  # Make sure weights are not all zero
            self.w_ = self.rng.normal(scale=0.01, size=1 + X_num_cols)
        self.w_initialized = True

    def _shuffle(self, X: np.array, y: np.array):
        r = self.rng.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        net_input = self.net_input(xi)
        error = target - self.activation(net_input)

        # Update weights
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * xi.dot(error)

        # Calculate standard error
        sse = 0.5 * error ** 2
        return sse

    def fit(self, X: np.array, y: np.array):
        # Initialise weights
        self._initialise_weights(X.shape[1])
        # Initialise cost
        self.cost_ = np.array([])
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            sse = np.array([])
            for xi, target in zip(X, y):
                sse = np.append(sse, self._update_weights(xi, target))
            self.cost_ = np.append(self.cost_, np.mean(sse))
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialise_weights(X.shape[1])
        if self.cost_ is None:
            # Initialise cost
            self.cost_ = np.array([])
        sse = np.array([])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                sse = np.append(sse, self._update_weights(xi, target))
            self.cost_ = np.append(self.cost_, np.mean(sse))
        else:
            sse = np.append(sse, self._update_weights(X, y))
            self.cost_ = np.append(self.cost_, np.mean(sse))
        return self


if __name__ == "__main__":
    from config import PATH_TO_DATA
    from plotting import plot_decision_surface, plot_comparison_of_adaline_learning_rates
    import os

    X = np.load(os.path.join(PATH_TO_DATA, "training_examples_iris.npy"))
    # Standardisation of X
    X_std = X.copy()
    X_std[:, 0] = (X_std[:, 0] - np.mean(X_std[:, 0])) / np.std(X_std[:, 0])
    X_std[:, 1] = (X_std[:, 1] - np.mean(X_std[:, 1])) / np.std(X_std[:, 1])
    y = np.load(os.path.join(PATH_TO_DATA, "training_target_iris_setosa.npy"))

    sgd_ap = SGDAdalinePerceptron(n_iter=15, eta=0.01, random_state=1)
    sgd_ap.fit(X_std, y)
    plot_decision_surface(X_std, y, sgd_ap)
    sns.lineplot(x=range(1, sgd_ap.cost_.shape[0] + 1), y=sgd_ap.cost_)
    plt.show()
