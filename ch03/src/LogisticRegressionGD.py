from AdalinePerceptron import AdalinePerceptron
from scipy.special import expit
import numpy as np
from sklearn.datasets import load_iris

class LogisticRegressionGD(AdalinePerceptron):
    def activation(self, X: np.array) -> np.array:
        """
        Compute sigmoid activation function
        :param X: nxm matrix where rows represent examples, and columns features
        :return: Activation of X using a sigmoid function, i.e. return 1/(1+exp(-X))
        """
        return expit(X)

    def fit(self, X: np.array, y: np.array):
        """
                Fits the weights of the perceptron to minimise the prediction error using gradient decent
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
            output = self.activation(net_input)
            errors = y - output

            # Update weights
            self.w_[0] += self.eta * np.sum(errors)
            self.w_[1:] += self.eta * np.dot(X.T, errors)

            # Update cost per iteration
            loglik = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output))
            self.cost_ = np.append(self.cost_, loglik)

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class (in {positive, negative}) of a given input example
        :param X: 1x(m+1) example-matrix
        :return: 1 for predicting the positive class and 0 otherwise
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)

if __name__ == "__main__":
    import sys
    sys.path.append("..\\..\\ch02\\src")
    from plotting import plot_decision_surface

    X = np.load("../../ch02/data/training_examples_iris.npy")
    y = np.load("../../ch02/data/training_target_iris_setosa.npy")
    y = np.where(y == -1, 0, y)
    X_std = X.copy()
    X_std[:, 0] = (X_std[:, 0] - np.mean(X_std[:, 0])) / np.std(X_std[:, 0])
    X_std[:, 1] = (X_std[:, 1] - np.mean(X_std[:, 1])) / np.std(X_std[:, 1])
    logreg_perceptron = LogisticRegressionGD(eta=0.005, n_iter=1000)
    logreg_perceptron.fit(X_std, y)
    plot_decision_surface(X_std, y, logreg_perceptron, xlabel="petal length [standardized]",
                          ylabel="petal width [standardized]")