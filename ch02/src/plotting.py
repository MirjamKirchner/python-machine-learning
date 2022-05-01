import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from BinaryClassPerceptron import BinaryClassPerceptron
from AdalinePerceptron import AdalinePerceptron
from config import configure_plots, PATH_TO_FIGURES
configure_plots()


def plot_training_data(X: np.array, y: np.array, save_fig=True):
    g = sns.scatterplot(x=X[:, 0], y=X[:, 1], style=np.where(y == 1, "versicolor", "setosa"))
    g.set(title="The perceptron model is appropriate since the Iris-classes, setosa and versicolor, are linearly"
                "separable", xlabel="sepal length [cm]", ylabel="sepal width [cm]")
    if save_fig:
        plt.savefig(os.path.join(PATH_TO_FIGURES, "binary_perceptron-training_data.png"), transparent=True,
                    bbox_inches="tight")
    plt.show()
    return g


def plot_number_of_updates(bcp: BinaryClassPerceptron, save_fig=True):
    g = sns.lineplot(x=range(1, bcp.errors_.shape[0]+1), y=bcp.errors_)
    g.set(title="After 4 epochs, the weights are no longer updated", xlabel="# epochs", ylabel="# weight updates")
    plt.yticks(range(int(np.amax(bcp.errors_)+1)))
    if save_fig:
        plt.savefig(os.path.join(PATH_TO_FIGURES, "binary_perceptron-number_of_updates.png"), transparent=True,
                    bbox_inches="tight")
    plt.show()
    return g


def plot_comparison_of_adaline_learning_rates(X: np.array, y: np.array, etas=None, n_iter=20, save_fig=True):
    if etas is None:
        etas = [0.0001, 0.005, 0.1]

    # Train Adaline perceptrons with different learning rates
    ap_small_eta = AdalinePerceptron(eta=etas[0], n_iter=n_iter)
    ap_small_eta.fit(X, y)
    ap_medium_eta = AdalinePerceptron(eta=etas[1], n_iter=n_iter)
    ap_medium_eta.fit(X, y)
    ap_large_eta = AdalinePerceptron(eta=etas[2], n_iter=n_iter)
    ap_large_eta.fit(X, y)


    # Comparison of learning rates
    fig, axes = plt.subplots(1, 3)
    sns.lineplot(x=range(1, ap_small_eta.cost_.shape[0] + 1), y=ap_small_eta.cost_, ax=axes[0])
    axes[0].set_title("Learning rate = " + str(etas[0]))
    sns.lineplot(x=range(1, ap_medium_eta.cost_.shape[0] + 1), y=ap_medium_eta.cost_, ax=axes[1])
    axes[1].set_title("Learning rate = " + str(etas[1]))
    sns.lineplot(x=range(1, ap_large_eta.cost_.shape[0] + 1), y=ap_large_eta.cost_, ax=axes[2])
    axes[2].set_title("Learning rate = " + str(etas[2]))
    axes[2].set_yscale("log")
    fig.suptitle("A too small learning rate requires a large number of epochs to converge to the global minimum cost.\n A too large learning rate overshoots the global minimum cost.")

    if save_fig:
        plt.savefig(os.path.join(PATH_TO_FIGURES, "adaline_perceptron-comparison-learning-rates.png"), transparent=True,
                    bbox_inches="tight")

    plt.show()


def plot_decision_surface(X: np.array, y: np.array,  classifier, resolution=0.02, save_fig=True, xlabel="x",
                          ylabel="y", is_test=None):
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    X_mesh = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(X_mesh)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3)
    if is_test is None:
        df = pd.DataFrame(np.concatenate((X, y.reshape((-1, 1))), axis=1), columns=["X0", "X1", "Class"])
        g = sns.scatterplot(data=df, x="X0", y="X1", style="Class")
    else:
        df = pd.DataFrame(np.concatenate((X, y.reshape((-1, 1)), is_test.reshape((-1, 1))), axis=1), columns=["X0", "X1", "Class", "Is_test_set"])
        g = sns.scatterplot(data=df, x="X0", y="X1", style="Class", hue="Is_test_set")
    g.set_xlim(xx1.min(), xx1.max())
    g.set_ylim(xx2.min(), xx2.max())
    g.set(title="Decision surface of the " + str(type(classifier).__name__) + "-classifier", xlabel=xlabel,
          ylabel=ylabel)
    if save_fig:
        plt.savefig(os.path.join(PATH_TO_FIGURES, str(type(classifier).__name__) + "-decision-surface.png"),
                    transparent=True, bbox_inches="tight")
    plt.show()
    return g


if __name__ == "__main__":
    from config import PATH_TO_DATA
    import os

    X = np.load(os.path.join(PATH_TO_DATA, "training_examples_iris.npy"))
    y = np.load(os.path.join(PATH_TO_DATA, "training_target_iris_setosa.npy"))
    plot_training_data(X, y)

    bcp = BinaryClassPerceptron(n_iter=10)
    bcp.fit(X, y)
    plot_number_of_updates(bcp)
    plot_decision_surface(X, y, bcp, xlabel="sepal length [cm]", ylabel="sepal width [cm]")

