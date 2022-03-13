import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from BinaryClassPerceptron import BinaryClassPerceptron
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


def plot_decision_surface(X: np.array, y: np.array,  classifier, resolution=0.02, save_fig=True, xlabel="x",
                          ylabel="y"):
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    X_mesh = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(X_mesh)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3)
    g = sns.scatterplot(x=X[:, 0], y=X[:, 1], style=y)
    g.set_xlim(xx1.min(), xx1.max())
    g.set_ylim(xx2.min(), xx2.max())
    g.set(title="Decision surface of the " + str(type(classifier).__name__) + "-classifier", xlabel=xlabel,
          ylabel=ylabel)
    if save_fig:
        plt.savefig(os.path.join(PATH_TO_FIGURES, "binary_perceptron-decision-surface.png"), transparent=True,
                    bbox_inches="tight")
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

