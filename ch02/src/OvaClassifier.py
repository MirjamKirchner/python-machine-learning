import numpy as np
from copy import deepcopy


class OvaClassifier:
    def __init__(self, binary_classifier):
        """
        The One-versus-All (OvA) classifier takes a binary classifier as input and combines them to a multi-class
        classifier
        :param binary_classifier: A binary classifier with functions fit, predict, and confidence_score
        """
        self.__binary_classifier = binary_classifier

    def fit(self, X: np.array, y: np.array):
        """
        Fits a binary classifier for each class in y.
        :param X: nxm matrix where rows represent examples, and columns features
        :param y: a nx1 target-matrix (vector)
        :return: None
        """

        # Preprocess target-data
        unique_target_values = np.unique(y)
        self.class_names_ = unique_target_values
        Y = [np.where(y == target_value, -1, 1) for target_value in unique_target_values]

        # Initialise binary classifiers
        binary_classifiers = [deepcopy(self.__binary_classifier) for i in range(unique_target_values.shape[0])]

        for binary_classifier, y in zip(binary_classifiers, Y):
            binary_classifier.fit(X, y)
        self.binary_classifiers_ = binary_classifiers

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class of a given input example using a set of binary classifiers. The binary classifier with the
        highest confidence score 'wins' the prediction
        :param X: 1x(m+1) example-matrix
        :return: A numpy array with predicted class labels. Each class is represented as an integer value # TODO return class names instead of numbers --> make sure plot_decision_surface can handle str as prediciton
        """
        confidence_scores = [binary_classifier.confidence_score(X) for binary_classifier in self.binary_classifiers_]
        confidence_scores = np.stack(confidence_scores, axis=1)
        return np.argmax(confidence_scores, axis=1)


if __name__ == "__main__":
    from BinaryClassPerceptron import BinaryClassPerceptron
    from data_preprocessing import load_training_data
    from plotting import plot_decision_surface

    X = load_training_data()
    y = X.pop("target")
    X = X.to_numpy()[:, [0, 2]]
    y = y.to_numpy()

    bcp = BinaryClassPerceptron()
    ova = OvaClassifier(bcp)
    ova.fit(X, y)
    ova.predict(X)
    plot_decision_surface(X, y, ova, save_fig=False)



