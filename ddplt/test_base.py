import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class TestBase(unittest.TestCase):

    def setUp(self) -> None:
        # import some data to play with
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.y_is_setosa = iris.target == 1  # to have a binary classification problem
        self.class_names = iris.target_names
        self.feature_names = iris.feature_names

        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0)

        # Run classifier, using a model that is too regularized (C too low) to see
        # the impact on the results
        self.svc = SVC(kernel='linear', C=0.01)
        self.y_pred = self.svc.fit(self.X_train, self.y_train).predict(self.X_test)


class TwoClassClassificationTestBase(unittest.TestCase):

    def setUp(self) -> None:
        # import some data to play with
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target == 1  # is versicolor (to have a binary classification problem)

        self.estimator = DecisionTreeClassifier()
