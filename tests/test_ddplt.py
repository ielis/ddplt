import matplotlib.pyplot as plt
import numpy as np

from ddplt import plot_confusion_matrix

import unittest

interactive = False


class TestDdplt(unittest.TestCase):

    def setUp(self) -> None:
        from sklearn import svm, datasets
        from sklearn.model_selection import train_test_split

        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        self.class_names = iris.target_names

        # Split the data into a training set and a test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)

        # Run classifier, using a model that is too regularized (C too low) to see
        # the impact on the results
        clf = svm.SVC(kernel='linear', C=0.01)
        self.y_pred = clf.fit(self.X_train, self.y_train).predict(self.X_test)

    def test_confusion_matrix(self):
        """Test correctness of confusion matrix creation.

        Set `interactive = True` if you want to see the created plot
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax, cm = plot_confusion_matrix(self.y_test, self.y_pred, self.class_names, title="The confusion matrix",
                                       normalize=True, ax=ax[0])

        self.assertIsNone(  # return None if the arrays are almost equal, raise AssertionError otherwise
            np.testing.assert_allclose(cm, np.array([[1., 0., 0.], [0., 0.625, 0.375], [0., 0., 1.]])))
        if interactive:
            fig.tight_layout()
            plt.show()
