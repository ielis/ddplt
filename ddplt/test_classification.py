import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from ddplt.classification import draw_roc_cv, draw_prc_cv, draw_roc_prc_cv
from .test_base import TwoClassClassificationTestBase

# set this to true if you want to see the plots
interactive = False


class TestClassification(TwoClassClassificationTestBase):

    def setUp(self) -> None:
        super(TestClassification, self).setUp()
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    def test_draw_roc_cv(self):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        draw_roc_cv(self.estimator, self.X, self.y, self.cv, ax=ax)

        if interactive:
            plt.show()

    def test_draw_prc_cv(self):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        draw_prc_cv(self.estimator, self.X, self.y, self.cv, ax=ax)

        if interactive:
            plt.show()

    def test_draw_roc_prc_cv(self):
        fig, (l, r) = plt.subplots(1, 2, figsize=(6, 4), dpi=100)
        draw_roc_prc_cv(self.estimator, self.X, self.y, self.cv, roc_ax=l, prc_ax=r)

        if interactive:
            plt.show()
