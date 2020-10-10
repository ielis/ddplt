import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddplt.heatmaps import plot_correlation_heatmap, plot_confusion_heatmap
from .test_base import TestBase

# set this to true if you want to see the plots
interactive = False


class TestDdplt(TestBase):

    def setUp(self) -> None:
        super(TestDdplt, self).setUp()

    def test_confusion_heatmap(self):
        """Test correctness of confusion heatmap creation.

        Set `interactive = True` if you want to see the created plot
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax, cm = plot_confusion_heatmap(self.y_test, self.y_pred, self.class_names, title="The confusion matrix",
                                        normalize=True, ax=ax[0])

        self.assertIsNone(  # return None if the arrays are almost equal, raise AssertionError otherwise
            np.testing.assert_allclose(cm, np.array([[1., 0., 0.], [0., 0.625, 0.375], [0., 0., 1.]])))
        if interactive:
            fig.tight_layout()
            plt.show()

    def test_corr_heatmap(self):
        """Test correctness of correlation heatmap.

        :return:
        """
        df = pd.DataFrame(data=self.X[:30], columns=self.feature_names)
        print(plot_correlation_heatmap(df))
