import pkg_resources

__version__ = pkg_resources.get_distribution('ddplt').version

from .heatmaps import plot_confusion_heatmap, plot_correlation_heatmap
