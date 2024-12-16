"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.figure import Figure

from trajectopy.core.alignment.parameters import AlignmentParameters


# see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, aspect="auto", **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# see https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_correlation_heatmap(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> Figure:
    """Plots the correlation heatmap of the alignment parameters using matplotlib.

    Args:
        estimated_parameters (AlignmentParameters): Estimated parameters.
        enabled_only (bool, optional): Whether to consider only enabled parameters. Defaults to True.

    Returns:
        plt.Figure: Correlation heatmap figure.
    """
    covariance_matrix = estimated_parameters.get_covariance_matrix(enabled_only=enabled_only)
    std_devs = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    np.fill_diagonal(correlation_matrix, np.nan)
    fig, ax = plt.subplots()
    ax.grid(False)
    im, _ = _heatmap(
        correlation_matrix,
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        ax=ax,
        cmap="coolwarm",
        cbarlabel="Correlation",
        cbar_kw={"format": "%.2f"},
    )
    _annotate_heatmap(im, valfmt="{x:.2f}")
    ax.set_aspect("auto")

    plt.tight_layout()
    return fig


def plot_covariance_heatmap(estimated_parameters: AlignmentParameters, enabled_only: bool = True) -> Figure:
    """Plots the covariance heatmap of the alignment parameters using matplotlib.

    Args:
        estimated_parameters (AlignmentParameters): Estimated parameters.
        enabled_only (bool, optional): Whether to consider only enabled parameters. Defaults to True.

    Returns:
        plt.Figure: Covariance heatmap figure.
    """
    covariance_matrix = estimated_parameters.get_covariance_matrix(enabled_only=enabled_only)
    fig, ax = plt.subplots()
    ax.grid(False)
    im, _ = _heatmap(
        covariance_matrix,
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        estimated_parameters.params_labels(enabled_only=enabled_only, lower_case=True),
        ax=ax,
        cmap="coolwarm",
        cbarlabel="Covariance",
        cbar_kw={"format": "%.2f"},
    )
    _annotate_heatmap(im, valfmt="{x:.3f}")
    ax.set_aspect("auto")
    plt.tight_layout()
    return fig
