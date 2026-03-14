import matplotlib.pyplot as plt
from typing import Literal

from paretobench.containers import History
from paretobench.metrics import Metric, eval_metrics


def plot_metric_history(
    hist: History,
    metric: Metric,
    x_axis: Literal["fevals", "generation"] = "fevals",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
):
    """
    Evaluate and plot evolution of a metric across the populations within a `History` object.

    Parameters
    ----------
    hist : History
        The genetic algorithm history to plot
    metric : Metric
        The metric to evaluate and plot
    x_axis : Literal["fevals", "generation"]
        What value to use for x-axis in plot
    fig : plt.Figure | None
        Matplotlib figure if plotting to user-provided figure (must also specify axis)
    ax : plt.Axes | None
        Matplotlib axis to place plot into (must also specify figure)

    Returns
    -------
    plt.Figure, plt.Axes
        The matplotlib figure and axis the data was plotted to
    """
    # Calculate the metrics from the history object
    df = eval_metrics(hist, metric)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Grab the x values and axis label
    if x_axis == "fevals":
        x_vals = df["fevals"]
        xlabel = "Function Evaluations"
    elif x_axis == "generation":
        x_vals = df["pop_idx"]
        xlabel = "Generation"
    else:
        raise ValueError(f"Unrecognized value for `x_axis`: {x_vals}")

    # Plot the data
    ax.plot(x_vals, df[metric.name])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.get_plot_label())

    return fig, ax
