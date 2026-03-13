import matplotlib.pyplot as plt
from typing import Literal

from paretobench.containers import History
from paretobench.metrics import Metric, eval_metrics


def plot_metric_history(
    hist: History, metric: Metric, x_axis: Literal["fevals", "generation"] = "fevals", fig=None, ax=None
):
    # Calculate the metrics from the history object
    df = eval_metrics(hist, metric)

    # Plot the data
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if x_axis == "fevals":
        x_vals = df["fevals"]
    elif x_axis == "generation":
        x_vals = df["pop_idx"]
    else:
        raise ValueError(f"Unrecognized value for `x_axis`: {x_vals}")

    ax.plot(x_vals, df[metric.name])
    if x_axis == "fevals":
        ax.set_xlabel("Function Evaluations")
    elif x_axis == "generation":
        ax.set_xlabel("Generation")
    ax.set_ylabel(metric.get_plot_label())
