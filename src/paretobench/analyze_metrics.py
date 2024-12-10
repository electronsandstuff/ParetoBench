from scipy.stats import ranksums
from typing import Union, Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
import paretobench

from .exceptions import UnknownProblemError
from .problem import Problem


def normalize_problem_name(name: str) -> str:
    """
    Standardizes the format (whitespace, order of parameters, and default parameters) of problem names so that they can be
    referenced easily tables of metrics (for example).

    If the problem is not registered to ParetoBench, the name is simply passed through unchanged.

    Parameters
    ----------
    name : str
        The problem name

    Returns
    -------
    str
        The normalized problem name
    """
    try:
        return Problem.from_line_fmt(name).to_line_fmt()
    except UnknownProblemError:
        return name


def get_index_before(xv: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """
    For every element in the "query array" xq, finds the index of the value in the sorted  "values array" xv which is closest to
    the query without exceeding its value.

    Parameters
    ----------
    xv : np.ndarray
        The sorted array of values
    xq : np.ndarray
        The query array

    Returns
    -------
    np.ndarray
        The indices (same shape as xq)
    """
    # Check that all query values are valid (larger than at least one element in xv)
    if np.any(xq < np.min(xv)):
        raise ValueError("At least one value in the queries (xq) is less than all values in xv. Cannot continue.")

    # Get the indices
    return np.maximum(np.searchsorted(xv, xq, side="right") - 1, 0)


def get_metric_names(df: pd.DataFrame) -> List[str]:
    """
    From a table containing the calculated metrics (and other columns) get the list of the metric names.

    Parameters
    ----------
    df : Dataframe
        The table of metric values
    """
    nm_keys = [
        "problem",
        "fevals",
        "run_idx",
        "pop_idx",
        "exp_idx",
        "fname",
        "exp_name",
    ]
    return [x for x in df.columns.to_list() if x not in nm_keys]


def apply_feval_cutoff(df: pd.DataFrame, max_feval: int = 300) -> pd.DataFrame:
    """
    Trims the rows in the dataframe from `eval_metrics_mult` to only include those which are associated with the last reporting
    interval before the optimization algorithm exceeded the specified number of function evaluations.

    Parameters
    ----------
    df : Dataframe
        The metric data
    max_feval : int, optional
        Cutoff of how many function evaluations are allowed, by default 300
    """
    df = df[df["fevals"] <= max_feval]
    idx = df.groupby(by=["run_idx", "exp_idx", "problem"])["fevals"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def aggregate_metrics_feval_budget(
    df: pd.DataFrame,
    max_feval: int = 300,
    metric_direction_override: Union[None, Dict[str, str]] = None,
    wilcoxon_p: float = 0.05,
    wilcoxon_idx: Union[int, None] = None,
) -> pd.DataFrame:
    """
    Calculate aggregate statistics from the metric dataframe generated with `eval_metrics_experiments`. The aggregation is
    performed on the last generation in each optimization run before it exceeded the specified maximum number of function
    evaluations. Mean, std. deviation and other statistics of the metric values at this generation are recorded.

    The Wilcoxon rank sum test is used to discern which runs are the "best" across a given problem and metric. Optionally, the
    Wilcoxon rank sum test can also be used to generate comparisons with a specific run (selected by its `exp_idx`). This is
    useful for benchmarking a new optimization algorithm.

    The output table will have hierarchical columns. The top level colunms are the metric names from the orginal table. The next
    level of columns are aggregate values. These are the following
     * mean: The mean value of the metric
     * std: The standard deviation of the metric's values
     * wilcoxon_best: boolean, are we the best value (small wins) of the metric for this problem
     * wilcoxon_comp: optional, has the value '+', '-', '='. Comparison of this metric value against the specified container
     * Other statistical values may be included

    Parameters
    ----------
    df : Dataframe
        The dataframe of metric values from `eval_metrics_experiments`
    max_feval : int
        Maximum number of function evaluations
    metric_direction_override : dict
        A mapping of metric names to '+' or '-' to indicate whether bigger is better (+) or smaller is better (-)
    wilcoxon_idx : int/None, optional
        The value of `exp_idx` to use as the reference in the Wilcoxon rank sum test comparisons, by default None
    wilcoxon_p : float, optional
        p value threshold for the rank sum test, by default 0.05

    Returns
    -------
    Dataframe
        A table with aggregated metric values
    """
    # Don't mutate the dataframe
    df = df.copy()

    # Set up the metric "directions"
    directions = {
        "igd": "-",
        "hypervolume": "+",
    }
    if metric_direction_override is not None:
        directions.update(metric_direction_override)

    # Clean up the problem names and cutoff the fevals
    df["problem"] = df.apply(lambda x: normalize_problem_name(x["problem"]), axis=1)
    df = apply_feval_cutoff(df, max_feval)

    def get_wilcoxon_comparison(x, metric):
        """
        Given the grouped evaluations for this problem and run, compare ourself with the "reference" run
        """
        # If we are the reference run don't perform the comparison against ourself
        if df.loc[x.index[0]]["exp_idx"] == wilcoxon_idx:
            return ""

        # Get the problem name
        problem = df.loc[x.index[0]]["problem"]

        # Get the values for the metric on this problem from the container we are comparing agianst
        y = df.loc[(df["exp_idx"] == wilcoxon_idx) & (df["problem"] == problem)][metric]

        # Use the stats test to compare values
        if ranksums(x.to_numpy(), y.to_numpy(), "less")[1] < wilcoxon_p:
            return {"-": "+", "+": "-"}[directions[metric]]
        if ranksums(x.to_numpy(), y.to_numpy(), "greater")[1] < wilcoxon_p:
            return directions[metric]
        return "="

    def is_best(x, metric):
        """
        Given one of the groupby objects, is this collection of evaluations one of the best in the table for this problem.
        """
        # Get the problem name
        problem = df.loc[x.index[0]]["problem"]

        # Go through each container figuring out if we are the best
        for n in df["exp_idx"].unique().tolist():
            y = df.loc[(df["exp_idx"] == n) & (df["problem"] == problem)][metric]

            if not len(y):
                continue
            if (
                ranksums(
                    x.to_numpy(),
                    y.to_numpy(),
                    {"-": "greater", "+": "less"}[directions[metric]],
                )[1]
                < wilcoxon_p
            ):
                return False
        return True

    # Get metric names and validate
    metrics = get_metric_names(df)
    for metric in metrics:
        if metric not in directions:
            raise ValueError(
                f'Must specify "direction" of metric with `metric_direction_override`: not defined for "{metric}"'
            )

    # Create the set of functions which we'll apply to aggregate the values
    agg_funs = {}
    for m in metrics:
        funs = [
            "mean",
            "std",
            "median",
            ("pct_2.5", lambda x: np.percentile(x, 2.5, axis=0)),
            ("pct_97.5", lambda x: np.percentile(x, 97.5, axis=0)),
            ("wilcoxon_best", lambda x: is_best(x, m)),
        ]
        (wilcoxon_idx is None) or funs.append(("wilcoxon_comp", lambda x: get_wilcoxon_comparison(x, m)))
        agg_funs[m] = funs

    # Apply the aggregation and return
    by = ["problem", "exp_idx"]
    by.append("fname")
    if "exp_name" in df.columns:
        by.append("exp_name")
    return df.groupby(by).agg(agg_funs)


def construct_metric_comparison_table(
    df: pd.DataFrame,
    metric: Union[str, None] = None,
    problem_params: Union[None, List[Union[str, Tuple[str, Callable], Tuple[str, str]]]] = None,
    mean_fmt_kwargs: Union[None, Dict[str, Any]] = None,
    std_fmt_kwargs: Union[None, Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Constructs a table with the problems as the index and the experiments as the columns with a summary of one of the metrics as
    the values along with comparisons and highlighting the best values in each row.

    The index will always include the problem name. Additional information about the problem (such as m, the number of
    objectives) can be included in the index. This is specified by a list of tuples of the names of the parameters and getter
    functions which retrieve their values from a problem object. To make things easier, `problem_params` can also accept a
    string in the tuple in place of a function which will pull the attribute by that name from the problem object. You can also
    use a string to replace the entire tuple which will get the attribute of that name and also set the index's name to its
    value.

    Parameters
    ----------
    df : Dataframe
        The aggregated metric dataframe
    metric : str
        The column name of the metric from which data will be extracted. Defaults to first metric in `df.columns`.
    problem_params : str/List[str/Tuple[str, str/function]], optional
        Tuples representing the name of one of the problem parameters to be placed into the index and a function to get its
        value. You can also use a string in place of the function to specify an attribute from the problem object. Any tuple can
        also be replaced by a string. The string x will get interpreted as the tuple (x, x).
    mean_fmt_kwargs : dict, optional
        Keyword arguments passed to `np.format_float_scientific(...)` when formatting the mean metric values in the table
    std_fmt_kwargs : str, optional
        Keyword arguments passed to `np.format_float_scientific(...)` when formatting the std. deviations

    Returns
    -------
    Dataframe
        The comparison table
    """
    # Avoid mutating original dataframe
    df = df.copy()

    # Empty problem params
    if problem_params is None:
        problem_params = []

    # Default formatting
    if mean_fmt_kwargs is None:
        mean_fmt_kwargs = {"precision": 3, "exp_digits": 1}
    if std_fmt_kwargs is None:
        std_fmt_kwargs = {"precision": 3, "exp_digits": 1}

    # Set unique to False by default to keep all digits
    for kwargs in [mean_fmt_kwargs, std_fmt_kwargs]:
        kwargs["unique"] = kwargs.get("unique", False)

    # Deal with default metric
    if metric is None:
        metric = df.columns[0][0]

    # Expand the problem params
    if isinstance(problem_params, str):
        problem_params = [problem_params]
    problem_params_cleaned = []
    for val in problem_params:
        if isinstance(val, str):
            val = (val, val)
        if isinstance(val[1], str):
            val = (val[0], lambda x, y=val[1]: getattr(x, y))
        problem_params_cleaned.append(val)
    problem_params = problem_params_cleaned

    def get_cell(x):
        """
        Used to convert a row in the aggregate dataframe into a string for the combined table
        """
        # Handle nans
        if not np.isfinite(x[(metric, "mean")]):
            return "-"

        # Handle the floating point numbers
        mean = np.format_float_scientific(x[(metric, "mean")], **mean_fmt_kwargs)
        std = np.format_float_scientific(x[(metric, "std")], **std_fmt_kwargs)

        # Construct the string for this cell in the table
        ret = f"{ mean } ({ std })"

        # Add the comparison, if provided
        if (metric, "wilcoxon_comp") in x and x[(metric, "wilcoxon_comp")]:
            ret = f'{ ret } { x[(metric, "wilcoxon_comp")] }'

        # Handle bolding the cell if we are the best in the row
        if x[(metric, "wilcoxon_best")]:
            ret = f"\\cellbold { ret }"
        return ret

    # Convert the metrics into text for the row
    df["cell_txt"] = df.apply(get_cell, axis=1)

    # Pivot so we get a table of just the txt values
    df = df.pivot_table(
        index="problem",
        columns=("exp_name" if "exp_name" in df.index.names else "exp_idx"),
        values=["cell_txt"],
        aggfunc=lambda x: x,
        fill_value="-",
    )

    # Remove the unecessary levels of columns (and its name)
    df.columns = df.columns.droplevel(level=[0, 1])
    df.columns.name = None

    # Construct the indices and names
    indices = []
    for x in df.index:
        # Get the problem object
        prob = paretobench.Problem.from_line_fmt(x)

        # Construct the index tuple
        index = [prob.__class__.__name__]
        for _, v in problem_params:
            index.append(v(prob))
        indices.append(tuple(index))

    # Set the indices and their names
    df.index = pd.MultiIndex.from_tuples(indices)
    df.index = df.index.set_names(["Problem"] + [n for n, _ in problem_params])

    # Sort by the index lexicographically
    df = df.sort_index(axis=0)
    return df


def gather_metric_values_stepwise(df: pd.DataFrame, metric: str, fevals: int) -> np.ndarray:
    """
    Gets the latest generation values of the metric before exhausting the function evaluation budgets in the array `fevals`.
    Operates on a table containing a single optimizer evaluation.

    Parameters
    ----------
    df : Dataframe
        The table to get metric values from
    metric : str
        The name of the metric to pull from
    fevals : np.ndarray
        The values to evaluate on

    Returns
    -------
    np.ndarray
        The metric values
    """
    # `get_index_before` assumes sorted values
    df_sorted = df.sort_values("fevals")

    # Perform the stepwise interpolation
    return df_sorted[metric].values[get_index_before(df_sorted["fevals"], fevals)]


def aggregate_metric_series_apply_fn(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a dataframe containing a single problem and run (with multiple evaluations) collect statistics of the metrics over all
    evaluations at each value of `fevals`. This function is used by `aggregate_metric_series`.

    Parameters
    ----------
    df : DataFrame
        The table containing the problems and metrics

    Returns
    -------
    pd.DataFrame
        The mean and std. deviation of the metrics and the points they were evaluated at
    """
    # The values of feval where we will evaluate the metrics
    fevals = np.sort(df["fevals"].unique())

    # Filter so that the evaluation points are greater than the starting value feval for each run
    # Note: lambda accepts keyword arguments as a hack to prevent an error with (what I think) is different pandas
    # versions. In earlier versions of pandas apply will try to pass `include_groups` as a kwarg to the lambda. However, in
    # later versions we need to set it here to avoid a deprecation warning.
    min_feval = max(df.groupby("run_idx").apply(lambda x, **kw: x["fevals"].min(), include_groups=False))
    fevals = fevals[fevals >= min_feval]
    fevals = fevals.astype(int)

    # The metric names
    metrics = get_metric_names(df)

    # Calculate each of the series statistics collecting them into a dict
    data = {}
    for metric in metrics:
        # The "interpolated" (step-wise) metric values at each of the function evaluations in `fevals`
        def apply_fn(g, **kw):
            return gather_metric_values_stepwise(g, metric, fevals)

        vals = df.groupby("run_idx").apply(apply_fn, include_groups=False)
        vals = np.array(
            vals.to_list()
        )  # Each row contains the values of the metric for a single evaluation at `fevals`

        # Calculate statistics on it and add to the dict of data
        data[(metric, "mean")] = np.mean(vals, axis=0)
        data[(metric, "median")] = np.median(vals, axis=0)
        data[(metric, "pct_2.5")] = np.percentile(vals, 2.5, axis=0)
        data[(metric, "pct_97.5")] = np.percentile(vals, 97.5, axis=0)
        data[(metric, "std")] = np.std(vals, axis=0)

    # Return as a dataframe
    df = pd.DataFrame(data, index=fevals)
    df.index.name = "fevals"
    return df


def aggregate_metric_series(df: pd.DataFrame, keep_filename: bool = False) -> pd.DataFrame:
    """
    Performs aggregation of metrics calculated on a single problem and within a single experiment for all fevals. The input
    dataframe should have the schema output by `eval_metrics_experiments`. This will have the value of metrics versus the number
    of function evaluations for multiple problems, multiple runs, and multiple evaluations in each of the runs.

    This analysis will calculate statistics on the metrics at each value of `fevals` for the runs. If the array of
    `fevals` is not the same between runs which are being aggregated then it happens at all unique values of `fevals`
    across the runs. These values are used as a cutoff and at any given number of function evaluations the data is being
    averaged at, all datapoints in a run up until that number of function evaluations has been exhausted are considered.

    Parameters
    ----------
    df : Dataframe
        The table containing metric evaluation data
    keep_filename : bool, optional
        Whether to pass through filenames to the output, by default False

    Returns
    -------
    Dataframe
        Table with the aggregated series
    """
    # Apply the aggregation and return
    by = ["problem", "exp_idx"]
    if keep_filename:
        by.append("filename")
    if "exp_name" in df.columns:
        by.append("exp_name")

    # The lambda has the parameter `kw` as a hack (see note in `aggregate_metric_series_apply_fn`)
    return df.groupby(by).apply(lambda x, **kw: aggregate_metric_series_apply_fn(x), include_groups=False)


def comparison_table_to_latex(df: pd.DataFrame) -> str:
    r"""
    Converts the table created from `construct_metric_comparison_table` into latex with nice formatting. Headers are centered
    and bolded. A summary row of the comparisons in the style of IEEE Transactions on Evolutionary Computation is added at the
    bottom.

    The table will be made in the `tabular` environment and uses the `multirow` package. A custom command for bolding of the
    cells should be defined at the top of your file.

    ```
    % Loaded for benchmark tables
    \usepackage{multirow}

    % For bolding the cells in the tables
    \newcommand{\cellbold}{\cellcolor{gray!50}}
    ```

    Parameters
    ----------
    df : DataFrame
        The table from `construct_metric_comparison_table`

    Returns
    -------
    str
        The latex table
    """
    # Get the latex representation
    latex_str = df.to_latex(multirow=True, escape=False, index=True).replace("multirow[t]", "multirow")

    # Count the number of each comparison for the columns and construct the summary to go at the bottom
    summary_str = r" \multicolumn{1}{c}{%d/%d/%d} "
    comparisons = df.map(lambda x: (x[-1] if len(x) > 4 else "")).apply(pd.Series.value_counts).fillna(0)
    comparisons = comparisons.apply(
        lambda x: summary_str % (int(x.get("+", 0)), int(x.get("-", 0)), int(x.get("=", 0)))
    )

    # Construct text for the final row
    comparison_cells = [(r" \multicolumn{%d}{c}{+/-/=} " % df.index.nlevels)] + comparisons.values[:-1].tolist() + [" "]
    summary_line = "&".join(comparison_cells) + r"\\"

    # Bold and center the header
    lines = []
    header = False
    header_cells = []
    for ln in latex_str.split("\n"):
        if r"\midrule" in ln:
            # Inject the header cells at the top
            lines.append("&".join(header_cells) + r"\\")
            header = False
        elif header:
            # Get the individual cells in the header and center/bold them
            ls = ln.replace(r"\\", "").split("&")
            ls = [r"\multicolumn{1}{c}{\textbf{" + lns + r"}}" if lns.strip() else lns for lns in ls]

            # Merge into the overall set of header cells
            if not header_cells:
                header_cells = ls
            else:
                for idx, (a, b) in enumerate(zip(ls, header_cells)):
                    header_cells[idx] = a + b
            continue
        elif r"\toprule" in ln:
            header = True

        # Inject the summary line into the table
        if r"\bottomrule" in ln:
            lines.append(summary_line)
        lines.append(ln)
    latex_str = "\n".join(lines)

    # Return it
    return latex_str.replace("=", r"$\approx$")
