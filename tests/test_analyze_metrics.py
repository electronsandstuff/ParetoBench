import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest

from paretobench import eval_metrics_experiments, Problem
from paretobench.analyze_metrics import (
    aggregate_metric_series,
    construct_metric_comparison_table,
    comparison_table_to_latex,
)
from paretobench.analyze_metrics import (
    aggregate_metrics_feval_budget,
    apply_feval_cutoff,
    normalize_problem_name,
)
from .utils import generate_moga_experiments, example_metric


def test_aggregate_metrics_feval_budget():
    """
    Test the aggregate function. Confirm that means and stds are taken correctly and the index data is carried through
    correctly.
    """
    # Create some test objects
    runs = generate_moga_experiments()

    # Get the metric values and aggregate
    df = eval_metrics_experiments(runs, metrics=("test", example_metric))
    agg = aggregate_metrics_feval_budget(df, metric_direction_override={"test": "-"})

    # Assert the right indices show up with the right types
    assert ptypes.is_string_dtype(agg.index.get_level_values("problem"))
    assert ptypes.is_integer_dtype(agg.index.get_level_values("exp_idx"))

    # Assert the right metrics show up
    assert ptypes.is_float_dtype(agg[("test", "mean")])
    assert ptypes.is_float_dtype(agg[("test", "std")])
    assert ptypes.is_float_dtype(agg[("test", "median")])
    assert ptypes.is_float_dtype(agg[("test", "pct_2.5")])
    assert ptypes.is_float_dtype(agg[("test", "pct_97.5")])
    assert ptypes.is_bool_dtype(agg[("test", "wilcoxon_best")])

    # Check the mean and std
    df["problem"] = df.apply(lambda x: normalize_problem_name(x["problem"]), axis=1)
    test_df = apply_feval_cutoff(df)
    for index, row in test_df.groupby(["problem", "exp_idx"]).agg({"test": ["mean", "std"]}).iterrows():
        np.testing.assert_allclose(row[("test", "mean")], agg.loc[index][("test", "mean")])


def test_aggregate_metrics_stats_test():
    """
    Test the statistical comparisons for the metric aggregation function. Several problems are generated and several different
    runs are made. The runs have evaluations with a normally distributed metric. The mean of the values is shifted to known
    positions. We then perform the comparisons and confirm the comparisons based on their centroid location.
    """
    # The centroid locations of the metric values for each problem
    run_locs = {
        "ZDT1": [0, 1, 2, 3],
        "ZDT2": [1, 2, 3, 0],
        "ZDT3": [2, 3, 0, 1],
        "ZDT4": [3, 0, 1, 2],
        "ZDT6": [1, 0, 0, 0],
        "DTLZ1": [0, 0, 0, 1],
    }

    # Construct a dataframe with metric values normally distributed so that idx
    np.random.seed(1)
    rows = []
    for exp_idx in range(4):
        for problem in run_locs:
            for run_idx, val in enumerate(np.random.normal(run_locs[problem][exp_idx], 0.1, size=64)):
                for fevals in range(10):
                    rows.append(
                        {
                            "run_idx": run_idx,
                            "exp_idx": exp_idx,
                            "problem": problem,
                            "fevals": fevals,
                            "test": val if fevals == 8 else 0.0,
                            "fname": "",
                        }
                    )
    df = pd.DataFrame(rows)

    # Aggregate to get stats test values
    agg = aggregate_metrics_feval_budget(df, max_feval=8, wilcoxon_idx=0, metric_direction_override={"test": "-"})

    # Check each problem for the comparisons
    for prob in run_locs:
        # The "normalized" name
        prob_norm = normalize_problem_name(prob)

        for exp_idx, loc in enumerate(run_locs[prob]):
            # Check the wilcoxon comparison
            if exp_idx == 0:
                wilcoxon_comp = ""
            elif loc < run_locs[prob][0]:
                wilcoxon_comp = "+"
            elif loc == run_locs[prob][0]:
                wilcoxon_comp = "="
            else:
                wilcoxon_comp = "-"
            assert agg.loc[prob_norm, exp_idx].iloc[0][("test", "wilcoxon_comp")] == wilcoxon_comp

            # Check if we are one of the best in the row
            assert agg.loc[prob_norm, exp_idx].iloc[0][("test", "wilcoxon_best")] == (loc == min(run_locs[prob]))


def test_feval_cutoff():
    """
    Test the cutoff function by making a table of values with a test metric equalling 1.0 only at fevals=7. The cutoff is made
    for fevals=7 and we check all the metric values.
    """
    # Create a test dataframe (big nasty nested loop)
    rows = []
    for run_idx in range(16):
        for exp_idx in range(4):
            for problem in ["ZDT1", "ZDT2"]:
                for fevals in range(10):
                    rows.append(
                        {
                            "run_idx": run_idx,
                            "exp_idx": exp_idx,
                            "problem": problem,
                            "fevals": fevals,
                            "test": 1.0 if fevals == 7 else 0.0,
                        }
                    )
    df = pd.DataFrame(rows)

    # Run the cutoff function
    df_cutoff = apply_feval_cutoff(df, max_feval=7)
    assert (df_cutoff["test"] == 1.0).all()


@pytest.mark.parametrize(
    "use_names, metric, problem_params",
    [
        (False, None, [("n", lambda x: x.n)]),
        (True, None, [("n", lambda x: x.n)]),
        (True, "test", [("n", lambda x: x.n)]),
        (True, None, [("n", "n")]),
        (True, None, ["n"]),
        (True, None, "n"),
    ],
)
def test_construct_metric_comparison_table(use_names, metric, problem_params):
    """
    Create a comparison table from a set of Experiment objects. Perform some basic checks that the rows and columns show up
    appropriately with and without specification of the names.
    """
    # Create some test objects
    names = ["a", "b", "c", "d"] if use_names else None
    runs = generate_moga_experiments(names=names)

    # Construct the comparison table
    df = eval_metrics_experiments(runs, metrics=("test", example_metric))
    df = aggregate_metrics_feval_budget(df, metric_direction_override={"test": "-"})
    dfm = construct_metric_comparison_table(df, metric, problem_params=problem_params)

    # Check we get the right row and column names
    indices_act = set(("ZDT1", Problem.from_line_fmt(p.problem).n) for p in runs[0].runs)
    indices_tst = set(dfm.index)
    assert indices_act == indices_tst
    assert set(dfm.columns) == set(r.name for r in runs)


def test_aggregate_metrics_series():
    """
    Test the series aggregation function by comparing with the other one (`aggregate_metrics_feval_budget`)
    """
    # Seed the RNG
    np.random.seed(0)

    # Create some test objects
    runs = generate_moga_experiments()

    # Get the metric values and aggregate
    df = eval_metrics_experiments(runs, metrics=("test", example_metric))
    agg = aggregate_metric_series(df)

    # Assert the right indices show up with the right types
    assert ptypes.is_string_dtype(agg.index.get_level_values("problem"))
    assert ptypes.is_integer_dtype(agg.index.get_level_values("exp_idx"))
    assert ptypes.is_integer_dtype(agg.index.get_level_values("fevals"))

    # Assert the right metrics show up
    assert ptypes.is_float_dtype(agg[("test", "mean")])
    assert ptypes.is_float_dtype(agg[("test", "std")])
    assert ptypes.is_float_dtype(agg[("test", "median")])
    assert ptypes.is_float_dtype(agg[("test", "pct_2.5")])
    assert ptypes.is_float_dtype(agg[("test", "pct_97.5")])

    # Check against the other aggregation method
    for exp_idx, run in enumerate(runs):
        # Get a few fevals to test the series
        test_fevals = np.random.choice(
            agg.xs(exp_idx, level="exp_idx").index.get_level_values("fevals"),
            size=3,
            replace=False,
        )

        # Test against the other aggregation function
        for fevals in test_fevals:
            for idx, actual_row in aggregate_metrics_feval_budget(
                df[df["exp_idx"] == exp_idx],
                max_feval=fevals,
                metric_direction_override={"test": "-"},
            ).iterrows():
                # Get the problem
                problem = idx[0]

                # Get the row which corresponds to this row in the test dataframe
                test_row = agg.loc[(problem, exp_idx, run.name, fevals)]

                # Compare with the "actual" value
                np.testing.assert_allclose(actual_row[("test", "mean")], test_row[("test", "mean")])


def test_comparison_table_to_latex():
    """
    Make sure that `comparison_table_to_latex` runs on reasonable input and produces something that looks like latex.
    """
    # Create some test objects
    runs = generate_moga_experiments(names=["a", "b", "c", "d"])

    # Construct the comparison table
    df = eval_metrics_experiments(runs, metrics=("test", example_metric))
    df = aggregate_metrics_feval_budget(df, metric_direction_override={"test": "-"})
    df = construct_metric_comparison_table(df, problem_params="n")

    # Test the conversion function
    latex_str = comparison_table_to_latex(df)

    # Some basic tests
    assert isinstance(latex_str, str)
    assert "tabular" in latex_str
