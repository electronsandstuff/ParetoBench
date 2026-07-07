import json
import os
import tempfile
import pandas as pd
import pytest
from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
from xopt.generators.ga.nsga2 import NSGA2Generator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
import numpy as np

# Handle Xopt 2.x and 3.x style VOCS
try:
    from xopt.vocs import get_variable_data, get_objective_data, get_constraint_data
    from xopt.vocs import MaximizeObjective, LessThanConstraint

    def _variable_data(vocs, data):
        return get_variable_data(vocs, data)

    def _objective_data(vocs, data):
        return get_objective_data(vocs, data, return_raw=False)

    def _constraint_data(vocs, data):
        return get_constraint_data(vocs, data)

    def _variable_bounds(var):
        return var.domain

    def _is_not_maximize(obj):
        return not isinstance(obj, MaximizeObjective)

    def _constraint_type(c):
        return "LESS_THAN" if isinstance(c, LessThanConstraint) else "GREATER_THAN"

    def _constraint_value(c):
        return c.value

except ImportError:

    def _variable_data(vocs, data):
        return vocs.variable_data(data)

    def _objective_data(vocs, data):
        return vocs.objective_data(data)

    def _constraint_data(vocs, data):
        return vocs.constraint_data(data)

    def _variable_bounds(var):
        return list(var)

    def _is_not_maximize(obj):
        return obj != "MAXIMIZE"

    def _constraint_type(c):
        return c[0]

    def _constraint_value(c):
        return c[1]


from paretobench.ext.xopt import (
    import_cnsga_history,
    import_nsga2_history,
    XoptProblemWrapper,
    import_nsga2_history_dir,
)


def _run_nsga2_to_dir(output_dir, population_size=16, n_generations=3, vocs=tnk_vocs):
    """Run NSGA2Generator into output_dir producing populations.csv and vocs.txt."""
    generator = NSGA2Generator(vocs=vocs, output_dir=output_dir, population_size=population_size)
    xx = Xopt(generator=generator, evaluator=Evaluator(function=evaluate_TNK, max_workers=1), vocs=vocs)
    try:
        for _ in range(n_generations * population_size):
            xx.step()
    finally:
        xx.generator.close_log_file()
    assert os.path.exists(os.path.join(output_dir, "populations.csv"))
    assert os.path.exists(os.path.join(output_dir, "vocs.txt"))


def _split_run_by_generation(src_dir, dst_dirs, boundaries):
    """
    Split src_dir's populations.csv into dst_dirs at the given generation boundaries.

    Emulates a checkpoint restart: xopt_generation numbering stays continuous across the
    resulting directories. boundaries has len(dst_dirs) - 1 entries; generations up to and
    including boundaries[i] go to dst_dirs[i]. vocs.txt is copied to each directory.
    """
    pop_df = pd.read_csv(os.path.join(src_dir, "populations.csv"))
    with open(os.path.join(src_dir, "vocs.txt")) as f:
        vocs_txt = f.read()
    edges = [-np.inf, *boundaries, np.inf]
    for i, dst in enumerate(dst_dirs):
        os.makedirs(dst, exist_ok=True)
        mask = (pop_df["xopt_generation"] > edges[i]) & (pop_df["xopt_generation"] <= edges[i + 1])
        pop_df[mask].to_csv(os.path.join(dst, "populations.csv"), index=False)
        with open(os.path.join(dst, "vocs.txt"), "w") as f:
            f.write(vocs_txt)


def test_import_nsga2_history():
    population_size = 32
    n_generations = 5
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            generator = NSGA2Generator(
                vocs=tnk_vocs,
                output_dir=output_dir,
                population_size=population_size,
            )

            # Run a few optimization steps
            xx = Xopt(
                generator=generator,
                evaluator=Evaluator(function=evaluate_TNK, max_workers=1),
                vocs=tnk_vocs,
            )

            # Run a few generations while copying population information
            fevals = 0
            xopt_xs = []
            xopt_fs = []
            xopt_gs = []
            for _ in range(n_generations):
                # Step the generator
                for _ in range(population_size):
                    xx.step()
                fevals += population_size

                # Get data from the population to test against
                xopt_xs.append(_variable_data(xx.generator.vocs, xx.generator.pop))
                xopt_fs.append(_objective_data(xx.generator.vocs, xx.generator.pop))
                xopt_gs.append(_constraint_data(xx.generator.vocs, xx.generator.pop))

            # Save xopt to yaml file
            xx.dump(os.path.join(output_dir, "xopt.yml"))

            # Verify that the data files are created
            assert os.path.exists(os.path.join(output_dir, "populations.csv"))
            assert os.path.exists(os.path.join(output_dir, "vocs.txt"))

            cases = [
                (os.path.join(output_dir, "vocs.txt"), None),
                (xx.generator.vocs, None),
                (None, os.path.join(output_dir, "xopt.yml")),
            ]
            for vocs, config in cases:
                # Load the data using ParetoBench
                test_hist = import_nsga2_history(os.path.join(output_dir, "populations.csv"), vocs=vocs, config=config)

                # Confirm the populations are the same
                assert len(xopt_xs) == len(test_hist)
                for idx, (rx, rf, rg, tp) in enumerate(zip(xopt_xs, xopt_fs, xopt_gs, test_hist.reports)):
                    # Confirm number of individuals and right shape of contents
                    assert len(rx) == len(tp)
                    assert rf.shape[1] == tp.m
                    assert rx.shape[1] == tp.n
                    assert rg.shape[1] == tp.n_constraints

                    def lexsorted(arr):
                        return arr[np.lexsort(arr.T[::-1])]

                    def df_comp(df1, df2):
                        assert set(df1.columns) == set(df2.columns)
                        cols = sorted(df1.columns)
                        np.testing.assert_allclose(lexsorted(df1[cols].to_numpy()), lexsorted(df2[cols].to_numpy()))

                    # Confirm names are in the correct groups
                    assert list(tp.names_x) == list(tnk_vocs.variable_names)
                    assert list(tp.names_f) == list(tnk_vocs.objective_names)
                    assert list(tp.names_g) == list(tnk_vocs.constraint_names)

                    # Confirm other metadata
                    assert tp.obj_directions == "--"
                    assert tp.constraint_directions == "><"
                    assert all(tp.constraint_targets == [0.0, 0.5])
                    assert tp.fevals == (idx + 1) * population_size

                    # Confirm data is correct
                    df_comp(rx, pd.DataFrame(tp.x, columns=tp.names_x))
                    rf.columns = [x.removeprefix("objective_") for x in rf.columns]
                    df_comp(rf, pd.DataFrame(tp.f, columns=tp.names_f))
                    rg.columns = [x.removeprefix("constraint_") for x in rg.columns]
                    df_comp(rg, pd.DataFrame(tp.g_canonical, columns=tp.names_g))

        # Close log file before exiting context
        finally:
            xx.generator.close_log_file()


def test_import_nsga2_history_dir():
    population_size = 32
    n_generations = 5
    with tempfile.TemporaryDirectory() as output_dir:
        try:
            generator = NSGA2Generator(
                vocs=tnk_vocs,
                output_dir=output_dir,
                population_size=population_size,
            )

            # Run a few optimization steps
            xx = Xopt(
                generator=generator,
                evaluator=Evaluator(function=evaluate_TNK, max_workers=1),
                vocs=tnk_vocs,
            )

            # Run a few generations while copying population information
            fevals = 0
            xopt_xs = []
            xopt_fs = []
            xopt_gs = []
            for _ in range(n_generations):
                # Step the generator
                for _ in range(population_size):
                    xx.step()
                fevals += population_size

                # Get data from the population to test against
                xopt_xs.append(_variable_data(xx.generator.vocs, xx.generator.pop))
                xopt_fs.append(_objective_data(xx.generator.vocs, xx.generator.pop))
                xopt_gs.append(_constraint_data(xx.generator.vocs, xx.generator.pop))

            # Verify that the data files are created
            assert os.path.exists(os.path.join(output_dir, "populations.csv"))
            assert os.path.exists(os.path.join(output_dir, "vocs.txt"))

            # Load the data using ParetoBench
            test_hist = import_nsga2_history_dir(output_dir)

            # Confirm the populations are the same
            assert len(xopt_xs) == len(test_hist)
            for idx, (rx, rf, rg, tp) in enumerate(zip(xopt_xs, xopt_fs, xopt_gs, test_hist.reports)):
                # Confirm number of individuals and right shape of contents
                assert len(rx) == len(tp)
                assert rf.shape[1] == tp.m
                assert rx.shape[1] == tp.n
                assert rg.shape[1] == tp.n_constraints

                def lexsorted(arr):
                    return arr[np.lexsort(arr.T[::-1])]

                def df_comp(df1, df2):
                    assert set(df1.columns) == set(df2.columns)
                    cols = sorted(df1.columns)
                    np.testing.assert_allclose(lexsorted(df1[cols].to_numpy()), lexsorted(df2[cols].to_numpy()))

                # Confirm names are in the correct groups
                assert list(tp.names_x) == list(tnk_vocs.variable_names)
                assert list(tp.names_f) == list(tnk_vocs.objective_names)
                assert list(tp.names_g) == list(tnk_vocs.constraint_names)

                # Confirm other metadata
                assert tp.obj_directions == "--"
                assert tp.constraint_directions == "><"
                assert all(tp.constraint_targets == [0.0, 0.5])
                assert tp.fevals == (idx + 1) * population_size

                # Confirm data is correct
                df_comp(rx, pd.DataFrame(tp.x, columns=tp.names_x))
                rf.columns = [x.removeprefix("objective_") for x in rf.columns]
                df_comp(rf, pd.DataFrame(tp.f, columns=tp.names_f))
                rg.columns = [x.removeprefix("constraint_") for x in rg.columns]
                df_comp(rg, pd.DataFrame(tp.g_canonical, columns=tp.names_g))

        # Close log file before exiting context
        finally:
            xx.generator.close_log_file()


def test_import_cnsga_history():
    population_size = 32
    n_generations = 3
    with tempfile.TemporaryDirectory() as output_dir:
        generator = CNSGAGenerator(
            vocs=tnk_vocs,
            output_path=output_dir,
            population_size=population_size,
        )

        xx = Xopt(
            generator=generator,
            evaluator=Evaluator(function=evaluate_TNK, max_workers=1),
            vocs=tnk_vocs,
        )

        # Save xopt config to yaml
        xx.dump(os.path.join(output_dir, "xopt.yml"))

        # Run generations and capture population data to test against
        xopt_xs = []
        xopt_fs = []
        xopt_gs = []
        for _ in range(n_generations):
            for _ in range(population_size):
                xx.step()
            xopt_xs.append(_variable_data(xx.generator.vocs, xx.generator.population))
            xopt_fs.append(_objective_data(xx.generator.vocs, xx.generator.population))
            xopt_gs.append(_constraint_data(xx.generator.vocs, xx.generator.population))

        # Save vocs to json for file-based loading test
        vocs_path = os.path.join(output_dir, "vocs.json")
        with open(vocs_path, "w") as f:
            f.write(xx.generator.vocs.model_dump_json())

        cases = [
            (vocs_path, None),
            (xx.generator.vocs, None),
            (None, os.path.join(output_dir, "xopt.yml")),
        ]
        for vocs, config in cases:
            test_hist = import_cnsga_history(output_dir, vocs=vocs, config=config)

            assert len(xopt_xs) == len(test_hist)
            for idx, (rx, rf, rg, tp) in enumerate(zip(xopt_xs, xopt_fs, xopt_gs, test_hist.reports)):
                assert len(rx) == len(tp)
                assert rf.shape[1] == tp.m
                assert rx.shape[1] == tp.n
                assert rg.shape[1] == tp.n_constraints

                def lexsorted(arr):
                    return arr[np.lexsort(arr.T[::-1])]

                def df_comp(df1, df2):
                    assert set(df1.columns) == set(df2.columns)
                    cols = sorted(df1.columns)
                    np.testing.assert_allclose(lexsorted(df1[cols].to_numpy()), lexsorted(df2[cols].to_numpy()))

                # Confirm names are in the correct groups
                assert list(tp.names_x) == list(tnk_vocs.variable_names)
                assert list(tp.names_f) == list(tnk_vocs.objective_names)
                assert list(tp.names_g) == list(tnk_vocs.constraint_names)

                # Confirm other metadata
                assert tp.obj_directions == "--"
                assert tp.constraint_directions == "><"
                assert all(tp.constraint_targets == [0.0, 0.5])
                assert tp.fevals == (idx + 1) * population_size

                # Confirm data is correct
                df_comp(rx, pd.DataFrame(tp.x, columns=tp.names_x))
                rf.columns = [x.removeprefix("objective_") for x in rf.columns]
                df_comp(rf, pd.DataFrame(tp.f, columns=tp.names_f))
                rg.columns = [x.removeprefix("constraint_") for x in rg.columns]
                df_comp(rg, pd.DataFrame(tp.g_canonical, columns=tp.names_g))


def test_import_nsga2_history_dir_list():
    population_size = 16
    with tempfile.TemporaryDirectory() as parent:
        # A single continuous run
        full_dir = os.path.join(parent, "full")
        os.makedirs(full_dir)
        _run_nsga2_to_dir(full_dir, population_size=population_size, n_generations=5)
        full = import_nsga2_history_dir(full_dir)

        # The same run split into two checkpoint directories (generations stay continuous)
        dir_a = os.path.join(parent, "run_0")
        dir_b = os.path.join(parent, "run_1")
        _split_run_by_generation(full_dir, [dir_a, dir_b], [3])

        combined = import_nsga2_history_dir([dir_a, dir_b])

        # Reloading the split checkpoints reconstructs the single continuous history exactly
        assert len(combined) == len(full)
        for src, dst in zip(full.reports, combined.reports):
            np.testing.assert_allclose(src.x, dst.x)
            np.testing.assert_allclose(src.f, dst.f)
            np.testing.assert_allclose(src.g, dst.g)
            assert src.fevals == dst.fevals


def test_import_nsga2_history_dir_glob():
    population_size = 16
    with tempfile.TemporaryDirectory() as parent:
        full_dir = os.path.join(parent, "full")
        os.makedirs(full_dir)
        _run_nsga2_to_dir(full_dir, population_size=population_size, n_generations=5)

        dir_a = os.path.join(parent, "run_0")
        dir_b = os.path.join(parent, "run_1")
        _split_run_by_generation(full_dir, [dir_a, dir_b], [3])

        combined_list = import_nsga2_history_dir([dir_a, dir_b])
        combined_glob = import_nsga2_history_dir(os.path.join(parent, "run_*"))

        assert len(combined_glob) == len(combined_list)
        for a, b in zip(combined_list.reports, combined_glob.reports):
            np.testing.assert_allclose(a.x, b.x)
            assert a.fevals == b.fevals


def test_import_nsga2_history_dir_vocs_mismatch():
    population_size = 16
    with tempfile.TemporaryDirectory() as parent:
        dir_a = os.path.join(parent, "run_0")
        dir_b = os.path.join(parent, "run_1")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        _run_nsga2_to_dir(dir_a, population_size=population_size, n_generations=2)

        # Second run directory with a VOCS whose variable names differ
        vocs_dict = json.loads(tnk_vocs.model_dump_json())
        first = next(iter(vocs_dict["variables"]))
        vocs_dict["variables"]["renamed_" + first] = vocs_dict["variables"].pop(first)
        with open(os.path.join(dir_b, "vocs.txt"), "w") as f:
            f.write(json.dumps(vocs_dict))

        with pytest.raises(ValueError):
            import_nsga2_history_dir([dir_a, dir_b])


@pytest.mark.parametrize("prob_name", ["CTP1", "ZDT1", "WFG1", "CF1"])
def test_xopt_problem_wrapper(prob_name):
    wrapper = XoptProblemWrapper.from_line_fmt(prob_name)
    prob = wrapper.prob

    # Verify VOCS structure
    vocs = wrapper.vocs

    assert set(vocs.variable_names) == {f"x{i}" for i in range(prob.n_vars)}
    for i, (lb, ub) in enumerate(zip(prob.var_lower_bounds, prob.var_upper_bounds)):
        np.testing.assert_allclose(_variable_bounds(vocs.variables[f"x{i}"]), [lb, ub])

    assert set(vocs.objective_names) == {f"f{i}" for i in range(prob.n_objs)}
    assert all(_is_not_maximize(vocs.objectives[name]) for name in vocs.objective_names)

    assert set(vocs.constraint_names) == {f"g{i}" for i in range(prob.n_constraints)}
    assert all(_constraint_type(vocs.constraints[name]) == "LESS_THAN" for name in vocs.constraint_names)
    assert all(_constraint_value(vocs.constraints[name]) == 0 for name in vocs.constraint_names)

    rng = np.random.default_rng(42)
    lbs, ubs = prob.var_lower_bounds, prob.var_upper_bounds
    expected_keys = {f"f{i}" for i in range(prob.n_objs)} | {f"g{i}" for i in range(prob.n_constraints)}

    # Run with scalar values
    x_scalar = rng.uniform(lbs, ubs)
    result = wrapper({f"x{i}": float(x_scalar[i]) for i in range(prob.n_vars)})
    assert set(result.keys()) == expected_keys
    pop = prob(x_scalar)
    for i in range(prob.n_objs):
        np.testing.assert_allclose(result[f"f{i}"], pop.f[:, i])
    for i in range(prob.n_constraints):
        np.testing.assert_allclose(result[f"g{i}"], pop.g[:, i])

    # Test vectorized wrapper
    x_arr = rng.uniform(lbs, ubs, size=(8, prob.n_vars))
    result = wrapper({f"x{i}": x_arr[:, i] for i in range(prob.n_vars)})
    pop = prob(x_arr)
    for i in range(prob.n_objs):
        np.testing.assert_allclose(result[f"f{i}"], pop.f[:, i])
    for i in range(prob.n_constraints):
        np.testing.assert_allclose(result[f"g{i}"], pop.g[:, i])
