import os
import tempfile
import pandas as pd
from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.cnsga import CNSGAGenerator
from xopt.generators.ga.nsga2 import NSGA2Generator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
import numpy as np

from paretobench.ext.xopt import import_cnsga_history, import_nsga2_history


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
                xopt_xs.append(xx.generator.vocs.variable_data(xx.generator.pop))
                xopt_fs.append(xx.generator.vocs.objective_data(xx.generator.pop))
                xopt_gs.append(xx.generator.vocs.constraint_data(xx.generator.pop))

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
                for rx, rf, rg, tp in zip(xopt_xs, xopt_fs, xopt_gs, test_hist.reports):
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
            xopt_xs.append(xx.generator.vocs.variable_data(xx.generator.population))
            xopt_fs.append(xx.generator.vocs.objective_data(xx.generator.population))
            xopt_gs.append(xx.generator.vocs.constraint_data(xx.generator.population))

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
            for rx, rf, rg, tp in zip(xopt_xs, xopt_fs, xopt_gs, test_hist.reports):
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

                df_comp(rx, pd.DataFrame(tp.x, columns=tp.names_x))
                rf.columns = [x.removeprefix("objective_") for x in rf.columns]
                df_comp(rf, pd.DataFrame(tp.f, columns=tp.names_f))
                rg.columns = [x.removeprefix("constraint_") for x in rg.columns]
                df_comp(rg, pd.DataFrame(tp.g_canonical, columns=tp.names_g))
