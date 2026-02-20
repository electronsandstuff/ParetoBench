import os
import tempfile
import pandas as pd
from xopt.base import Xopt
from xopt.evaluator import Evaluator
from xopt.generators.ga.nsga2 import NSGA2Generator
from xopt.resources.test_functions.tnk import evaluate_TNK, tnk_vocs
import numpy as np

from paretobench.ext.xopt import import_nsga2_history


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

            # Verify that the data files are created
            assert os.path.exists(os.path.join(output_dir, "populations.csv"))
            assert os.path.exists(os.path.join(output_dir, "vocs.txt"))

            # Load the data using ParetoBench
            test_hist = import_nsga2_history(
                os.path.join(output_dir, "populations.csv"), os.path.join(output_dir, "vocs.txt")
            )

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
