from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Union
import h5py
import re


@dataclass
class Population:
    x: np.ndarray
    f: np.ndarray
    g: np.ndarray
    feval: int
    
    def _to_h5py_group(self, g: h5py.Group):
        # Write datasets
        g['x'] = self.x
        g['f'] = self.f
        g['g'] = self.g
        
        # Add metadata
        g.attrs['feval'] = self.feval

    @classmethod
    def _from_h5py_group(cls, g: h5py.Group):
        return cls(
            x=g['x'][()],
            f=g['f'][()],
            g=g['g'][()],
            feval=g.attrs['feval']
        )


@dataclass
class History:
    reports: List[Population]
    problem: str
    metadata: Dict[str, Union[str, int, float, bool]]

    def _to_h5py_group(self, g: h5py.Group):
        # Save the metadata
        g.attrs['problem'] = self.problem
        g_md = g.create_group("metadata")
        for k, v in self.metadata.items():
            g_md.attrs[k] = v
        
        # Save each report into its own group
        for idx, report in enumerate(self.report):
            report._to_h5py_group(g.create_group("report_{:d}".format(idx)))

    @classmethod
    def _from_h5py_group(cls, g: h5py.Group):
        # Load each of the runs keeping track of the order of the indices
        idx_reports = []
        for idx_str, report_grp in g.items():
            m = re.match(r'report_(\d+)', idx_str)
            if m:
                idx_reports.append((int(m.group(1)), Population._from_h5py_group(report_grp)))
        reports = [x[1] for x in sorted(idx_reports, key=lambda x: x[0])]

        # Return as a history object
        return cls(
            problem=g.attrs['problem'],
            reports=reports,
            metadata={k: v for k, v in g['metadata'].attrs.items()},
        )

    
@dataclass
class Experiment:
    runs: List[History]
    identifier: str
    
    def empty(self):
        self.runs = []
        self.identifier = ''
        
    def save(self, fname):
        with h5py.File(fname, mode='w') as f:
            # Save metadata as attributes
            f.attrs['identifier'] = self.identifier
            
            # Save each run into its own group
            for idx, run in enumerate(self.runs):
                run._to_h5py_group(f.create_group("run_{:d}".format(idx)))

    @classmethod
    def load(cls, fname):        
        # Load the data
        with h5py.File(fname, mode='r') as f:
            # Load each of the runs keeping track of the order of the indices
            idx_runs = []
            for idx_str, run_grp in f.items():
                m = re.match(r'report_(\d+)', idx_str)
                if m:
                    idx_runs.append((int(m.group(1)), History._from_h5py_group(run_grp)))
            runs = [x[1] for x in sorted(idx_runs, key=lambda x: x[0])]
            
            # Return as an experiment object
            return cls(
                identifier=f.attrs['identifier'],
                runs=runs,
            )
