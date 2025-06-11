import os
import numpy as np
import json
from typing import Dict, Tuple, Optional
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ..core.md_runner import MDRunner
from ..core.monitor import MDMonitor

class MACEStabilityTest:

    def __init__(self, atoms, calculator, base_output_dir="mace_stablity_test"):

        self.atoms = atoms.copy()
        self.calculator = calculator
        self.base_output_dir = base_output_dir

        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(f"{self.base_output_dir}/track_a", exist_ok=True)
        os.makedirs(f"{self.base_output_dir}/track_b", exist_ok=True)

        write(f"{self.base_output_dir}/initial_structure.xyz", self.atoms)
        self.results = {}

    def run_track_a(self, nvt_params=None, npt_params=None, nve_params=None, monitor_interval=50) -> Tuple[Dict, MDMonitor]:
        """direct MLIP testing from Packmol structure (*no equilibration)"""

        if nvt_params is None:
            nvt_params = {'duration_ps': 1.0, 'temperature': 300, 'timestep': 1.0}
        if npt_params is None:
            npt_params = {'duration_ps': 2.0, 'temperature': 300, 'pressure': 0.0, 'timestep': 1.0}
        if nve_params is None:
            nve_params = {'duration_ps': 1.0, 'timestep': 1.0}

        atoms_a = self.atoms.copy()
        atoms_a.calc = self.calculator

        MaxwellBoltzmannDistribution(atoms_a, temperature_K=nvt_params['temperature']) #initialize velocities

        monitor = MDMonitor(atoms_a, metrics=['energy', 'forces', 'temperature'], check_interval=monitor_interval)

        runner = MDRunner(atoms_a, self.calculator, monitor=monitor)

        try:
            results = runner.run_sequential_protocol(
                nvt_params=nvt_params,
                npt_params=npt_params,
                nve_params=nve_params,
                base_filename=f"{self.base_output_dir}/track_a/track_a"
            )

            write(f"{self.base_output_dir}/track_a/final_structure.xyz", results['nve_final'])

            self.results['track_a'] = {
                'energy_drift': results['energy_drift'],
                'nve_initial_energy': results['nve_initial_energy'],
                'nve_final_energy': results['nve_final_energy'],
                'success': True,
                'monitor_summary': monitor.get_summary()
            }

            print(f"\n✓ Track A completed successfully")
            print(f"  Energy drift: {results['energy_drift']:.6f} eV ({results['energy_drift']/len(atoms_a)*1000:.3f} meV/atom)")

        except Exception as e:
            print(f"\n✗ Track A failed: {e}")
            self.results['track_a'] = {
                'success': False,
                'error': str(e),
                'monitor_summary': monitor.get_summary() if monitor else {}
            }
            
        return self.results.get('track_a', {}), monitor