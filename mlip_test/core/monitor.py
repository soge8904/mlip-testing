from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
import ase.units as units
import time
import numpy as np
import json

class MDMonitor:
    "real-time monitoring"
    def __init__(self, atoms, metrics=None, check_interval=10):
        if metrics is None:
            metrics = ['energy', 'forces', 'temperature']

        self.metrics = metrics
        self.check_interval = check_interval
        self.step_count = 0
        self.data = {metric: [] for metric in metrics} #make dict where each  metric gets a list
        self.data['steps'] = []
        self.data['time'] = []

        self.initial_positions = atoms.positions.copy()
        self.natoms = len(atoms)

        #safety thresholds
        self.max_force = 50.0 #eV/A
        self.max_displacement = 5.0
        self.last_positions = atoms.positions.copy()

    def __call__(self, atoms):
        self.step_count +=1
        if self.step_count % self.check_interval != 0:
            return
        
        self._safety_checks(atoms)

        self.data['steps'].append(self.step_count)
        self.data['time'].append(time.time())

        if 'energy' in self.metrics:
            try:
                energy = atoms.get_total_energy()
                self.data['energy'].append(energy)
            except Exception as e:
                print(f"Warning: Could not get energy at step {self.step_count}: {e}")
                self.data['energy'].append(np.nan)

        if 'forces' in self.metrics:
            try:
                forces = atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))
                rms_force = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
                self.data['forces'].append({'max': max_force, 'rms': rms_force})
            except Exception as e:
                print(f"Warning: Could not get forces at step {self.step_count}: {e}")
                self.data['forces'].append({'max': np.nan, 'rms': np.nan})

        if 'temperature' in self.metrics:
            try:
                velocities = atoms.get_velocities()
                if velocities is not None:
                    ke = 0.5 * np.sum(atoms.get_masses()[:, np.newaxis] * velocities**2)
                    temp = 2 * ke / (3 * len(atoms) * units.kB)
                    self.data['temperature'].append(temp)
                else:
                    self.data['temperature'].append(np.nan)
            except Exception as e:
                print(f"Warning: Could not calculate temperature at step {self.step_count}: {e}")
                self.data['temperature'].append(np.nan)

        self.last_positions = atoms.positions.copy()

    def _safety_checks(self, atoms):
        """check for problems during the simulation"""
        try:
            displacements = np.linalg.norm(atoms.positions - self.last_positions, axis=1)
            max_disp = np.max(displacements)

            if max_disp > self.max_displacement:
                print(f"WARNING: Large atomic displacement detected at step {self.step_count}: {max_disp:.3f} Å")

            if np.any(np.isnan(atoms.positions)):
                raise RuntimeError(f"NaN positions detected at step {self.step_count}")
            
        except Exception as e:
            print(f"Safety check failed at step {self.step_count}: {e}")

    def get_summary(self):
        summary = {}

        if self.data['energy']:
            energies = [e for e in self.data['energy'] if not np.isnan(e)]
            if energies:
                summary['energy'] = {
                    'initial': energies[0],
                    'final': energies[-1],
                    'drift': energies[-1] - energies[0],
                    'std': np.std(energies)
                }
        
        if self.data['forces']:
            max_forces = [f['max'] for f in self.data['forces'] if not np.isnan(f['max'])]
            if max_forces:
                summary['forces'] = {
                    'max_encountered': np.max(max_forces),
                    'avg_max': np.mean(max_forces),
                    'final_max': max_forces[-1]
                }

        summary['total_steps'] = self.step_count
        summary['data_points'] = len(self.data['steps'])

        return summary

