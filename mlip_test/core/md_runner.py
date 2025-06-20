from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.optimize import BFGS, LBFGS, GPMin, FIRE
import ase.units as units
import numpy as np
import json
from functools import partial
from time import time
from datetime import datetime

class MDRunner:
    """class to run MD simulations. """

    def __init__(self, atoms, calculator, monitor=None):
        """initialize"""
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        self.monitor = monitor

    def run_nvt(self, temperature=300, timestep=1.0, duration_ps=1.0, loginterval=100, trajectory_file=None):

        print(f"Running NVT: {duration_ps} ps at {temperature} K")
        start_time = datetime.now()
        print(f"Simulation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        dyn = NVTBerendsen(self.atoms,
                       timestep=timestep*units.fs,
                       temperature_K=temperature,
                       taut=100*units.fs)
        
        if self.monitor is not None:
            dyn.attach(partial(self.monitor, self.atoms), interval=1)

        if trajectory_file:
            from ase.io import Trajectory
            traj = Trajectory(trajectory_file, 'w', self.atoms)
            dyn.attach(traj.write, interval=loginterval)

        self._current_step_nvt = 0
        total_steps = int(duration_ps*1000/timestep)
        dyn.attach(partial(self._log_progress, 
                        phase='NVT', 
                        total_steps=total_steps, 
                        step_counter_name='_current_step_nvt'), 
                interval=loginterval)

        try:
            dyn.run(total_steps)
            end_time = datetime.now()
            print("NVT completed successfully")
            print(f"Simulation ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = end_time - start_time
            print(f"Total duration: {duration}")
        except Exception as e:
            import traceback
            print(f"NVT simulation failed: {e}")
            traceback.print_exc()
            if self.monitor:
                print("Monitor summary:")
                print(self.monitor.get_summary())
            raise

        return self.atoms.copy()
    
    def run_npt(self, temperature=300, pressure=0.0, timestep=1.0, duration_ps=2.0, loginterval=100, trajectory_file=None):

        print(f"Running NPT: {duration_ps} ps at {temperature} K")

        dyn = NPT(self.atoms,
                timestep=timestep*units.fs,
                temperature_K=temperature,
                externalstress=pressure, #this is atmospheric pressure
                ttime=100*units.fs,
                pfactor=75*units.fs**2)
        
        if self.monitor is not None:
            from functools import partial
            dyn.attach(partial(self.monitor, self.atoms), interval=1)

        if trajectory_file:
            from ase.io import Trajectory
            traj = Trajectory(trajectory_file, 'w', self.atoms)
            dyn.attach(traj.write, interval=loginterval)
            
        self._current_step_npt = 0
        total_steps = int(duration_ps*1000/timestep)
        dyn.attach(partial(self._log_progress,
                        phase='NPT', 
                        total_steps=total_steps,
                        step_counter_name='_current_step_npt'), 
                interval=loginterval)
        
        try:
            dyn.run(total_steps)
            print("NPT completed successfully")
        except Exception as e:
            print(f"NPT simulation failed: {e}")
            if self.monitor:
                print("Monitor summary:")
                print(self.monitor.get_summary())
            raise

        return self.atoms.copy()
    
    def run_nve(self, timestep=1.0, duration_ps=2.0, loginterval=100, trajectory_file=None):

        print(f"Running NVE: {duration_ps} ps")

        initial_energy = self.atoms.get_total_energy()
        dyn = VelocityVerlet(self.atoms,
                             timestep=timestep*units.fs)
        if self.monitor is not None:
            from functools import partial
            dyn.attach(partial(self.monitor, self.atoms), interval=1)
            
        if trajectory_file:
            from ase.io import Trajectory
            traj = Trajectory(trajectory_file, 'w', self.atoms)
            dyn.attach(traj.write, interval=loginterval)
            
        self._current_step_nve = 0
        total_steps = int(duration_ps*1000/timestep)
        dyn.attach(partial(self._log_progress,
                        phase='NVE', 
                        total_steps=total_steps,
                        step_counter_name='_current_step_nve'), 
                interval=loginterval)
        
        try:
            dyn.run(total_steps)
            final_energy = self.atoms.get_total_energy()
            energy_drift = abs(final_energy - initial_energy)
            print(f"NVE completed successfully")
            print(f"Energy drift: {energy_drift:.6f} eV ({energy_drift/len(self.atoms)*1000:.3f} meV/atom)")
        except Exception as e:
            print(f"NVE simulation failed: {e}")
            if self.monitor:
                print("Monitor summary:")
                print(self.monitor.get_summary())
            raise
        
        return self.atoms.copy(), initial_energy, final_energy
    
    def _log_progress(self, phase, total_steps, step_counter_name=None):
        """Internal method to log simulation progress"""
        if step_counter_name is None:
            step_counter_name = f'_current_step_{phase.lower()}'

        current_step = getattr(self, step_counter_name, 0)
        progress_percent = (current_step / total_steps) * 100 if total_steps > 0 else 0

        if hasattr(self.atoms, 'get_total_energy'):
            try:
                energy = self.atoms.get_total_energy()
                energy_str = f"E={energy:.3f} eV"
            except Exception as e:
                energy_str = f"E=Error({str(e)[:20]})"

            try:
                velocities = self.atoms.get_velocities()
                if velocities is not None and not np.any(np.isnan(velocities)):
                    # Calculate kinetic energy
                    masses = self.atoms.get_masses()
                    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
                    # Convert to temperature using equipartition theorem
                    # 3/2 * N * kB * T = KE, so T = 2*KE / (3*N*kB)
                    import ase.units as units
                    temp = 2 * ke / (3 * len(self.atoms) * units.kB)
                    temp_str = f"T={temp:.1f} K"
                else:
                    temp_str = "T=No velocities"
            except Exception as e:
                temp_str = f"T=Error({str(e)[:20]})"

            print(f"{phase} Step {current_step:4d}/{total_steps} ({progress_percent:5.1f}%): {energy_str}, {temp_str}")
            setattr(self, step_counter_name, current_step + 1)

    def run_sequential_protocol(self, nvt_params=None, npt_params=None, nve_params=None, base_filename="md_run"):
        """run NVT/NPT/NVE protocol for testing"""

        if nvt_params is None:
            nvt_params = {'duration_ps': 1.0, 'temperature': 300}
        if npt_params is None:
            npt_params = {'duration_ps': 2.0, 'temperature': 300, 'pressure': 0.0}
        if nve_params is None:
            nve_params = {'duration_ps': 1.0}

        results = {}

        nvt_params['trajectory_file'] = f"{base_filename}_nvt.traj"
        npt_params['trajectory_file'] = f"{base_filename}_npt.traj"  
        nve_params['trajectory_file'] = f"{base_filename}_nve.traj"

        print("="*50)
        print("Starting Sequential MD Protocol")
        print("="*50)

        results['nvt_final'] = self.run_nvt(**nvt_params)
        results['npt_final'] = self.run_npt(**npt_params)
        nve_result = self.run_nve(**nve_params)
        results['nve_final'] = nve_result[0]
        results['nve_initial_energy'] = nve_result[1] 
        results['nve_final_energy'] = nve_result[2]
        results['energy_drift'] = abs(nve_result[2] - nve_result[1])

        print("="*50)
        print("Protocol Summary:")
        print(f"Energy drift in NVE: {results['energy_drift']:.6f} eV")
        if self.monitor:
            print("Monitoring Summary:")
            summary = self.monitor.get_summary()
            for key, value in summary.items():
                print(f"  {key}: {value}")
        print("="*50)
        #save to json?
        return results
    
class MinimizationRunner:
    def __init__(self, atoms, calculator, output_dir="minimization_output", monitor=None):
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        self.output_dir = output_dir
        self.monitor = monitor
        self.results = {}

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")

    def minimize(self, optimizer='BFGS', fmax=0.01, steps=500, traj_file=None, logfile=None, **optimizer_kwargs):

        import os
        if trajectory_file is None:
            trajectory_file = f"{optimizer.lower()}_optimization.traj"
        if logfile is None:
            logfile = f"{optimizer.lower()}_optimization.log"
        
        # Make file paths relative to output_dir
        trajectory_path = os.path.join(self.output_dir, trajectory_file)
        logfile_path = os.path.join(self.output_dir, logfile)
        
        print(f"Trajectory file: {trajectory_path}")
        print(f"Log file: {logfile_path}")
        self.save_initial_structure()

        initial_energy = self.atoms.get_potential_energy()
        initial_forces = self.atoms.get_forces()
        initial_max_force = np.max(np.linalg.norm(initial_forces, axis=1))

        print(f"Initial energy: {initial_energy:.6f} eV")
        print(f"Initial max force: {initial_max_force:.6f} eV/Å")

        if logfile:
            optimizer_kwargs['logfile'] = logfile

        opt = BFGS(self.atoms, trajectory=traj_file)
        if self.monitor is not None:
            opt.attach(partial(self.monitor, self.atoms))
        opt.attach(partial(self._log_optimization_progress, initial_energy, fmax))

        start_time = time()

        try:
            # Run optimization
            converged = opt.run(fmax=fmax, steps=steps)
            
            # Get final state
            final_energy = self.atoms.get_potential_energy()
            final_forces = self.atoms.get_forces()
            final_max_force = np.max(np.linalg.norm(final_forces, axis=1))
            final_rms_force = np.sqrt(np.mean(np.sum(final_forces**2, axis=1)))
            
            optimization_time = time() - start_time
            actual_steps = opt.get_number_of_steps()
            
            # Store results
            self.results = {
                'converged': converged,
                'initial_energy': initial_energy,
                'final_energy': final_energy,
                'energy_change': final_energy - initial_energy,
                'initial_max_force': initial_max_force,
                'final_max_force': final_max_force,
                'final_rms_force': final_rms_force,
                'force_reduction': initial_max_force - final_max_force,
                'steps_taken': actual_steps,
                'max_steps': steps,
                'fmax_criterion': fmax,
                'optimization_time': optimization_time,
                'optimizer': optimizer,
                'success': converged and final_max_force <= fmax
            }
            
            from ase.io import write
            final_structure_path = os.path.join(self.output_dir, 'final_structure.xyz')
            
            # Save final structure
            final_atoms = self.atoms.copy()
            final_atoms.calc = None
            write(final_structure_path, final_atoms)
            
            print(f"Final structure saved to: {final_structure_path}")
            
            # Auto-save results
            self.save_results()
            
            # Print summary
            print(f"\n{'='*50}")
            print("OPTIMIZATION SUMMARY")
            print(f"{'='*50}")
            print(f"Converged: {'YES' if converged else 'NO'}")
            print(f"Steps taken: {actual_steps}/{steps}")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"Energy change: {self.results['energy_change']:.6f} eV")
            print(f"Final energy: {final_energy:.6f} eV")
            print(f"Final max force: {final_max_force:.6f} eV/Å (criterion: {fmax} eV/Å)")
            print(f"Final RMS force: {final_rms_force:.6f} eV/Å")
            print(f"Force reduction: {self.results['force_reduction']:.6f} eV/Å")
            
            if converged:
                print("✓ Optimization completed successfully")
            else:
                print("⚠ Optimization did not converge within step limit")
                
        except Exception as e:
            print(f"✗ Optimization failed: {e}")
            self.results = {
                'converged': False,
                'success': False,
                'error': str(e),
                'steps_taken': opt.get_number_of_steps() if hasattr(opt, 'get_number_of_steps') else 0,
                'optimization_time': time.time() - start_time
            }
            if self.monitor:
                print("Monitor summary:")
                print(self.monitor.get_summary())
            raise
        
        return self.results
    
    def _log_optimization_progress(self, initial_energy, fmax):
        """Log optimization progress"""
        try:
            current_energy = self.atoms.get_potential_energy()
            current_forces = self.atoms.get_forces()
            max_force = np.max(np.linalg.norm(current_forces, axis=1))
            
            energy_change = current_energy - initial_energy
            converged_str = "✓" if max_force <= fmax else "→"
            
            print(f"{converged_str} E={current_energy:.6f} eV (ΔE={energy_change:+.6f}), "
                  f"Fmax={max_force:.6f} eV/Å")
                  
        except Exception as e:
            print(f"Progress logging failed: {e}")
            

    def get_results(self):
        """Return optimization results"""
        return self.results.copy() if self.results else {}
    
    def save_results(self, filename):
        """Save optimization results to JSON file"""
        if not self.results:
            print("No results to save. Run minimize() first.")
            return
        
        import os
        
        if filename is None:
            filename = "minimization_results.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        
        # Convert numpy types to Python types for JSON serialization
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, (np.integer, np.floating)):
                results_serializable[key] = value.item()
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to: {filepath}")

    def save_initial_structure(self, atoms=None, filename=None):
        """Save initial structure before optimization starts"""
        import os
        from ase.io import write
        
        if filename is None:
            filename = "initial_structure.xyz"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if atoms is None:
            atoms = self.atoms.copy()
        
        # Remove calculator to avoid issues during writing
        atoms_to_save = atoms.copy()
        atoms_to_save.calc = None
        
        write(filepath, atoms_to_save)
        print(f"Initial structure saved to: {filepath}")
        return filepath





