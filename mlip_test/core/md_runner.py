from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
import ase.units as units
import numpy as np
import json
from functools import partial

class MDRunner:
    """class to run MD simulations. """

    def __init__(self, atoms, calculator, monitor=None):
        """initialize"""
        self.atoms = atoms.copy()
        self.atoms.calc = calculator
        self.monitor = monitor

    def run_nvt(self, temperature=300, timestep=1.0, duration_ps=1.0, loginterval=100, trajectory_file=None):

        print(f"Running NVT: {duration_ps} ps at {temperature} K")

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
            print("NVT completed successfully")
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