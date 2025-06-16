import os
import numpy as np
import json
from typing import Dict, Tuple, Optional, List
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import matplotlib.pyplot as plt
from scipy import stats

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
    
class MACEStabilityAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.analysis_results = {}

    def analyze_monitor_data(self, monitor_data: Dict=None, json_file="monitor_data.json") -> Dict:
        if monitor_data is not None:
            self.monitor_data = monitor_data
        elif json_file is not None:
            import json
            with open(os.path.join(self.results_dir, json_file), "r") as json_file:
                self.monitor_data = json.load(json_file)
        
        analysis = {}

        if 'energy' in monitor_data:
            energies = [e for e in monitor_data['energy'] if not np.isnan(e)]
            if len(energies) > 1:
                analysis['energy'] = self._analyze_energy_series(energies)
        
        # Force analysis  
        if 'forces' in monitor_data:
            max_forces = [f['max'] for f in monitor_data['forces'] if not np.isnan(f['max'])]
            rms_forces = [f['rms'] for f in monitor_data['forces'] if not np.isnan(f['rms'])]
            if max_forces:
                analysis['forces'] = self._analyze_force_series(max_forces, rms_forces)
        
        # Temperature analysis
        if 'temperature' in monitor_data:
            temps = [t for t in monitor_data['temperature'] if not np.isnan(t)]
            if temps:
                analysis['temperature'] = self._analyze_temperature_series(temps)
        
        # Overall stability assessment
        analysis['stability_score'] = self._calculate_stability_score(analysis)

    def _analyze_energy_series(self, energies: List[float]) -> Dict:
        """Analyze energy time series for conservation and drift"""
        energies = np.array(energies)
        
        # Basic statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Energy drift (linear trend)
        x = np.arange(len(energies))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, energies)
        
        # Energy fluctuations (after detrending)
        detrended = energies - (slope * x + intercept)
        fluctuation_rms = np.sqrt(np.mean(detrended**2))
        
        # Relative measures
        rel_drift = abs(slope * len(energies)) / abs(mean_energy) if mean_energy != 0 else np.inf
        rel_fluctuation = fluctuation_rms / abs(mean_energy) if mean_energy != 0 else np.inf
        
        return {
            'mean': mean_energy,
            'std': std_energy,
            'drift_rate': slope,  # eV/step
            'total_drift': abs(energies[-1] - energies[0]),
            'relative_drift': rel_drift,
            'fluctuation_rms': fluctuation_rms,
            'relative_fluctuation': rel_fluctuation,
            'r_squared': r_value**2,
            'is_conserved': rel_drift < 1e-4 and rel_fluctuation < 1e-3  # Conservative thresholds
        }
    
    def _analyze_force_series(self, max_forces: List[float], rms_forces: List[float]) -> Dict:
        """Analyze force time series for consistency"""
        max_forces = np.array(max_forces)
        rms_forces = np.array(rms_forces)
        
        # Detect force spikes (outliers)
        max_q75, max_q25 = np.percentile(max_forces, [75, 25])
        max_iqr = max_q75 - max_q25
        max_outlier_threshold = max_q75 + 1.5 * max_iqr
        
        force_spikes = np.sum(max_forces > max_outlier_threshold)
        
        # Force stability
        max_force_cv = np.std(max_forces) / np.mean(max_forces) if np.mean(max_forces) > 0 else np.inf
        rms_force_cv = np.std(rms_forces) / np.mean(rms_forces) if np.mean(rms_forces) > 0 else np.inf
        
        return {
            'max_force_mean': np.mean(max_forces),
            'max_force_std': np.std(max_forces),
            'max_force_max': np.max(max_forces),
            'rms_force_mean': np.mean(rms_forces),
            'rms_force_std': np.std(rms_forces),
            'force_spikes': int(force_spikes),
            'max_force_cv': max_force_cv,
            'rms_force_cv': rms_force_cv,
            'forces_stable': max_force_cv < 0.5 and force_spikes < len(max_forces) * 0.05
        }
    
    def _analyze_temperature_series(self, temperatures: List[float]) -> Dict:
        """Analyze temperature time series"""
        temps = np.array(temperatures)
        
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        temp_cv = std_temp / mean_temp if mean_temp > 0 else np.inf
        
        # Temperature drift
        x = np.arange(len(temps))
        slope, _, _, _, _ = stats.linregress(x, temps)
        
        return {
            'mean': mean_temp,
            'std': std_temp,
            'coefficient_of_variation': temp_cv,
            'drift_rate': slope,
            'min': np.min(temps),
            'max': np.max(temps),
            'temp_stable': temp_cv < 0.1  # Temperature should be fairly stable
        }
    
    def _calculate_stability_score(self, analysis: Dict) -> float:
        """Calculate overall stability score (0-100)"""
        score = 100.0
        
        # Energy penalties
        if 'energy' in analysis:
            energy = analysis['energy']
            if not energy.get('is_conserved', False):
                score -= 30
            if energy.get('relative_drift', 0) > 1e-3:
                score -= 20
            if energy.get('relative_fluctuation', 0) > 1e-2:
                score -= 15
        
        # Force penalties
        if 'forces' in analysis:
            forces = analysis['forces']
            if not forces.get('forces_stable', False):
                score -= 20
            if forces.get('force_spikes', 0) > 0:
                score -= 10
        
        # Temperature penalties
        if 'temperature' in analysis:
            temp = analysis['temperature']
            if not temp.get('temp_stable', False):
                score -= 15
        
        return max(0.0, score)
    
    def generate_plots(self, monitor_data: Dict, save_dir: str = None):
        """Generate diagnostic plots"""
        if save_dir is None:
            save_dir = self.results_dir
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'MACE Stability Analysis', fontsize=14)
        
        # Energy plot
        if 'energy' in monitor_data and 'steps' in monitor_data:
            steps = monitor_data['steps']
            energies = monitor_data['energy']
            
            # Filter out NaN values
            valid_idx = [i for i, e in enumerate(energies) if not np.isnan(e)]
            if valid_idx:
                steps_clean = [steps[i] for i in valid_idx]
                energies_clean = [energies[i] for i in valid_idx]
                
                axes[0,0].plot(steps_clean, energies_clean, 'b-', linewidth=1)
                axes[0,0].set_xlabel('MD Steps')
                axes[0,0].set_ylabel('Total Energy (eV)')
                axes[0,0].set_title('Energy Conservation')
                axes[0,0].grid(True, alpha=0.3)
        
        # Forces plot
        if 'forces' in monitor_data and 'steps' in monitor_data:
            steps = monitor_data['steps']
            forces = monitor_data['forces']
            
            max_forces = []
            rms_forces = []
            steps_clean = []
            
            for i, f in enumerate(forces):
                if not np.isnan(f['max']):
                    max_forces.append(f['max'])
                    rms_forces.append(f['rms'])
                    steps_clean.append(steps[i])
            
            if max_forces:
                axes[0,1].plot(steps_clean, max_forces, 'r-', label='Max Force', linewidth=1)
                axes[0,1].plot(steps_clean, rms_forces, 'g-', label='RMS Force', linewidth=1)
                axes[0,1].set_xlabel('MD Steps')
                axes[0,1].set_ylabel('Force (eV/Å)')
                axes[0,1].set_title('Force Consistency')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
        
        # Temperature plot
        if 'temperature' in monitor_data and 'steps' in monitor_data:
            steps = monitor_data['steps']
            temps = monitor_data['temperature']
            
            valid_idx = [i for i, t in enumerate(temps) if not np.isnan(t)]
            if valid_idx:
                steps_clean = [steps[i] for i in valid_idx]
                temps_clean = [temps[i] for i in valid_idx]
                
                axes[1,0].plot(steps_clean, temps_clean, 'm-', linewidth=1)
                axes[1,0].set_xlabel('MD Steps')
                axes[1,0].set_ylabel('Temperature (K)')
                axes[1,0].set_title('Temperature Evolution')
                axes[1,0].grid(True, alpha=0.3)
        
        # Energy histogram
        if 'energy' in monitor_data:
            energies = [e for e in monitor_data['energy'] if not np.isnan(e)]
            if energies:
                axes[1,1].hist(energies, bins=30, alpha=0.7, color='blue', density=True)
                axes[1,1].set_xlabel('Total Energy (eV)')
                axes[1,1].set_ylabel('Density')
                axes[1,1].set_title('Energy Distribution')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/stability_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, analysis_results: Dict, save_path: str = None) -> str:
        """Generate a comprehensive stability report"""
        
        report = []
        report.append("="*60)
        report.append("MACE POTENTIAL STABILITY TEST REPORT")
        report.append("="*60)
        report.append("")
        
        for track_name, analysis in analysis_results.items():
            report.append(f"TRACK: {track_name.upper()}")
            report.append("-" * 40)
            
            # Overall stability
            stability_score = analysis.get('stability_score', 0)
            status = "PASS" if stability_score >= 70 else "FAIL"
            report.append(f"Overall Stability Score: {stability_score:.1f}/100 [{status}]")
            report.append("")
            
            # Energy analysis
            if 'energy' in analysis:
                energy = analysis['energy']
                report.append("Energy Conservation:")
                report.append(f"  - Conserved: {'YES' if energy.get('is_conserved', False) else 'NO'}")
                report.append(f"  - Total drift: {energy.get('total_drift', 0):.6f} eV")
                report.append(f"  - Relative drift: {energy.get('relative_drift', 0):.2e}")
                report.append(f"  - RMS fluctuation: {energy.get('fluctuation_rms', 0):.6f} eV")
                report.append("")
            
            # Force analysis
            if 'forces' in analysis:
                forces = analysis['forces']
                report.append("Force Consistency:")
                report.append(f"  - Stable: {'YES' if forces.get('forces_stable', False) else 'NO'}")
                report.append(f"  - Average max force: {forces.get('max_force_mean', 0):.3f} eV/Å")
                report.append(f"  - Force spikes detected: {forces.get('force_spikes', 0)}")
                report.append(f"  - Max force CV: {forces.get('max_force_cv', 0):.3f}")
                report.append("")
            
            # Temperature analysis
            if 'temperature' in analysis:
                temp = analysis['temperature']
                report.append("Temperature Stability:")
                report.append(f"  - Stable: {'YES' if temp.get('temp_stable', False) else 'NO'}")
                report.append(f"  - Mean temperature: {temp.get('mean', 0):.1f} K")
                report.append(f"  - Temperature CV: {temp.get('coefficient_of_variation', 0):.4f}")
                report.append("")
            
            report.append("=" * 60)
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        all_scores = [analysis.get('stability_score', 0) for analysis in analysis_results.values()]
        avg_score = np.mean(all_scores) if all_scores else 0
        
        if avg_score >= 80:
            report.append("✓ MACE potential shows excellent stability")
            report.append("✓ Safe to use for production MD simulations")
        elif avg_score >= 70:
            report.append("⚠ MACE potential shows good stability with minor issues")
            report.append("⚠ Monitor closely in production runs")
        else:
            report.append("✗ MACE potential shows stability issues")
            report.append("✗ Consider different model or parameters")
            report.append("✗ Investigate force spikes and energy drift")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


    

