import os
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
from ase.io import read, write
import torch
from mace.calculators.mace import MACECalculator

class MACETrajectoryEvaluator:
    """Evaluate MACE potential against reference trajectory data"""
    
    def __init__(self, model_path: str, device: str = 'cuda', default_dtype: str = 'float64', 
                 output_dir: str = 'mace_trajectory_evaluation'):
        """
        Initialize MACE trajectory evaluator
        
        Args:
            model_path: Path to MACE model (.pth file)
            device: Device to use (cuda/cpu)  
            default_dtype: Data type for calculations (float32/float64)
            output_dir: Directory for output files
        """
        self.model_path = model_path
        self.device = device
        self.default_dtype = default_dtype
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            self.calculator = MACECalculator(
                model_paths=model_path, 
                dispersion=False, 
                default_dtype=default_dtype, 
                device=device, 
                enable_cueq=False
            )
            print(f"✓ Loaded MACE model from: {model_path}")
        except Exception as e:
            print(f"✗ Failed to load MACE model: {e}")
            raise
        
        self.results = {}
    
    def evaluate_trajectory(self, xyz_file: str, energy_key: str = 'energy_REF', 
                          forces_key: str = 'forces_REF') -> Dict:
        """
        Evaluate MACE potential on trajectory frames
        
        Args:
            xyz_file: Path to XYZ trajectory file
            energy_key: Key for reference energy in atoms.info
            forces_key: Key for reference forces in atoms.arrays
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Loading trajectory: {xyz_file}")
        
        try:
            atoms_list = read(xyz_file, index=':')
        except Exception as e:
            print(f"Error reading trajectory file: {e}")
            raise
        
        print(f"Total frames to evaluate: {len(atoms_list)}")
        
        dft_energies = []
        mace_energies = []
        dft_forces = []
        mace_forces = []
        frame_indices = []
        failed_frames = []
        

        for i, atoms in enumerate(atoms_list):
            try:
                atoms_for_mace = atoms.copy()
                
                # Get DFT reference data
                try:
                    # Try to get energy from the specified key first, then fallback
                    if energy_key in atoms.info:
                        dft_energy = atoms.info[energy_key]
                    else:
                        dft_energy = atoms.get_potential_energy()
                    
                    # Try to get forces from specified key first, then fallback  
                    if forces_key in atoms.arrays:
                        dft_force = atoms.arrays[forces_key]
                    else:
                        dft_force = atoms.get_forces()
                        
                except Exception as e:
                    print(f"Warning: Could not get reference data for frame {i}: {e}")
                    failed_frames.append(i)
                    continue
                
                # Calculate MACE predictions
                atoms_for_mace.calc = self.calculator
                mace_energy = atoms_for_mace.get_potential_energy()
                mace_force = atoms_for_mace.get_forces()
                
                dft_energies.append(dft_energy)
                mace_energies.append(mace_energy)
                dft_forces.append(dft_force.flatten())
                mace_forces.append(mace_force.flatten())
                frame_indices.append(i)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(atoms_list)} frames")
                    
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                failed_frames.append(i)
                continue
        
        if not dft_energies:
            raise ValueError("No valid frames could be processed")
        
        self.results = {
            'dft_energies': np.array(dft_energies),
            'mace_energies': np.array(mace_energies),
            'dft_forces': np.concatenate(dft_forces),
            'mace_forces': np.concatenate(mace_forces),
            'frame_indices': np.array(frame_indices),
            'failed_frames': failed_frames,
            'n_atoms': len(atoms_list[0]) if atoms_list else 0,
            'n_frames_processed': len(frame_indices),
            'n_frames_total': len(atoms_list)
        }
        
        print(f"✓ Successfully processed {len(frame_indices)}/{len(atoms_list)} frames")
        if failed_frames:
            print(f"⚠ Failed frames: {len(failed_frames)}")
        
        return self.results
    
    def save_results(self, filename: str = 'trajectory_evaluation_results.json'):
        """Save evaluation results to JSON file"""
        if not self.results:
            print("No results to save. Run evaluate_trajectory first.")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            else:
                results_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"✓ Results saved to: {filepath}")

    def load_results(self, filename: str = 'trajectory_evaluation_results.json'):
        """Load evaluation results from JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        # Convert lists back to numpy arrays
        self.results = {}
        for key, value in results_dict.items():
            if key in ['dft_energies', 'mace_energies', 'dft_forces', 'mace_forces', 'frame_indices']:
                self.results[key] = np.array(value)
            else:
                self.results[key] = value
        
        print(f"✓ Results loaded from: {filepath}")
        return self.results
    

class MACETrajectoryAnalyzer:
    """Analyze MACE trajectory evaluation results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.analysis_results = {}
    
    def analyze_results(self, results: Dict = None, results_file: str = None) -> Dict:
        """
        Analyze MACE vs reference data correlation and errors
        
        Args:
            results: Results dictionary from evaluator
            results_file: Path to results JSON file (if results not provided)
            
        Returns:
            Analysis results dictionary
        """
        if results is not None:
            self.results = results
        elif results_file is not None:
            with open(os.path.join(self.results_dir, results_file), 'r') as f:
                results_dict = json.load(f)
            # Convert back to numpy arrays
            self.results = {}
            for key, value in results_dict.items():
                if key in ['dft_energies', 'mace_energies', 'dft_forces', 'mace_forces', 'frame_indices']:
                    self.results[key] = np.array(value)
                else:
                    self.results[key] = value
        else:
            raise ValueError("Either results dict or results_file must be provided")
        
        analysis = {}
        analysis['energy'] = self._analyze_energies()
        analysis['forces'] = self._analyze_forces()
        self.analysis_results = analysis

        return analysis
    
    def _analyze_energies(self) -> Dict:
        """Analyze energy predictions"""
        dft_e = self.results['dft_energies']
        mace_e = self.results['mace_energies']
        n_atoms = self.results['n_atoms']

        #ERRORS
        errors = mace_e - dft_e
        abs_errors = np.abs(errors)
        errors_per_atom = errors / n_atoms * 1000  # meV/atom
        abs_errors_per_atom = abs_errors / n_atoms * 1000  # meV/atom

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        mae_per_atom = np.mean(abs_errors_per_atom)
        rmse_per_atom = np.sqrt(np.mean(errors_per_atom**2))
        
        # Correlation
        r_value, p_value = pearsonr(dft_e, mace_e)

        return {
            'mae': mae,  # eV
            'rmse': rmse,  # eV  
            'mae_per_atom': mae_per_atom,  # meV/atom
            'rmse_per_atom': rmse_per_atom,  # meV/atom
            'correlation': r_value,
            'correlation_p_value': p_value,
            'r_squared': r_value**2,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_abs_error': np.max(abs_errors),
            'n_frames': len(dft_e)
        }
    
    def _analyze_forces(self) -> Dict:
        """Analyze force predictions"""
        dft_f = self.results['dft_forces']
        mace_f = self.results['mace_forces']
        
        # Errors
        errors = mace_f - dft_f
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Correlation
        r_value, p_value = pearsonr(dft_f, mace_f)
        
        # Force component analysis
        n_atoms = self.results['n_atoms']
        n_frames = len(self.results['frame_indices'])
        dft_f_reshaped = dft_f.reshape(n_frames, n_atoms, 3) #nned to reshape to get per component analysis
        mace_f_reshaped = mace_f.reshape(n_frames, n_atoms, 3)
        component_maes = []
        for i in range(3):  # x, y, z components
            comp_errors = np.abs(mace_f_reshaped[:, :, i] - dft_f_reshaped[:, :, i])
            component_maes.append(np.mean(comp_errors))
        
        return {
            'mae': mae,  # eV/Å
            'rmse': rmse,  # eV/Å
            'correlation': r_value,
            'correlation_p_value': p_value,
            'r_squared': r_value**2,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_abs_error': np.max(abs_errors),
            'component_mae_x': component_maes[0],
            'component_mae_y': component_maes[1],
            'component_mae_z': component_maes[2],
            'n_components': len(dft_f)
        }
    
    def generate_plots(self, save_dir: str = None):
        if not hasattr(self, 'results'):
            raise ValueError("No results loaded. Run analyze_results first.")
        
        if save_dir is None:
            save_dir = self.results_dir
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MACE Trajectory Evaluation Results', fontsize=14)

         # Energy parity plot
        dft_e = self.results['dft_energies']
        mace_e = self.results['mace_energies']
        dft_e_per_atom = self.results['dft_energies']/408
        mace_e_per_atom = self.results['mace_energies']/408

        
        axes[0,0].scatter(dft_e_per_atom, mace_e_per_atom, alpha=0.6, s=20)
        axes[0,0].plot([dft_e_per_atom.min(), dft_e_per_atom.max()], [dft_e_per_atom.min(), dft_e_per_atom.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('DFT Energy (eV/atom)')
        axes[0,0].set_ylabel('MACE Energy (eV/atom)')
        axes[0,0].set_title('Energy Parity Plot')
        axes[0,0].grid(True, alpha=0.3)

         # Force parity plot (sample for visibility)
        dft_f = self.results['dft_forces']
        mace_f = self.results['mace_forces']
        
        # Sample forces for plotting (too many points otherwise)
        n_sample = min(10000, len(dft_f))
        idx = np.random.choice(len(dft_f), n_sample, replace=False)
        
        axes[0,1].scatter(dft_f[idx], mace_f[idx], alpha=0.4, s=10)
        axes[0,1].plot([dft_f.min(), dft_f.max()], [dft_f.min(), dft_f.max()], 'r--', lw=2)
        axes[0,1].set_xlabel('DFT Forces (eV/Å)')
        axes[0,1].set_ylabel('MACE Forces (eV/Å)')
        axes[0,1].set_title(f'Force Parity Plot (sample of {n_sample})')
        axes[0,1].grid(True, alpha=0.3)

        # Energy error distribution
        energy_errors = (mace_e - dft_e) / self.results['n_atoms'] * 1000  # meV/atom
        axes[1,0].hist(energy_errors, bins=50, alpha=0.7, density=True)
        axes[1,0].axvline(np.mean(energy_errors), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(energy_errors):.2f} meV/atom')
        axes[1,0].set_xlabel('Energy Error (meV/atom)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Energy Error Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Force error distribution
        force_errors = mace_f - dft_f
        axes[1,1].hist(force_errors, bins=50, alpha=0.7, density=True)
        axes[1,1].axvline(np.mean(force_errors), color='red', linestyle='--',
                         label=f'Mean: {np.mean(force_errors):.4f} eV/Å')
        axes[1,1].set_xlabel('Force Error (eV/Å)')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Force Error Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/trajectory_evaluation_plots.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def generate_report(self, analysis_results: Dict = None, save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if not analysis_results:
            raise ValueError("No analysis results available. Run analyze_results first.")
        
        report = []
        report.append("="*60)
        report.append("MACE TRAJECTORY EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Energy analysis
        if 'energy' in analysis_results:
            energy = analysis_results['energy']
            report.append("ENERGY EVALUATION:")
            report.append("-" * 20)
            report.append(f"  Frames analyzed: {energy['n_frames']}")
            report.append(f"  MAE: {energy['mae']:.6f} eV ({energy['mae_per_atom']:.2f} meV/atom)")
            report.append(f"  RMSE: {energy['rmse']:.6f} eV ({energy['rmse_per_atom']:.2f} meV/atom)")
            report.append(f"  Correlation (R²): {energy['r_squared']:.6f}")
            report.append(f"  Max absolute error: {energy['max_abs_error']:.6f} eV")
            report.append("")
        
        # Force analysis
        if 'forces' in analysis_results:
            forces = analysis_results['forces']
            report.append("FORCE EVALUATION:")
            report.append("-" * 20)
            report.append(f"  Force components analyzed: {forces['n_components']}")
            report.append(f"  MAE: {forces['mae']:.6f} eV/Å")
            report.append(f"  RMSE: {forces['rmse']:.6f} eV/Å") 
            report.append(f"  Correlation (R²): {forces['r_squared']:.6f}")
            report.append(f"  Component MAE (x,y,z): {forces['component_mae_x']:.4f}, {forces['component_mae_y']:.4f}, {forces['component_mae_z']:.4f} eV/Å")
            report.append(f"  Max absolute error: {forces['max_abs_error']:.6f} eV/Å")
            report.append("")
        
        
        if 'energy' in analysis_results and analysis_results['energy']['mae_per_atom'] > 20:
            report.append("⚠ High energy errors")
        
        if 'forces' in analysis_results and analysis_results['forces']['mae'] > 0.2:
            report.append("⚠ High force errors")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text