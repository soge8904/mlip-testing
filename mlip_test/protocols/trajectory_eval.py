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
    
class MACECommitteeAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.analysis_results = {}

    def load_individual_model_results(self, result_files: List[str]=None) -> Dict:
        """
        Load and combine results from separate JSON files for each model
        
        Args:
            model_result_files: List of paths to individual model result JSON files
            
        Returns:
            Combined results dictionary in committee format
        """
        individual_results = []
        model_paths = []

        for i, result_file in enumerate(result_files):
            file_path = os.path.join(self.results_dir, result_file) if not os.path.isabs(result_file) else result_file
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Results file not found: {file_path}")
            
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            # Convert lists back to numpy arrays
            processed_result = {}
            for key, value in result.items():
                if key in ['dft_energies', 'mace_energies', 'dft_forces', 'mace_forces', 'frame_indices']:
                    processed_result[key] = np.array(value)
                else:
                    processed_result[key] = value
            
            individual_results.append(processed_result)
            model_paths.append(result_file)
            print(f"✓ Loaded model {i+1}: {result_file}")

        # Find common frame indices across all models
        all_frame_indices = [set(result['frame_indices']) for result in individual_results]
        common_frames = set.intersection(*all_frame_indices)
        
        if len(common_frames) < len(all_frame_indices[0]):
            print(f"Warning: Models evaluated different frames. Using {len(common_frames)} common frames.")
            missing_per_model = [len(all_frame_indices[0]) - len(common_frames & frames) 
                               for frames in all_frame_indices]
            for i, missing in enumerate(missing_per_model):
                if missing > 0:
                    print(f"  Model {i+1} missing {missing} frames from common set")
        
        common_frames_sorted = sorted(list(common_frames))
        
        # Align data to common frames
        aligned_results = []
        reference_energies = None
        reference_forces = None
        
        for i, result in enumerate(individual_results):
            # Find indices for common frames
            frame_mask = np.isin(result['frame_indices'], common_frames_sorted)
            common_indices = result['frame_indices'][frame_mask]
            
            # Sort to match common_frames_sorted order
            sort_order = np.argsort(common_indices)
            final_mask = frame_mask.nonzero()[0][sort_order]
            
            aligned_result = {
                'dft_energies': result['dft_energies'][final_mask],
                'mace_energies': result['mace_energies'][final_mask],
                'frame_indices': result['frame_indices'][final_mask]
            }
            
            # Handle forces (need to reshape and align)
            if 'dft_forces' in result and 'mace_forces' in result:
                n_atoms = result.get('n_atoms', 0)
                if n_atoms > 0:
                    # Reshape forces to (n_frames, n_atoms, 3) then select and flatten
                    dft_forces_reshaped = result['dft_forces'].reshape(-1, n_atoms * 3)
                    mace_forces_reshaped = result['mace_forces'].reshape(-1, n_atoms * 3)
                    
                    aligned_result['dft_forces'] = dft_forces_reshaped[final_mask].flatten()
                    aligned_result['mace_forces'] = mace_forces_reshaped[final_mask].flatten()
                else:
                    aligned_result['dft_forces'] = result['dft_forces'][final_mask]
                    aligned_result['mace_forces'] = result['mace_forces'][final_mask]
            
            # Set reference data from first model
            if i == 0:
                reference_energies = aligned_result['dft_energies']
                reference_forces = aligned_result['dft_forces']
            else:
                # Verify consistency with reference
                if not np.allclose(aligned_result['dft_energies'], reference_energies, rtol=1e-8):
                    print(f"Warning: Model {i+1} has slightly different reference energies (max diff: {np.max(np.abs(aligned_result['dft_energies'] - reference_energies)):.2e})")
                if 'dft_forces' in aligned_result and not np.allclose(aligned_result['dft_forces'], reference_forces, rtol=1e-8):
                    print(f"Warning: Model {i+1} has slightly different reference forces (max diff: {np.max(np.abs(aligned_result['dft_forces'] - reference_forces)):.2e})")
            
            aligned_results.append(aligned_result)
        
        # Combine into committee format
        committee_results = {
            'dft_energies': reference_energies,
            'dft_forces': reference_forces,
            'committee_energies': [result['mace_energies'] for result in aligned_results],
            'committee_forces': [result['mace_forces'] for result in aligned_results],
            'frame_indices': np.array(common_frames_sorted),
            'failed_frames': [],  # Assume no failed frames if we have results
            'n_atoms': individual_results[0].get('n_atoms', 0),
            'n_frames_processed': len(common_frames_sorted),
            'n_frames_total': max(result.get('n_frames_total', len(common_frames_sorted)) for result in individual_results),
            'n_models': len(individual_results),
            'model_paths': model_paths
        }
        
        # Combine failed frames from all models (exclude frames that are just missing from other models)
        all_failed_frames = []
        for result in individual_results:
            failed = result.get('failed_frames', [])
            # Only include as failed if it's not just missing from other models
            truly_failed = [f for f in failed if f in common_frames]
            all_failed_frames.extend(truly_failed)
        committee_results['failed_frames'] = list(set(all_failed_frames))
        
        print(f"✓ Successfully combined {len(individual_results)} model results")
        print(f"  Common frames: {len(common_frames_sorted)}")
        print(f"  Committee size: {len(individual_results)} models")
        
        return committee_results
    
    def analyze_committee_results(self, results: Dict = None, results_file: str = None, 
                                model_result_files: List[str] = None) -> Dict:
        """
        Analyze MACE committee vs reference data correlation and errors
        
        Args:
            results: Results dictionary from committee evaluator
            results_file: Path to committee results JSON file 
            model_result_files: List of paths to individual model result JSON files
            
        Returns:
            Analysis results dictionary
        """
        if results is not None:
            self.results = results
        elif model_result_files is not None:
            # Load and combine individual model results
            self.results = self.load_individual_model_results(model_result_files)
        elif results_file is not None:
            with open(os.path.join(self.results_dir, results_file), 'r') as f:
                results_dict = json.load(f)
            # Convert back to numpy arrays
            self.results = {}
            for key, value in results_dict.items():
                if key == 'committee_energies':
                    self.results[key] = [np.array(model_data) for model_data in value]
                elif key == 'committee_forces':
                    self.results[key] = [np.array(model_data) for model_data in value]
                elif key in ['dft_energies', 'dft_forces', 'frame_indices']:
                    self.results[key] = np.array(value)
                else:
                    self.results[key] = value
        else:
            raise ValueError("Must provide either results dict, results_file, or model_result_files")
        
        analysis = {}
        analysis['energy'] = self._analyze_committee_energies()
        analysis['forces'] = self._analyze_committee_forces()
        analysis['ensemble'] = self._analyze_ensemble_statistics()
        analysis['uncertainty'] = self._analyze_uncertainty()
        
        self.analysis_results = analysis
        return analysis
    
    def _analyze_committee_energies(self) -> Dict:
        """Analyze energy predictions for committee"""
        dft_e = self.results['dft_energies']
        committee_e = self.results['committee_energies']
        n_atoms = self.results['n_atoms']
        n_models = self.results['n_models']
        
        # Calculate ensemble mean and std
        committee_array = np.stack(committee_e, axis=0)  # Shape: (n_models, n_frames)
        ensemble_mean = np.mean(committee_array, axis=0)
        ensemble_std = np.std(committee_array, axis=0)
        
        # Individual model statistics
        model_stats = []
        for i, model_energies in enumerate(committee_e):
            errors = model_energies - dft_e
            abs_errors = np.abs(errors)
            errors_per_atom = errors / n_atoms * 1000  # meV/atom
            abs_errors_per_atom = abs_errors / n_atoms * 1000
            
            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(errors**2))
            mae_per_atom = np.mean(abs_errors_per_atom)
            rmse_per_atom = np.sqrt(np.mean(errors_per_atom**2))
            r_value, p_value = pearsonr(dft_e, model_energies)
            
            model_stats.append({
                'model_id': i,
                'mae': mae,
                'rmse': rmse,
                'mae_per_atom': mae_per_atom,
                'rmse_per_atom': rmse_per_atom,
                'correlation': r_value,
                'r_squared': r_value**2
            })
        
        # Ensemble statistics
        ensemble_errors = ensemble_mean - dft_e
        ensemble_abs_errors = np.abs(ensemble_errors)
        ensemble_errors_per_atom = ensemble_errors / n_atoms * 1000
        ensemble_abs_errors_per_atom = ensemble_abs_errors / n_atoms * 1000
        
        ensemble_mae = np.mean(ensemble_abs_errors)
        ensemble_rmse = np.sqrt(np.mean(ensemble_errors**2))
        ensemble_mae_per_atom = np.mean(ensemble_abs_errors_per_atom)
        ensemble_rmse_per_atom = np.sqrt(np.mean(ensemble_errors_per_atom**2))
        ensemble_r_value, ensemble_p_value = pearsonr(dft_e, ensemble_mean)
        
        return {
            'model_stats': model_stats,
            'ensemble': {
                'mae': ensemble_mae,
                'rmse': ensemble_rmse,
                'mae_per_atom': ensemble_mae_per_atom,
                'rmse_per_atom': ensemble_rmse_per_atom,
                'correlation': ensemble_r_value,
                'r_squared': ensemble_r_value**2,
                'mean_uncertainty': np.mean(ensemble_std),
                'max_uncertainty': np.max(ensemble_std),
                'mean_uncertainty_per_atom': np.mean(ensemble_std) / n_atoms * 1000  # meV/atom
            },
            'committee_diversity': np.mean(ensemble_std),  # Average standard deviation across frames
            'n_models': n_models
        }
    
    def _analyze_committee_forces(self) -> Dict:
        """Analyze force predictions for committee"""
        dft_f = self.results['dft_forces']
        committee_f = self.results['committee_forces']
        n_models = self.results['n_models']
        
        # Calculate ensemble statistics
        committee_array = np.stack(committee_f, axis=0)  # Shape: (n_models, n_force_components)
        ensemble_mean = np.mean(committee_array, axis=0)
        ensemble_std = np.std(committee_array, axis=0)
        
        # Individual model statistics
        model_stats = []
        for i, model_forces in enumerate(committee_f):
            errors = model_forces - dft_f
            abs_errors = np.abs(errors)
            
            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(errors**2))
            r_value, p_value = pearsonr(dft_f, model_forces)
            
            model_stats.append({
                'model_id': i,
                'mae': mae,
                'rmse': rmse,
                'correlation': r_value,
                'r_squared': r_value**2
            })
        
        # Ensemble statistics
        ensemble_errors = ensemble_mean - dft_f
        ensemble_abs_errors = np.abs(ensemble_errors)
        
        ensemble_mae = np.mean(ensemble_abs_errors)
        ensemble_rmse = np.sqrt(np.mean(ensemble_errors**2))
        ensemble_r_value, ensemble_p_value = pearsonr(dft_f, ensemble_mean)
        
        return {
            'model_stats': model_stats,
            'ensemble': {
                'mae': ensemble_mae,
                'rmse': ensemble_rmse,
                'correlation': ensemble_r_value,
                'r_squared': ensemble_r_value**2,
                'mean_uncertainty': np.mean(ensemble_std),
                'max_uncertainty': np.max(ensemble_std)
            },
            'committee_diversity': np.mean(ensemble_std),
            'n_models': n_models
        }
    
    def _analyze_ensemble_statistics(self) -> Dict:
        """Analyze ensemble-specific statistics"""
        committee_e = self.results['committee_energies']
        committee_f = self.results['committee_forces']
        
        # Energy ensemble statistics
        energy_array = np.stack(committee_e, axis=0)
        energy_range = np.max(energy_array, axis=0) - np.min(energy_array, axis=0)
        energy_agreement = 1.0 - (np.std(energy_array, axis=0) / np.mean(np.abs(energy_array), axis=0))
        
        # Force ensemble statistics  
        force_array = np.stack(committee_f, axis=0)
        force_range = np.max(force_array, axis=0) - np.min(force_array, axis=0)
        force_agreement = 1.0 - (np.std(force_array, axis=0) / (np.mean(np.abs(force_array), axis=0) + 1e-8))
        
        return {
            'energy_range_mean': np.mean(energy_range),
            'energy_range_max': np.max(energy_range),
            'energy_agreement_mean': np.mean(energy_agreement[~np.isnan(energy_agreement)]),
            'force_range_mean': np.mean(force_range),
            'force_range_max': np.max(force_range),
            'force_agreement_mean': np.mean(force_agreement[~np.isnan(force_agreement)]),
        }
    
    def _analyze_uncertainty(self) -> Dict:
        """Analyze uncertainty quantification capabilities"""
        dft_e = self.results['dft_energies']
        dft_f = self.results['dft_forces']
        committee_e = self.results['committee_energies']
        committee_f = self.results['committee_forces']
        
        # Calculate ensemble predictions and uncertainties
        energy_array = np.stack(committee_e, axis=0)
        force_array = np.stack(committee_f, axis=0)
        
        energy_mean = np.mean(energy_array, axis=0)
        energy_std = np.std(energy_array, axis=0)
        force_mean = np.mean(force_array, axis=0)
        force_std = np.std(force_array, axis=0)
        
        # Calculate absolute errors for ensemble predictions
        energy_abs_errors = np.abs(energy_mean - dft_e)
        force_abs_errors = np.abs(force_mean - dft_f)
        
        # Correlation between uncertainty and error
        energy_uncertainty_error_corr, _ = pearsonr(energy_std, energy_abs_errors)
        force_uncertainty_error_corr, _ = pearsonr(force_std, force_abs_errors)
        
        return {
            'energy_uncertainty_error_correlation': energy_uncertainty_error_corr,
            'force_uncertainty_error_correlation': force_uncertainty_error_corr,
            'energy_uncertainty_range': (np.min(energy_std), np.max(energy_std)),
            'force_uncertainty_range': (np.min(force_std), np.max(force_std)),
            'high_uncertainty_frames': np.where(energy_std > np.percentile(energy_std, 95))[0].tolist()
        }
    
    def generate_committee_plots(self, save_dir: str = None, max_points: int = 5000):
        """Generate comprehensive committee analysis plots"""
        if not hasattr(self, 'results'):
            raise ValueError("No results loaded. Run analyze_committee_results first.")
        
        if save_dir is None:
            save_dir = self.results_dir
        
        # Create subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Energy parity plot with all models
        ax1 = plt.subplot(3, 3, 1)
        self._plot_energy_parity_all_models(ax1, max_points)
        
        # Plot 2: Energy parity plot with ensemble mean and error bars
        ax2 = plt.subplot(3, 3, 2)
        self._plot_energy_parity_ensemble(ax2, max_points)
        
        # Plot 3: Force parity plot with all models
        ax3 = plt.subplot(3, 3, 3)
        self._plot_force_parity_all_models(ax3, max_points)
        
        # Plot 4: Force parity plot with ensemble mean and error bars
        ax4 = plt.subplot(3, 3, 4)
        self._plot_force_parity_ensemble(ax4, max_points)
        
        # Plot 5: Energy uncertainty vs error
        ax5 = plt.subplot(3, 3, 5)
        self._plot_uncertainty_vs_error(ax5, 'energy', max_points)
        
        # Plot 6: Force uncertainty vs error
        ax6 = plt.subplot(3, 3, 6)
        self._plot_uncertainty_vs_error(ax6, 'force', max_points)
        
        # Plot 7: Model performance comparison
        ax7 = plt.subplot(3, 3, 7)
        self._plot_model_performance_comparison(ax7)
        
        # Plot 8: Uncertainty distribution
        ax8 = plt.subplot(3, 3, 8)
        self._plot_uncertainty_distribution(ax8)
        
        # Plot 9: Agreement between models
        ax9 = plt.subplot(3, 3, 9)
        self._plot_model_agreement(ax9)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/committee_analysis_plots.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def _plot_energy_parity_all_models(self, ax, max_points):
        """Plot energy parity with all models overlaid"""
        dft_e = self.results['dft_energies']
        committee_e = self.results['committee_energies']
        n_atoms = self.results['n_atoms']
        
        dft_e_per_atom = dft_e / n_atoms
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(committee_e)))
        
        # Sample points if too many
        if len(dft_e) > max_points:
            idx = np.random.choice(len(dft_e), max_points, replace=False)
        else:
            idx = np.arange(len(dft_e))
        
        for i, model_energies in enumerate(committee_e):
            model_e_per_atom = model_energies / n_atoms
            ax.scatter(dft_e_per_atom[idx], model_e_per_atom[idx], 
                      alpha=0.6, s=10, color=colors[i], label=f'Model {i+1}')
        
        # Perfect prediction line
        min_val, max_val = ax.get_xlim()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        ax.set_xlabel('DFT Energy (eV/atom)')
        ax.set_ylabel('MACE Energy (eV/atom)')
        ax.set_title('Energy Parity - All Models')
        ax.grid(True, alpha=0.3)
        if len(committee_e) <= 10:  # Only show legend if not too many models
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_energy_parity_ensemble(self, ax, max_points):
        """Plot energy parity with ensemble mean and error bars"""
        dft_e = self.results['dft_energies']
        committee_e = self.results['committee_energies']
        n_atoms = self.results['n_atoms']
        
        # Calculate ensemble statistics
        energy_array = np.stack(committee_e, axis=0)
        ensemble_mean = np.mean(energy_array, axis=0)
        ensemble_std = np.std(energy_array, axis=0)
        
        dft_e_per_atom = dft_e / n_atoms
        ensemble_mean_per_atom = ensemble_mean / n_atoms
        ensemble_std_per_atom = ensemble_std / n_atoms
        
        # Sample points if too many
        if len(dft_e) > max_points:
            idx = np.random.choice(len(dft_e), max_points, replace=False)
        else:
            idx = np.arange(len(dft_e))
        
        ax.errorbar(dft_e_per_atom[idx], ensemble_mean_per_atom[idx], 
                   yerr=ensemble_std_per_atom[idx], fmt='o', alpha=0.6, 
                   markersize=4, capsize=2, color='blue')
        
        # Perfect prediction line
        min_val, max_val = ax.get_xlim()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        ax.set_xlabel('DFT Energy (eV/atom)')
        ax.set_ylabel('Ensemble Mean Energy (eV/atom)')
        ax.set_title('Energy Parity - Ensemble Mean ± Std')
        ax.grid(True, alpha=0.3)
    
    def _plot_force_parity_all_models(self, ax, max_points):
        """Plot force parity with all models overlaid"""
        dft_f = self.results['dft_forces']
        committee_f = self.results['committee_forces']
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(committee_f)))
        
        # Sample points if too many
        if len(dft_f) > max_points:
            idx = np.random.choice(len(dft_f), max_points, replace=False)
        else:
            idx = np.arange(len(dft_f))
        
        for i, model_forces in enumerate(committee_f):
            ax.scatter(dft_f[idx], model_forces[idx], 
                      alpha=0.4, s=5, color=colors[i], label=f'Model {i+1}')
        
        # Perfect prediction line
        min_val, max_val = ax.get_xlim()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        ax.set_xlabel('DFT Forces (eV/Å)')
        ax.set_ylabel('MACE Forces (eV/Å)')
        ax.set_title('Force Parity - All Models')
        ax.grid(True, alpha=0.3)
    
    def _plot_force_parity_ensemble(self, ax, max_points):
        """Plot force parity with ensemble mean and error bars"""
        dft_f = self.results['dft_forces']
        committee_f = self.results['committee_forces']
        
        # Calculate ensemble statistics
        force_array = np.stack(committee_f, axis=0)
        ensemble_mean = np.mean(force_array, axis=0)
        ensemble_std = np.std(force_array, axis=0)
        
        # Sample points if too many
        if len(dft_f) > max_points:
            idx = np.random.choice(len(dft_f), max_points, replace=False)
        else:
            idx = np.arange(len(dft_f))
        
        # Use every nth point for error bars to avoid overcrowding
        n_error_bars = min(500, len(idx))
        error_idx = idx[::len(idx)//n_error_bars] if len(idx) > n_error_bars else idx
        
        ax.scatter(dft_f[idx], ensemble_mean[idx], alpha=0.4, s=5, color='blue')
        ax.errorbar(dft_f[error_idx], ensemble_mean[error_idx], 
                   yerr=ensemble_std[error_idx], fmt='none', alpha=0.3, 
                   capsize=1, color='blue')
        
        # Perfect prediction line
        min_val, max_val = ax.get_xlim()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
        
        ax.set_xlabel('DFT Forces (eV/Å)')
        ax.set_ylabel('Ensemble Mean Forces (eV/Å)')
        ax.set_title('Force Parity - Ensemble Mean ± Std')
        ax.grid(True, alpha=0.3)
    
    def _plot_uncertainty_vs_error(self, ax, quantity, max_points):
        """Plot uncertainty vs actual error"""
        if quantity == 'energy':
            dft_data = self.results['dft_energies']
            committee_data = self.results['committee_energies']
            n_atoms = self.results['n_atoms']
            ylabel = 'Energy Uncertainty (eV/atom)'
            xlabel = 'Energy Error (eV/atom)'
        else:  # force
            dft_data = self.results['dft_forces']
            committee_data = self.results['committee_forces']
            n_atoms = 1
            ylabel = 'Force Uncertainty (eV/Å)'
            xlabel = 'Force Error (eV/Å)'
        
        # Calculate ensemble statistics
        data_array = np.stack(committee_data, axis=0)
        ensemble_mean = np.mean(data_array, axis=0)
        ensemble_std = np.std(data_array, axis=0)
        
        # Calculate errors
        errors = np.abs(ensemble_mean - dft_data)
        
        if quantity == 'energy':
            errors = errors / n_atoms
            ensemble_std = ensemble_std / n_atoms
        
        # Sample points if too many
        if len(errors) > max_points:
            idx = np.random.choice(len(errors), max_points, replace=False)
        else:
            idx = np.arange(len(errors))
        
        ax.scatter(errors[idx], ensemble_std[idx], alpha=0.6, s=10)
        
        # Calculate and show correlation
        corr, _ = pearsonr(ensemble_std, errors)
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{quantity.capitalize()} Uncertainty vs Error')
        ax.grid(True, alpha=0.3)
    
    def _plot_model_performance_comparison(self, ax):
        """Plot comparison of individual model performance"""
        if 'energy' not in self.analysis_results:
            return
            
        energy_stats = self.analysis_results['energy']['model_stats']
        
        model_ids = [stat['model_id'] + 1 for stat in energy_stats]
        maes = [stat['mae_per_atom'] for stat in energy_stats]
        r_squared = [stat['r_squared'] for stat in energy_stats]
        
        # Add ensemble performance
        ensemble_mae = self.analysis_results['energy']['ensemble']['mae_per_atom']
        ensemble_r2 = self.analysis_results['energy']['ensemble']['r_squared']
        
        model_ids.append('Ensemble')
        maes.append(ensemble_mae)
        r_squared.append(ensemble_r2)
        
        # Create bar plot
        x_pos = np.arange(len(model_ids))
        bars = ax.bar(x_pos, maes, alpha=0.7)
        
        # Color ensemble bar differently
        bars[-1].set_color('red')
        bars[-1].set_alpha(0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('MAE (meV/atom)')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_ids, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add R² values as text
        for i, (mae, r2) in enumerate(zip(maes, r_squared)):
            ax.text(i, mae + max(maes)*0.01, f'R²={r2:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_uncertainty_distribution(self, ax):
        """Plot distribution of prediction uncertainties"""
        committee_e = self.results['committee_energies']
        committee_f = self.results['committee_forces']
        n_atoms = self.results['n_atoms']
        
        # Calculate uncertainties
        energy_array = np.stack(committee_e, axis=0)
        force_array = np.stack(committee_f, axis=0)
        
        energy_std = np.std(energy_array, axis=0) / n_atoms * 1000  # meV/atom
        force_std = np.std(force_array, axis=0)
        
        # Plot histograms
        ax.hist(energy_std, bins=50, alpha=0.7, label='Energy (meV/atom)', density=True)
        ax.hist(force_std, bins=50, alpha=0.7, label='Force (eV/Å)', density=True)
        
        ax.set_xlabel('Prediction Uncertainty')
        ax.set_ylabel('Density')
        ax.set_title('Uncertainty Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_agreement(self, ax):
        """Plot agreement between models across frames"""
        committee_e = self.results['committee_energies']
        n_atoms = self.results['n_atoms']
        
        # Calculate pairwise correlations between models
        n_models = len(committee_e)
        correlations = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                corr, _ = pearsonr(committee_e[i], committee_e[j])
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        # Plot correlation matrix
        im = ax.imshow(correlations, cmap='RdYlBu_r', vmin=0.8, vmax=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = ax.text(j, i, f'{correlations[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Model')
        ax.set_title('Model Agreement (Energy Correlations)')
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels([f'M{i+1}' for i in range(n_models)])
        ax.set_yticklabels([f'M{i+1}' for i in range(n_models)])
    
    def generate_committee_report(self, analysis_results: Dict = None, save_path: str = None) -> str:
        """Generate comprehensive committee evaluation report"""
        
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if not analysis_results:
            raise ValueError("No analysis results available. Run analyze_committee_results first.")
        
        report = []
        report.append("="*70)
        report.append("MACE COMMITTEE EVALUATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Committee overview
        if 'energy' in analysis_results:
            n_models = analysis_results['energy']['n_models']
            report.append(f"Committee Size: {n_models} models")
            report.append("")
        
        # Energy analysis
        if 'energy' in analysis_results:
            energy = analysis_results['energy']
            report.append("ENERGY EVALUATION:")
            report.append("-" * 25)
            
            # Individual model performance
            report.append("Individual Model Performance:")
            for model_stat in energy['model_stats']:
                mid = model_stat['model_id'] + 1
                mae = model_stat['mae_per_atom']
                rmse = model_stat['rmse_per_atom']
                r2 = model_stat['r_squared']
                report.append(f"  Model {mid}: MAE={mae:.2f} meV/atom, RMSE={rmse:.2f} meV/atom, R²={r2:.4f}")
            
            report.append("")
            
            # Ensemble performance
            ens = energy['ensemble']
            report.append("Ensemble Performance:")
            report.append(f"  MAE: {ens['mae_per_atom']:.2f} meV/atom")
            report.append(f"  RMSE: {ens['rmse_per_atom']:.2f} meV/atom")
            report.append(f"  Correlation (R²): {ens['r_squared']:.4f}")
            report.append(f"  Mean Uncertainty: {ens['mean_uncertainty_per_atom']:.2f} meV/atom")
            report.append(f"  Committee Diversity: {energy['committee_diversity']:.4f} eV")
            report.append("")
        
        # Force analysis
        if 'forces' in analysis_results:
            forces = analysis_results['forces']
            report.append("FORCE EVALUATION:")
            report.append("-" * 20)
            
            # Individual model performance
            report.append("Individual Model Performance:")
            for model_stat in forces['model_stats']:
                mid = model_stat['model_id'] + 1
                mae = model_stat['mae']
                rmse = model_stat['rmse']
                r2 = model_stat['r_squared']
                report.append(f"  Model {mid}: MAE={mae:.4f} eV/Å, RMSE={rmse:.4f} eV/Å, R²={r2:.4f}")
            
            report.append("")
            
            # Ensemble performance
            ens = forces['ensemble']
            report.append("Ensemble Performance:")
            report.append(f"  MAE: {ens['mae']:.4f} eV/Å")
            report.append(f"  RMSE: {ens['rmse']:.4f} eV/Å") 
            report.append(f"  Correlation (R²): {ens['r_squared']:.4f}")
            report.append(f"  Mean Uncertainty: {ens['mean_uncertainty']:.4f} eV/Å")
            report.append(f"  Committee Diversity: {forces['committee_diversity']:.4f} eV/Å")
            report.append("")
        
        # Uncertainty analysis
        if 'uncertainty' in analysis_results:
            uncertainty = analysis_results['uncertainty']
            report.append("UNCERTAINTY QUANTIFICATION:")
            report.append("-" * 30)
            report.append(f"  Energy uncertainty-error correlation: {uncertainty['energy_uncertainty_error_correlation']:.3f}")
            report.append(f"  Force uncertainty-error correlation: {uncertainty['force_uncertainty_error_correlation']:.3f}")
            
            e_range = uncertainty['energy_uncertainty_range']
            f_range = uncertainty['force_uncertainty_range']
            report.append(f"  Energy uncertainty range: {e_range[0]:.4f} - {e_range[1]:.4f} eV")
            report.append(f"  Force uncertainty range: {f_range[0]:.4f} - {f_range[1]:.4f} eV/Å")
            
            n_high_unc = len(uncertainty['high_uncertainty_frames'])
            report.append(f"  High uncertainty frames (>95th percentile): {n_high_unc}")
            report.append("")
        
        # Ensemble statistics
        if 'ensemble' in analysis_results:
            ensemble = analysis_results['ensemble']
            report.append("ENSEMBLE STATISTICS:")
            report.append("-" * 22)
            report.append(f"  Energy range (mean): {ensemble['energy_range_mean']:.4f} eV")
            report.append(f"  Energy range (max): {ensemble['energy_range_max']:.4f} eV")
            report.append(f"  Energy agreement: {ensemble['energy_agreement_mean']:.3f}")
            report.append(f"  Force range (mean): {ensemble['force_range_mean']:.4f} eV/Å")
            report.append(f"  Force range (max): {ensemble['force_range_max']:.4f} eV/Å")
            report.append(f"  Force agreement: {ensemble['force_agreement_mean']:.3f}")
            report.append("")
        
        # Performance assessment
        report.append("COMMITTEE ASSESSMENT:")
        report.append("-" * 23)
        
        # Check if ensemble improves over individual models
        if 'energy' in analysis_results:
            individual_maes = [stat['mae_per_atom'] for stat in analysis_results['energy']['model_stats']]
            ensemble_mae = analysis_results['energy']['ensemble']['mae_per_atom']
            best_individual = min(individual_maes)
            
            if ensemble_mae < best_individual:
                improvement = (best_individual - ensemble_mae) / best_individual * 100
                report.append(f"✓ Ensemble improves energy prediction by {improvement:.1f}% vs best individual model")
            else:
                degradation = (ensemble_mae - best_individual) / best_individual * 100
                report.append(f"⚠ Ensemble degrades energy prediction by {degradation:.1f}% vs best individual model")
        
        # Uncertainty reliability
        if 'uncertainty' in analysis_results:
            energy_unc_corr = analysis_results['uncertainty']['energy_uncertainty_error_correlation']
            force_unc_corr = analysis_results['uncertainty']['force_uncertainty_error_correlation']
            
            if energy_unc_corr > 0.5:
                report.append("✓ Energy uncertainty is well-calibrated (good error correlation)")
            else:
                report.append("⚠ Energy uncertainty may not be well-calibrated")
                
            if force_unc_corr > 0.5:
                report.append("✓ Force uncertainty is well-calibrated (good error correlation)")
            else:
                report.append("⚠ Force uncertainty may not be well-calibrated")
        
        # Committee diversity
        if 'energy' in analysis_results:
            diversity = analysis_results['energy']['committee_diversity']
            if diversity > 0.01:  # Threshold in eV
                report.append("✓ Committee shows good diversity in predictions")
            else:
                report.append("⚠ Committee models may be too similar (low diversity)")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 17)
        
        # Generate recommendations based on results
        if 'energy' in analysis_results:
            ensemble_mae = analysis_results['energy']['ensemble']['mae_per_atom']
            if ensemble_mae < 10:  # meV/atom
                report.append("✓ Excellent committee performance for energy predictions")
            elif ensemble_mae < 50:
                report.append("✓ Good committee performance for energy predictions")
            else:
                report.append("⚠ Committee energy predictions may need improvement")
        
        if 'forces' in analysis_results:
            ensemble_mae = analysis_results['forces']['ensemble']['mae']
            if ensemble_mae < 0.1:  # eV/Å
                report.append("✓ Excellent committee performance for force predictions")
            elif ensemble_mae < 0.3:
                report.append("✓ Good committee performance for force predictions")
            else:
                report.append("⚠ Committee force predictions may need improvement")
        
        if 'uncertainty' in analysis_results:
            energy_unc_corr = analysis_results['uncertainty']['energy_uncertainty_error_correlation']
            if energy_unc_corr > 0.6:
                report.append("✓ Use uncertainty estimates for active learning and error detection")
            else:
                report.append("⚠ Consider uncertainty calibration methods")
        
        report.append("")
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
