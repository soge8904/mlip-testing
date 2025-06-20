import os
import argparse
from ase.io import read, write
from mace.calculators import MACECalculator
from mlip_test.core.md_runner import MinimizationRunner
from mlip_test.core.monitor import MDMonitor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate MACE potential on trajectory data')
    
    # Required arguments
    parser.add_argument('--structure', required=True, 
                       help='Path to XYZ file with structure to minimize')
    parser.add_argument('--output_dir', default='mace_minimization',
                       help='Output directory for results and plots')
    parser.add_argument('--model', required=True, 
                       help='Path to MACE model (.pth file)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for calculations')
    
    return parser.parse_args()

def main():
    args = parse_args()
    atoms = read(args.structure)
    calculator = MACECalculator(args.model, device=args.device)
    monitor = MDMonitor(atoms, metrics=['energy', 'forces'], check_interval=10)
    minimizer = MinimizationRunner(atoms=atoms, calculator=calculator, monitor=monitor)

    results = minimizer.minimize(
        optimizer='BFGS',
        fmax=0.01,  # eV/Å
        steps=500,
        trajectory_file='optimization.traj'
    )