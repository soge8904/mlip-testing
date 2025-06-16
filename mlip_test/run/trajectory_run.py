
#!/usr/bin/env python3

import os
import argparse
from mlip_test.protocols.trajectory_eval import MACETrajectoryEvaluator, MACETrajectoryAnalyzer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate MACE potential on trajectory data')
    
    # Required arguments
    parser.add_argument('--trajectory', required=True, 
                       help='Path to XYZ trajectory file with reference data')
    parser.add_argument('--model', required=True, 
                       help='Path to MACE model (.pth file)')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='mace_trajectory_evaluation',
                       help='Output directory for results and plots')
    parser.add_argument('--energy_key', default='energy_REF',
                       help='Key for reference energy in atoms.info')
    parser.add_argument('--forces_key', default='forces_REF', 
                       help='Key for reference forces in atoms.arrays')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for calculations')
    parser.add_argument('--dtype', default='float64', choices=['float32', 'float64'],
                       help='Default dtype for calculations')
    
    # Analysis options
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate diagnostic plots')
    parser.add_argument('--generate_report', action='store_true', 
                       help='Generate evaluation report')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON file')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    print("="*60)
    print("MACE TRAJECTORY EVALUATION")
    print("="*60)
    print(f"Trajectory file: {args.trajectory}")
    print(f"MACE model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Check if input files exist
    if not os.path.exists(args.trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"MACE model not found: {args.model}")
    
    try:
        # Initialize evaluator
        evaluator = MACETrajectoryEvaluator(
            model_path=args.model,
            device=args.device,
            default_dtype=args.dtype,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        print("\nStarting trajectory evaluation...")
        results = evaluator.evaluate_trajectory(
            xyz_file=args.trajectory,
            energy_key=args.energy_key,
            forces_key=args.forces_key
        )
        
        # Save results if requested
        if args.save_results:
            evaluator.save_results()
        
        # Initialize analyzer
        analyzer = MACETrajectoryAnalyzer(args.output_dir)
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = analyzer.analyze_results(results=results)
        
        # Generate plots if requested
        if args.generate_plots:
            print("Generating diagnostic plots...")
            analyzer.generate_plots()
        
        # Generate report if requested
        if args.generate_report:
            print("Generating evaluation report...")
            report_path = os.path.join(args.output_dir, 'trajectory_evaluation_report.txt')
            report = analyzer.generate_report(save_path=report_path)
            print(f"Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        energy_results = analysis.get('energy', {})
        force_results = analysis.get('forces', {})
        overall_score = analysis.get('overall_score', 0)
        
        print(f"Overall Score: {overall_score:.1f}/100")
        print(f"Energy MAE: {energy_results.get('mae_per_atom', 0):.2f} meV/atom")
        print(f"Energy R²: {energy_results.get('r_squared', 0):.6f}")
        print(f"Force MAE: {force_results.get('mae', 0):.6f} eV/Å")
        print(f"Force R²: {force_results.get('r_squared', 0):.6f}")
        print(f"Frames processed: {results.get('n_frames_processed', 0)}/{results.get('n_frames_total', 0)}")
        
        if overall_score >= 75:
            print("\n✓ Model shows good performance!")
        elif overall_score >= 60:
            print("\n⚠ Model shows fair performance - monitor closely")
        else:
            print("\n✗ Model shows poor performance - improvement needed")
        
        print("="*60)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())