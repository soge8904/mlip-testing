import os
from mlip_test.protocols.trajectory_eval import MACECommitteeAnalyzer
import matplotlib.pyplot as plt
seed_137 = "trajectory_evaluation_test_data.json"
seed_6 = "seed_6_trajectory_evaluation_test_data.json"
seed_89 = "seed_89_trajectory_evaluation_test_data.json"
result_files = [seed_137, seed_6, seed_89]
results_dir = "/Users/sophi/DATA/IrO2/MLIP/all_data_06_14_2025/testing/model_stage2"
# analyzer = MACECommitteeAnalyzer(results_dir=results_dir)
# results = analyzer.load_individual_model_results(result_files=result_files)
# analysis = analyzer.analyze_committee_results(results)
# analyzer.generate_committee_plots(save_dir=results_dir)
# try:
#     analyzer = MACECommitteeAnalyzer(results_dir=results_dir)
#     analysis = analyzer.analyze_committee_results(model_result_files=result_files)
    
#     # Generate plots
#     analyzer.generate_committee_plots(save_dir=results_dir)
    
#     # Generate and save report
#     report = analyzer.generate_committee_report(save_path=os.path.join(results_dir, 'committee_report.txt'))
#     print("Committee Analysis Summary:")
#     print("="*50)
#     print(f"Energy ensemble MAE: {analysis['energy']['ensemble']['mae_per_atom']:.2f} meV/atom")
#     print(f"Force ensemble MAE: {analysis['forces']['ensemble']['mae']:.4f} eV/Å")
#     print(f"Energy uncertainty correlation: {analysis['uncertainty']['energy_uncertainty_error_correlation']:.3f}")
#     print("="*50)
    
# except Exception as e:
#     print(f"Error in committee analysis: {e}")
#     raise

try:
    analyzer = MACECommitteeAnalyzer(results_dir=results_dir)
    analysis = analyzer.analyze_committee_results(model_result_files=result_files)
    
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    analyzer._plot_energy_parity_all_models(ax1, 5000)
    analyzer._plot_force_parity_all_models(ax2, 5000)
    plt.show()
    
except Exception as e:
    print(f"Error in committee analysis: {e}")
    raise