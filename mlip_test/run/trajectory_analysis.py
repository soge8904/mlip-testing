import os
from mlip_test.protocols.trajectory_eval import MACETrajectoryAnalyzer

analyzer = MACETrajectoryAnalyzer("/Users/sophi/DATA/IrO2/MLIP/all_data_06_14_2025/testing/model_stage2")
analysis_results = analyzer.analyze_results(results_file="seed_6_trajectory_evaluation_test_data.json")
analyzer.generate_plots()
