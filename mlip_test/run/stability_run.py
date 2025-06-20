import os
from ase.io import read
from json_tricks import dump
from mace.calculators import mace_mp
from mlip_test.protocols.stability import MACEStabilityTest, MACEStabilityAnalyzer

atoms = read("/Users/sophi/envs/atomate/src/mlip_testing/mlip_test/test_files/initial_structure.xyz")
calculator = mace_mp(model='medium', device='cuda')
output_dir = "/Users/sophi/envs/atomate/src/mlip_testing/mlip_test/test_files"
stability_test = MACEStabilityTest(atoms, calculator, output_dir)

nvt = {'duration_ps': 0.01, 'temperature': 300, 'timestep': 1.0}
npt = {'duration_ps': 0.01, 'temperature': 300, 'timestep': 1.0, 'pressure': 0.0}
nve = {'duration_ps': 0.01, 'timestep': 1.0}

track_a, monitor = stability_test.run_track_a(nvt_params=nvt, npt_params=npt, nve_params=nve, monitor_interval=2)
save_monitor_data = True
monitor_data = monitor.data
if save_monitor_data is True:
    with open(os.path.join(output_dir, "track_a", "monitor_data.json"), "w") as json_file:
        dump(monitor, json_file)

results_dir=os.path.join(output_dir, "track_a")
analyzer = MACEStabilityAnalyzer(results_dir)
analysis = analyzer.analyze_monitor_data(monitor_data=monitor_data)
analyzer.generate_plots(monitor_data, results_dir)
report = analyzer.generate_report({"track_a": analysis}, 
                                     f"{results_dir}/stability_report.txt")



