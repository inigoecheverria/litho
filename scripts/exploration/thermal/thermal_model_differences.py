import functools
import numpy as np
from cmcrameri import cm
from litho.plots import diff_map, plot, heatmap_map
from litho.utils import makedir, makedir_from_filename, export_csv, calc_deviation
#from litho.colormaps import ()
from scripts.exploration.thermal.parameter_variations import (
    TM1_H0_variation, TM1_k_variation, TM2_H_variation, TM3_k_variation,
    thermal_model_variation
)
from scripts.exploration.thermal.thermal_results import thermal_results
lplot = plot

def main():
    save_dir = 'output/' + 'thermal_differences/'
    makedir(save_dir)

    results = thermal_model_variation(
        thermal_results(save_dir=save_dir, plot=True)
    )

main()
