import functools
import numpy as np
from litho import Lithosphere
from litho.thermal import TM1, TM2, TM3
from scripts.inputs import thermal_inputs, thermal_constants
from scripts.inputs_javi import thermal_inputs as thermal_inputs_javi
from scripts.inputs_0 import thermal_inputs as thermal_inputs_0
from litho.utils import makedir, makedir_from_filename
from scripts.exploration.thermal.parameter_variations import (
    TM1_H0_variation, TM1_k_variation, TM1_delta_variation,
    TM2_H_variation, TM3_k_variation, TM_Tp_variation,
    thermal_model_variation
)
from scripts.exploration.thermal.thermal_results import (
    thermal_results, plot_diffs_against_ref
)
from scripts.exploration.thermal.data_extractors import (
    extract_boundary_data
)

def main():
    save_dir = 'output/' + 'TM1_TM2_diff/'
    makedir(save_dir)

    #name_0_a = 'TM1_0'
    #save_dir_0_a = save_dir + name_0_a + '/'
    #TM1_0 = TM1(**thermal_inputs_0['TM1'])
    #thermal_results(save_dir=save_dir_0_a, plot=True)(TM1_0, None, name=name_0_a)

    #name_0_b = 'TM2_0'
    #save_dir_0_b = save_dir + name_0_b + '/'
    #TM2_0 = TM2(**thermal_inputs_0['TM2'])
    #thermal_results(save_dir=save_dir_0_b, plot=True)(TM2_0, None, name=name_0_b)


    #name_0_c = 'TM3_0'
    #save_dir_0_c = save_dir + name_0_c + '/'
    #TM3_0 = TM3(**thermal_inputs_0['TM3'])
    #thermal_results(save_dir=save_dir_0_c, plot=True)(TM3_0, None, name=name_0_c)

    #name_0_d = 'TM2_javi'
    #save_dir_0_d = save_dir + name_0_d + '/'
    #TM2_javi = TM2(**thermal_inputs_javi['TM2'])
    #thermal_results(save_dir=save_dir_0_d, plot=True)(TM2_javi, None, name=name_0_d)

    ###############################################################
    thermal_constants['Tp'] = 1350

    name_1 = 'TM1_base'
    save_dir_1 = save_dir + name_1 + '/'
    TM1_base = TM1(H0=3.0e-6, k=3.0, delta=10, constants=thermal_constants)
    #moho_temp_base = get_moho_temp_base(TM1_base)
    #shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM1_base)
    thermal_results(save_dir=save_dir_1, plot=True)(TM1_base, None, name=name_1)

    name_2 = 'TM1_ICD'
    save_dir_2 = save_dir + name_2 + '/'
    TM1_ICD = TM1(H0=3.0e-6, k=3.0, delta=None, constants=thermal_constants)
    thermal_results(save_dir=save_dir_2, plot=True)(TM1_ICD, None, name=name_2)

    name_3 = 'TM3_base'
    save_dir_3 = save_dir + name_3 + '/'
    TM3_base = TM3(
        Huc=1.75e-6, Hlc= 7.5e-7, kuc=3.0, klcm=2.25,
        constants=thermal_constants
    )
    thermal_results(save_dir=save_dir_3, plot=True)(TM3_base, None, name=name_3)

    ##########################################################

    #thermal_constants['Tp'] = 1350

    #name_1_b = 'TM1_base_b'
    #save_dir_1_b = save_dir + name_1_b + '/'
    #TM1_base_b = TM1(H0=3.0e-6, k=3.0, delta=15, constants=thermal_constants)
    ##moho_temp_base = get_moho_temp_base(TM1_base)
    ##shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM1_base)
    #thermal_results(save_dir=save_dir_1_b, plot=True)(TM1_base_b, None, name=name_1)

    #name_2_b = 'TM1_ICD_b'
    #save_dir_2_b = save_dir + name_2_b + '/'
    #TM1_ICD_b = TM1(H0=3.0e-6, k=3.0, delta=None, constants=thermal_constants)
    #thermal_results(save_dir=save_dir_2_b, plot=True)(TM1_ICD_b, None, name=name_2_b)

    #name_3_b = 'TM3_base_b'
    #save_dir_3_b = save_dir + name_3_b + '/'
    #TM3_base_b = TM3(
    #    Huc=1.75e-6, Hlc= 7.5e-7, kuc=3.0, klcm=2.25,
    #    constants=thermal_constants
    #)
    #thermal_results(save_dir=save_dir_3_b, plot=True)(TM3_base_b, None, name=name_3_b)

    ##########################################################


    #moho_temp_base = get_moho_temp_base(TM3_base)
    #shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM3_base)

main()
