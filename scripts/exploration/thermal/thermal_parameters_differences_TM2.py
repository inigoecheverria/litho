import functools
import numpy as np
from litho import Lithosphere
from litho.thermal import TM1, TM2, TM3
from scripts.inputs import thermal_inputs, thermal_constants
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
    extract_boundary_data, extract_thermal_data
)

def main():
    save_dir = 'output/' + 'thermal_parameters_differences/'
    makedir(save_dir)

    # Ragos extremos TM2
    # Huc = 1.5 - 2.0    ---->  1.75
    # Hlc = 0.5 - 1.0    ---->  0.75
    # k = 2 - 4          ---->  3
    # Tp = 1200 - 1500   ---->  1350

    thermal_constants['Tp'] = 1350
    TM2_base = TM2(
        Huc=1.75e-6, Hlc= 7.5e-7, k=3.0,
        constants=thermal_constants
    )
    moho_temp_base = get_moho_temp_base(TM2_base)
    shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM2_base)
    ###########################################################################

    cbar_limit_moho_temp = 75
    cbar_limit_shf = 15

    ###########################################################################
    # Huc
    save_dir_Huc = save_dir + 'TM2/Huc/'
    makedir(save_dir_Huc)

    print('looping over Huc variations')
    Huc_results = TM2_H_variation(
        thermal_results(
            save_dir = save_dir_Huc, plot=False
        ),
        TM=TM2_base, Huc_params=[1.5e-6, 2.0e-6]
    )
    Huc_moho_temps = {
        key: value['moho_temp'] for key, value in Huc_results.items()
    }
    Huc_shfs = {
        key: value['shf'] for key, value in Huc_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        Huc_moho_temps, moho_temp_base, save_dir=save_dir_Huc,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        Huc_shfs, shf_base, save_dir=save_dir_Huc,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

    ###########################################################################
    # Hlc
    save_dir_Hlc = save_dir + 'TM2/Hlc/'
    makedir(save_dir_Hlc)

    print('looping over Hlc variations')
    Hlc_results = TM2_H_variation(
        thermal_results(
            save_dir = save_dir_Hlc, plot=False
        ),
        TM=TM2_base, Hlc_params=[5e-7, 1e-6]
    )
    Hlc_moho_temps = {
        key: value['moho_temp'] for key, value in Hlc_results.items()
    }
    Hlc_shfs = {
        key: value['shf'] for key, value in Hlc_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        Hlc_moho_temps, moho_temp_base, save_dir=save_dir_Hlc,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        Hlc_shfs, shf_base, save_dir=save_dir_Hlc,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

    ###########################################################################
    # k
    save_dir_k = save_dir + 'TM2/k/'
    makedir(save_dir)

    print('looping over k variations')
    k_results = TM1_k_variation(
        thermal_results(
            save_dir = save_dir_k, plot=True
        ),
        TM=TM2_base, k_params=[2, 4]
    )
    k_moho_temps = {
        key: value['moho_temp'] for key, value in k_results.items()
    }
    k_shfs = {
        key: value['shf'] for key, value in k_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        k_moho_temps, moho_temp_base, save_dir=save_dir_k,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        k_shfs, shf_base, save_dir=save_dir_k,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

    ###########################################################################
    # Tp
    save_dir_Tp = save_dir + 'TM2/Tp/'
    makedir(save_dir)

    print('looping over Tp variations')
    Tp_results = TM_Tp_variation(
        thermal_results(
            save_dir = save_dir_Tp, plot=True
        ),
        TM=TM2_base, Tp_params=[1200, 1500]
    )
    Tp_moho_temps = {
        key: value['moho_temp'] for key, value in Tp_results.items()
    }
    Tp_shfs = {
        key: value['shf'] for key, value in Tp_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        Tp_moho_temps, moho_temp_base, save_dir=save_dir_Tp,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        Tp_shfs, shf_base, save_dir=save_dir_Tp,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

def get_moho_temp_base(TM):
    L = Lithosphere()
    L.set_thermal_state(TM)
    crust_base = extract_boundary_data(L)['crust_base']
    #boundaries = L.get_boundaries()
    #crust_base = np.maximum(boundaries['Zm'], boundaries['Zb'] + 1)
    moho_temp_base = L.get_geotherm({}).sel(depth=crust_base, method='nearest')
    return moho_temp_base

def get_surface_heat_flow_base(TM):
    L = Lithosphere()
    L.set_thermal_state(TM)
    shf = L.get_surface_heat_flow()
    estimators, shf_data = L.stats()
    return shf, estimators, shf_data

main()

