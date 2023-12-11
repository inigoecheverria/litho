import functools
import numpy as np
from litho import Lithosphere
from litho.thermal import TM1, TM2, TM3
from scripts.inputs_lit import thermal_inputs, thermal_constants
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
    save_dir = 'output/' + 'thermal_parameters_differences/'
    makedir(save_dir)

    # Ragos extremos TM1
    # H0 = 2.5 - 3.5    ---->  3.0
    # k = 2 - 4         ---->  3
    # delta = 10 - 15    ---->  15
    # Tp = 1200 - 1500  ---->  1350

    thermal_constants['Tp'] = 1350
    #TM1_base = TM1(H0=3.0e-6, k=3.0, delta=10, constants=thermal_constants)
    TM1_base = TM1(H0=3.0e-6, k=3.0, delta=None, constants=thermal_constants)
    moho_temp_base = get_moho_temp_base(TM1_base)
    shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM1_base)
    ###########################################################################

    cbar_limit_moho_temp = 75
    cbar_ticks_limit_moho_temp = np.arange(-75,75+15,15)
    cbar_limit_shf = 15

    ###########################################################################
    # H0
    save_dir_H0 = save_dir + 'TM1/H0/'
    makedir(save_dir_H0)

    print('looping over H0 variations')
    H0_results = TM1_H0_variation(
        thermal_results(
            save_dir = save_dir_H0, plot=False
        ),
        TM=TM1_base, H0_params=[2.5e-6, 3.5e-6]
    )
    H0_moho_temps = {
        key: value['moho_temp'] for key, value in H0_results.items()
    }
    H0_shfs = {
        key: value['shf'] for key, value in H0_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        H0_moho_temps, moho_temp_base, save_dir=save_dir_H0,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp,
        cbar_ticks=cbar_ticks_limit_moho_temp
    )
    plot_diffs_against_ref(
        H0_shfs, shf_base, save_dir=save_dir_H0,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

    ###########################################################################
    # k
    save_dir_k = save_dir + 'TM1/k/'
    makedir(save_dir)

    print('looping over k variations')
    k_results = TM1_k_variation(
        thermal_results(
            save_dir = save_dir_k, plot=True
        ),
        TM=TM1_base, k_params=[2, 4]
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
        cbar_limit=cbar_limit_moho_temp,
        cbar_ticks=cbar_ticks_limit_moho_temp
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

    ############################################################################
    ## delta
    #save_dir_delta = save_dir + 'TM1/delta/'
    #makedir(save_dir)

    #print('looping over delta variations')
    #delta_results = TM1_delta_variation(
    #    thermal_results(
    #        save_dir = save_dir_delta, plot=True
    #    ),
    #    TM=TM1_base, delta_params=[5, 15]
    #)
    #delta_moho_temps = {
    #    key: value['moho_temp'] for key, value in delta_results.items()
    #}
    #delta_shfs = {
    #    key: value['shf'] for key, value in delta_results.items()
    #}

    #print('comparing against base thermal model')
    #plot_diffs_against_ref(
    #    delta_moho_temps, moho_temp_base, save_dir=save_dir_delta,
    #    name='moho_temp_diff', diff_colormap='Spectral_r',
    #    ref_name='moho_temp_ref', ref_colormap='coolwarm',
    #    title='Moho temperature', label='Temperature [ºC]',
    #    cbar_limit=cbar_limit_moho_temp
    #)
    #plot_diffs_against_ref(
    #    delta_shfs, shf_base, save_dir=save_dir_delta,
    #    shf_data=shf_data_base, estimators=estimators_base,
    #    name='shf_diff', diff_colormap='Spectral_r',
    #    ref_name='shf_ref', ref_colormap='afmhot',
    #    title='Surface Heat Flow', label='Heat Flow [mW]',
    #    cbar_limit=cbar_limit_shf
    #)
    ############################################################################


    ###########################################################################
    # Tp
    save_dir_Tp = save_dir + 'TM1/Tp/'
    makedir(save_dir)

    print('looping over Tp variations')
    Tp_results = TM_Tp_variation(
        thermal_results(
            save_dir = save_dir_Tp, plot=True
        ),
        TM=TM1_base, Tp_params=[1200, 1500]
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
        cbar_limit=cbar_limit_moho_temp,
        cbar_ticks=cbar_ticks_limit_moho_temp
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

