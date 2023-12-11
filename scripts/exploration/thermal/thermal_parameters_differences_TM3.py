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

    # Ragos extremos TM3
    # Huc = 1.5 - 2.0    ---->  1.75
    # Hlc = 0.5 - 1.0    ---->  0.75
    # kuc = 2 - 4        ---->  3
    # klcm = 2 - 2.5     ---->  2.25
    # Tp = 1200 - 1500   ---->  1350

    thermal_constants['Tp'] = 1350
    TM3_base = TM3(
        Huc=1.75e-6, Hlc= 7.5e-7, kuc=3.0, klcm=2.25,
        constants=thermal_constants
    )
    moho_temp_base = get_moho_temp_base(TM3_base)
    shf_base, estimators_base, shf_data_base = get_surface_heat_flow_base(TM3_base)
    ###########################################################################

    cbar_limit_moho_temp = 50
    cbar_limit_shf = 10

    ###########################################################################
    # Huc
    save_dir_Huc = save_dir + 'TM3/Huc/'
    makedir(save_dir_Huc)

    print('looping over Huc variations')
    Huc_results = TM2_H_variation(
        thermal_results(
            save_dir = save_dir_Huc, plot=False
        ),
        TM=TM3_base, Huc_params=[1.5e-6, 2.0e-6]
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
    save_dir_Hlc = save_dir + 'TM3/Hlc/'
    makedir(save_dir_Hlc)

    print('looping over Hlc variations')
    Hlc_results = TM2_H_variation(
        thermal_results(
            save_dir = save_dir_Hlc, plot=False
        ),
        TM=TM3_base, Hlc_params=[5e-7, 1e-6]
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
    # kuc
    save_dir_kuc = save_dir + 'TM3/kuc/'
    makedir(save_dir)

    print('looping over kuc variations')
    kuc_results = TM3_k_variation(
        thermal_results(
            save_dir = save_dir_kuc, plot=True
        ),
        TM=TM3_base, kuc_params=[2, 4]
    )
    kuc_moho_temps = {
        key: value['moho_temp'] for key, value in kuc_results.items()
    }
    kuc_shfs = {
        key: value['shf'] for key, value in kuc_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        kuc_moho_temps, moho_temp_base, save_dir=save_dir_kuc,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        kuc_shfs, shf_base, save_dir=save_dir_kuc,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################

    ###########################################################################
    # klcm
    save_dir_klcm = save_dir + 'TM3/klcm/'
    makedir(save_dir)

    print('looping over klcm variations')
    klcm_results = TM3_k_variation(
        thermal_results(
            save_dir = save_dir_klcm, plot=True
        ),
        TM=TM3_base, klcm_params=[2, 2.5]
    )
    klcm_moho_temps = {
        key: value['moho_temp'] for key, value in klcm_results.items()
    }
    klcm_shfs = {
        key: value['shf'] for key, value in klcm_results.items()
    }

    print('comparing against base thermal model')
    plot_diffs_against_ref(
        klcm_moho_temps, moho_temp_base, save_dir=save_dir_klcm,
        name='moho_temp_diff', diff_colormap='Spectral_r',
        ref_name='moho_temp_ref', ref_colormap='coolwarm',
        title='Moho temperature', label='Temperature [ºC]',
        cbar_limit=cbar_limit_moho_temp
    )
    plot_diffs_against_ref(
        klcm_shfs, shf_base, save_dir=save_dir_klcm,
        shf_data=shf_data_base, estimators=estimators_base,
        name='shf_diff', diff_colormap='Spectral_r',
        ref_name='shf_ref', ref_colormap='afmhot',
        title='Surface Heat Flow', label='Heat Flow [mW]',
        cbar_limit=cbar_limit_shf
    )
    ###########################################################################


    ###########################################################################
    # Tp
    save_dir_Tp = save_dir + 'TM3/Tp/'
    makedir(save_dir)

    print('looping over Tp variations')
    Tp_results = TM_Tp_variation(
        thermal_results(
            save_dir = save_dir_Tp, plot=True
        ),
        TM=TM3_base, Tp_params=[1200, 1500]
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

