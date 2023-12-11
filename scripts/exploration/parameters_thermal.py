import functools
import numpy as np
from cmcrameri import cm
from litho.plots import diff_map
from litho.utils import makedir, makedir_from_filename, calc_deviation
from litho.utils import export_csv, import_csv
from litho.colormaps import (
    eet_tassara_07, eet_pg_07, jet_white_r, get_elevation_diff_cmap
)
from scripts.exploration.parameter_variations import rheo_variation
from scripts.exploration.data_extractors import get_model_data, extract_eet_data
from scripts.exploration.rheo_mosaic import rheo_mosaic

from litho.thermal import TM1, TM2, TM3
from litho.mechanic import MM
from scripts.inputs_lit import thermal_inputs, thermal_constants, mechanic_constants

from scripts.exploration.thermal.parameter_variations import (
    TM1_H0_variation, TM1_k_variation, TM1_delta_variation,
    TM2_H_variation, TM3_k_variation, TM_Tp_variation,
    thermal_model_variation
)

from litho import Lithosphere()

def main():
    save_dir = 'output/' + 'thermal_influence_in_mechanic/'
    makedir(save_dir)

    ##########################################################################

    save_dir = save_dir + '7_17_26/'
    makedir(save_dir)

    thermal_constants['Tp'] = 1350
    #TM1_base = TM1(H0=3.0e-6, k=3.0, delta=10, constants=thermal_constants)
    TM1_base = TM1(H0=3.0e-6, k=3.0, delta=None, constants=thermal_constants)
    MM_base = MM(uc=7,lc=17,lm=26,constants=mechanic_constants)
    eet_base = get_eet_base(TM1_base,MM_base)

    print('looping over H0 variations')
    H0_results = rheo_variation(
        mechanic_results(
            save_dir=save_dir, plot=False
        ),
        TM=TM1_base, MM=MM_base, H0_params=[2.5e-6, 3.5e-6]
    )
    H0_teqs = {
        key: value['teq'] for key, value in H0_results.items()
    }
    print('loop finished')


    print('comparing against base thermal model')


    H0_results = TM1_H0_variation(
        thermal_results(
            save_dir = save_dir_H0, plot=False
        ),
        TM=TM1_base, H0_params=[2.5e-6, 3.5e-6]
    )


    eets = [
        nested_result['Teq']
        for result in results.values()
        for nested_result in result.values()]
    eets_diffs = [
        nested_result['diffs']
        for result in results.values()
        for nested_result in result.values()]
    uc_rheos = [
        nested_result['uc']
        for result in results.values()
        for nested_result in result.values()]
    lc_rheos = [
        nested_result['lc']
        for result in results.values()
        for nested_result in result.values()]

    print('detemining rheo mosaics')
    rheo_mosaic(
        eet_effective_dict, eets, eets_diffs,
        uc_rheos=uc_rheos, lc_rheos=lc_rheos, lm_rheos=None, crust_rheos=None,
        data_keys=['Te_Tassara', 'Te_PG_400', 'Te_PG_600', 'Te_PG_800'],
        data_name='eet', value_name='Teq',
        save_dir=save_dir + 'mosaic/',
        extended_plot=False,
        diff_map_func=diff_map_func,
    )

    ########################################################################

    print('detemining rheo average')
    eets_prom = sum(eets)/len(eets)
    eets_prom_filename = save_dir + 'files/eet_prom.csv'
    makedir_from_filename(eets_prom_filename)
    export_csv(eets_prom, eets_prom_filename, 'eet')
    plot_eet_equivalent_vs_effective(
        eet_effective_dict, eets_prom,
        save_dir=save_dir + 'maps/', name='prom_diff'
    )

    print('done!')

diff_map_func = functools.partial(
    diff_map,
    colormap=jet_white_r, colormap_diff=get_elevation_diff_cmap(100),
    cbar_limits=[0,100], cbar_limits_diff=[-100,100],
    cbar_label='EET [km.]', cbar_label_diff='Dif. EET [km.]',
    title_1='Espesor Elástico Equivalente',
    title_2='Espesor Elástico Efectivo',
    title_3='Dif. (EET eq. - EET ef.)',
    labelpad=-48, labelpad_diff=-56,
)

def plot_eet_equivalent_vs_effective(
        eet_effective_dict, Teq, save_dir='EET', name='eet_diff'
    ):
    for eet_effective in eet_effective_dict.values():
        Tef = import_csv(eet_effective['file'])['eet'].reindex_like(Teq)
        eet_diff = Teq - Tef
        sd = calc_deviation(Teq, Tef)
        diff_map_func(
            Teq, Tef, eet_diff, sd=sd,
            filename=save_dir + eet_effective['dir'] + name
        )

def mechanic_results(save_dir='mechanic_results', plot=False):
    save_dir_maps = save_dir + 'maps/'
    save_dir_files = save_dir + 'files/'
    def results(TM, MM, name):
        data = get_model_data(TM, MM, extract_eet_data)
        filename_eet = save_dir_files + name
        makedir_from_filename(filename_eet)
        export_csv(data['eet'], filename_eet + '_Teq.csv', name='eet')
        if plot is True:
            heatmap_map(
                data['eet'],
                colormap=jet_white_r,
                cbar_limits=[100,0],
                cbar_label='Espesor Elástico [km]',
                title='Espesor Elástico Equivalente',
                filename=save_dir_maps + '_teq_' + name
            )
        return {
            'teq': data['eet']
        }
    return results

def Teq_vs_Tef_results(eet_effective_dict, save_dir='teq_vs_tef', plot=False):
    save_dir_maps = save_dir + 'maps/'
    save_dir_files = save_dir + 'files/'
    def results(TM, MM, name):
        diffs = {}
        print(f'...extracting Teq data')
        data = get_model_data(TM, MM, extract_eet_data)
        filename_eet = save_dir_files + name
        makedir_from_filename(filename_eet)
        export_csv(data['eet'], filename_eet + '_Teq.csv', name='eet')
        uc_array = np.ones(data['eet'].data.shape) * MM.uc
        lc_array = np.ones(data['eet'].data.shape) * MM.lc
        lm_array = np.ones(data['eet'].data.shape) * MM.lm
        crust_array = uc_array if MM.uc == MM.lc else None
        for key, eet_effective in zip(
                eet_effective_dict.keys(), eet_effective_dict.values()
            ):
            print(f'...comparing against {key}')
            Teq = data['eet']
            #Teq = Teq.where(Tef.notnull()) #TODO: needed?
            Tef = import_csv(eet_effective['file'])['eet'].reindex_like(Teq)
            eet_diff = Teq - Tef
            makedir_from_filename(save_dir_files + eet_effective['dir'] + name)
            export_csv(
                eet_diff,
                save_dir_files + eet_effective['dir'] + name + '_diff.csv',
                name='diff'
            )
            sd = calc_deviation(Teq, Tef)
            if plot is True:
                print(f'...ploting')
                diff_map_func(
                    Teq, Tef, eet_diff, sd=sd,
                    filename=save_dir_maps + eet_effective['dir'] + name + '_diff'
                )
            diffs[key] = eet_diff
        return {
            'diffs': diffs,
            'Teq': Teq,
            'uc': uc_array,
            'lc': lc_array,
            'lm': lm_array,
            'crust': crust_array
        }
    return results

def get_eet_base(TM, MM):
    L = Lithosphere()
    L.set_thermal_state(TM1_base)
    L.set_mechanic_state(MM_base)
    eet_base = L.get_eet()
    return eet_base

main()
