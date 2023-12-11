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

def main():
    save_dir = 'output/' + 'tef_vs_teq/'
    makedir(save_dir)

    eet_effective_dict = {
        'Te_Tassara': {
             'file': 'data/Te_invertido/Interpolados/Te_Tassara_df.csv',
             'dir': 'Tassara_07/',
             'colormap': eet_tassara_07},
        'Te_PG_400': {
             'file': 'data/Te_invertido/Interpolados/Te_PG_400_df.csv',
             'dir': 'Perez_Gussinye_07/400/',
             'colormap': eet_pg_07},
        'Te_PG_600': {
             'file': 'data/Te_invertido/Interpolados/Te_PG_600_df.csv',
             'dir': 'Perez_Gussinye_07/600/',
             'colormap': eet_pg_07},
        'Te_PG_800': {
             'file': 'data/Te_invertido/Interpolados/Te_PG_800_df.csv',
             'dir': 'Perez_Gussinye_07/800/',
             'colormap': eet_pg_07}}

    # Default params
    #uc_params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #lc_params = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    #lm_params = [23,24,25,26,27,28,29,30]
    #flm_params = [23]


    # EET eq. vs EET ef. / Mosaics #########################################

    ##########################################################################
    ## LC & UC with same rheology

    #save_dir = save_dir + 'crust/'
    #makedir(save_dir)

    ##crust_params = [
    ##    1, 2, 3, 4, 5,
    ##    6, 7, 8, 9, 10,
    ##    11, 12, 13, 14,
    ##    15, 16, 17, 18,
    ##    19, 20, 21, 22
    ##]

    #crust_params = [1, 3, 5, 7, 8, 11, 12, 13, 16, 17, 19, 22]

    #print('looping over rheo variations')
    #results = rheo_variation(
    #    Teq_vs_Tef_results(
    #        eet_effective_dict,
    #        save_dir = save_dir,
    #        plot=True
    #        #plot=False
    #    ),
    #    crust_params=crust_params
    #)
    #print('loop finished')
    #eets = [result['Teq'] for result in list(results.values())]
    #eets_diffs = [result['diffs'] for result in list(results.values())]
    #crust_rheos = [result['crust'] for result in list(results.values())]

    #print('detemining rheo mosaics')
    #rheo_mosaic(
    #    eet_effective_dict, eets, eets_diffs,
    #    crust_rheos=crust_rheos, uc_rheos=None, lc_rheos=None, lm_rheos=None,
    #    data_keys=['Te_Tassara', 'Te_PG_400', 'Te_PG_600', 'Te_PG_800'],
    #    data_name='eet', value_name='Teq',
    #    save_dir=save_dir + 'mosaic/',
    #    extended_plot=False,
    #    diff_map_func=diff_map_func,
    #)

    ##########################################################################
    # LC & UC with different rheology

    save_dir = save_dir + 'LC_UC_different/'
    makedir(save_dir)

    uc_params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    lc_params = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    print('looping over rheo variations')
    results = rheo_variation(
        functools.partial(
            rheo_variation,
            Teq_vs_Tef_results(
                eet_effective_dict, save_dir=save_dir,
                plot=False
            ),
            uc_params=uc_params
        ),
        lc_params=lc_params
    )
    print('loop finished')

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

main()
