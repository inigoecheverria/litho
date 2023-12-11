import numpy as np
from scripts.exploration.thermal.data_extractors import (
    get_model_data, extract_boundary_data, extract_thermal_data
)
from litho.utils import export_csv, import_csv
from litho.utils import makedir, makedir_from_filename, calc_deviation
from litho.plots import diff_map, plot, heatmap_map
#from litho.colormaps import categorical_cmap
#import matplotlib.colors as mcolors
lplot = plot

def thermal_results(save_dir='thermal_differences', plot=False):
    save_dir_maps = save_dir + 'maps/'
    save_dir_files = save_dir + 'files/'
    def results(TM, MM, name):
        extractors = [extract_boundary_data, extract_thermal_data]
        print(f'...extracting thermal data')
        boundary_data, thermal_data = get_model_data(TM, MM, extractors)
        moho_temp = thermal_data['geotherm'].sel(
            depth=boundary_data['crust_base'], method='nearest'
        )
        shf = thermal_data['surface_heat_flow']
        shf_data = thermal_data['shf_df']
        estimators = thermal_data['estimators']
        print(f'...saving thermal data')
        filename_thermal = save_dir_files + name
        makedir_from_filename(filename_thermal)
        export_csv(moho_temp, filename_thermal + '_moho_temp.csv', name='moho_temp')
        export_csv(shf, filename_thermal + '_shf.csv', name='shf')
        if plot is True:
            print(f'...ploting')
            heatmap_map(
                moho_temp,
                colormap='coolwarm',
                cbar_limits=[0, 1300],
                cbar_label='Temperatura [ºC]',
                title='Temperatura Moho',
                #labelpad=-48,
                filename=save_dir_maps + '_temp_moho_' + name
            )
            lplot(
                shf,
                shf_data=shf_data, estimators=estimators,
                diff=True,
                filename=save_dir_maps + '_shf_' + name,
                vmin=0, vmax=90
            )
            #heatmap_map(
            #    data['surface_heat_flow'], colormap='afmhot',
            #    cbar_label='Heat Flow [W/m²]', title='Surface Heat Flow',
            #    filename = save_dir_maps + name, cbar_limits=[0,100])
        return {
            'moho_temp': moho_temp,
            'shf': shf,
            'shf_df': shf_data,
            'estimators': estimators
        }
    return results

def plot_diffs_against_ref(
    values, ref, save_dir='thermal',
    shf_data=None, estimators=None, cbar_limit=None,
    title='Moho temperature', label='Temperature [ºC]',
    name='moho_temp_diff', diff_colormap='Spectral_r',
    ref_name='moho_temp_ref', ref_colormap='coolwarm', 
    cbar_ticks=None
):
    stds_0 = [np.nanstd(arr) for arr in list(values.values())]
    means_0 = [np.nanmean(arr) for arr in list(values.values())]
    print('stds_original', stds_0)
    print('means_original:', means_0)

    values_diffs = {
        key: value - ref for key, value in values.items()
    }

    stds = [np.nanstd(arr) for arr in list(values_diffs.values())]
    means = [np.nanmean(arr) for arr in list(values_diffs.values())]
    print('stds:', stds)
    print('means:', means)
    print(f'3 standard deviations: {3*max(stds):.2f}')

    #cbar_limit = 3*max(stds)
    if cbar_limit is None:
        cbar_limit = 50
    norm=None
    cmap='Spectral_r'

    print('min ref:', ref.max())
    print('max ref:', ref.min())
    print('ref std:', np.nanstd(ref))
    print('ref mean:', np.nanmean(ref))

    print('stds_original', stds_0)
    print('means_original:', means_0)

    if shf_data is not None and estimators is not None:
        print(shf_data)
        print(estimators)
        lplot(
            ref,
            shf_data=shf_data,
            estimators=estimators,
            diff=True,
            filename=save_dir + ref_name,
            vmin=0, vmax=100
        )
    else:
        heatmap_map(
            ref,
            colormap=ref_colormap,
            cbar_label=label,
            title=title + ' reference',
            #labelpad = -48,
            filename=save_dir + ref_name,
            cbar_limits=[1350,0]
        )
    
    for key, value_diff in values_diffs.items():
        print(f'max_diff: {value_diff.max().data:.2f}')
        print(f'min_diff: {value_diff.min().data:.2f}')
        heatmap_map(
            value_diff,
            colormap=cmap,
            cbar_limits=[cbar_limit, -cbar_limit],
            cbar_label=label,
            norm=norm,
            title=title + ' diff.',
            #labelpad = -48,
            cbar_ticks=cbar_ticks,
            filename=save_dir + name + str(key)
        )
