import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from litho.plots import heatmap_map, diff_map
from litho.utils import makedir, import_csv, export_csv, calc_deviation
from litho.mechanic import rhe_data
from litho.colormaps import (
    eet_tassara_07, eet_pg_07, jet_white_r, get_elevation_diff_cmap
)
from cmcrameri import cm

def rheo_mosaic(
        data_dict, values, values_datas_diffs, values_infos=None,
        crust_rheos=None, uc_rheos=None, lc_rheos=None, lm_rheos=None,
        data_keys=None, data_name='prop', value_name='value',
        extended_plot=True, save_dir='rheo_mosaics',
        diff_map_func=None
    ):

    # Check arguments
    if not any([crust_rheos, uc_rheos, lc_rheos, lm_rheos]):
    #if uc_rheos is None and lc_rheos is None and lm_rheos is None:
        raise ValueError("At least one list of rheologies must be provided.")
    if data_keys is None:
        data_keys = list(data_dict.keys())
    rheos = {}
    if crust_rheos is not None:
        crust_array = np.dstack(tuple(crust_rheos))
        rheos['crust'] = {'array': crust_array}
    if uc_rheos is not None:
        uc_array = np.dstack(tuple(uc_rheos))
        rheos['uc'] = {'array': uc_array}
    if lc_rheos is not None:
        lc_array = np.dstack(tuple(lc_rheos))
        rheos['lc'] = {'array': lc_array}
    if lm_rheos is not None:
        lm_array = np.dstack(tuple(lm_rheos))
        rheos['lm'] = {'array': lm_array}

    # Create directories
    save_dir_maps = save_dir + 'maps/'
    save_dir_files = save_dir + 'files/'
    makedir(save_dir_maps)
    makedir(save_dir_files)

    # Convert values to array
    values_array = np.dstack(tuple(values))
    if values_infos is not None: values_infos_array = np.dstack(tuple(values_infos))

    # Iterate over the datasets to compare against
    for data_key in data_keys:

        save_dir_files_i = save_dir_files + data_key
        save_dir_maps_i = save_dir_maps + data_key

        # Load data
        data = import_csv(
            data_dict[data_key]['file']
        )[data_name].reindex_like(values[0])

        # Save data
        export_csv(data, save_dir_files_i + '_df.csv', name=data_name)

        # Extract differences for this dataset
        values_data_diffs = [
            values_datas_diff[data_key]
            for values_datas_diff in values_datas_diffs
        ]

        # Convert the differences to an array
        values_data_diffs_array = np.dstack(tuple(values_data_diffs))
        values_data_diffs_array_m = np.ma.array(
            values_data_diffs_array, mask=np.isnan(values_data_diffs_array)
        )

        # Get indices of minimum difference
        k = np.nanargmin(abs(values_data_diffs_array_m), axis=2)
        m,n = k.shape
        i,j = np.ogrid[:m,:n]

        # Get the rheologies, values and differences at these indeces
        # (i.e the rheologies, values and diffs that minimize the difference)
        def get_fit(array):
            fit = array[i, j, k]
            fit = xr.DataArray(
                fit, dims=values[0].dims, coords=values[0].coords
            ).where(data.notnull())
            return fit
        # Rheo
        for rheo_key, rheo in zip(rheos.keys(), rheos.values()):
            rheo['fit'] = get_fit(rheo['array'])
            export_csv(
                rheo['fit'],
                save_dir_files_i + f'_{rheo_key}_df.csv',
                name='rheo'
            )
        # Value
        value_fit = get_fit(values_array)
        export_csv(
            value_fit,
            save_dir_files_i + '_value_fit_df.csv',
            name=value_name
        )
        # Diff
        value_data_diff_fit = get_fit(values_data_diffs_array) # Same as k
        export_csv(
            value_data_diff_fit,
            save_dir_files_i + '_value_fit_diff_df.csv',
            name='diff'
        )
        # Info
        if values_infos is not None:
            value_fit_info = get_fit(values_infos_array)
            export_csv(
                value_fit_info,
                save_dir_files_i + 'value_fit_info_df.csv',
                name='info'
            )
        ### Plot
        titles_rheo = [rheo_key + ' mosaic' for rheo_key in rheos.keys()]
        filenames_rheo = [
            save_dir_maps_i + '_' + rheo_key
            for rheo_key in rheos.keys()
        ]
        #filename_diff = save_dir_maps
        filename_diff = save_dir_maps_i + '_diff'
        if extended_plot is True:
            fig = plt.figure(figsize=(len(rheos)*4+12,6))
            gs = gridspec.GridSpec(1,len(rheos)+3)
            axs_rheo = [
                fig.add_subplot(
                    gs[0,n], projection=ccrs.PlateCarree()
                ) for n in range(len(rheos))
            ]
            axs_value_vs_data = [
                fig.add_subplot(
                    gs[0,len(rheos)+n], projection=ccrs.PlateCarree()
                ) for n in np.arange(3)
            ]
            filenames_rheo=[None]*len(rheos)
            filename_diff = None
        else:
            axs_rheo = [None]*len(rheos)
            axs_value_vs_data = None
            fig = None
        for i, rheo in enumerate(rheos.values()):
            u = np.unique(rheo['fit'].data[~np.isnan(rheo['fit'].data)])
            bounds = np.concatenate(
                ([min(u)-1], u[:-1]+np.diff(u)/2., [max(u)+1]))
            ncolors = len(bounds) - 1
            norm = mcolors.BoundaryNorm(bounds, ncolors)
            #cmap = plt.cm.get_cmap('viridis', ncolors)
            cmap = plt.get_cmap('viridis', ncolors)
            cbar_ticks = bounds[:-1]+np.diff(bounds)/2.
            cbar_tick_labels = [
                rhe_data[str(int(round(val)))]['name'] for val in u]
            heatmap_map(
                rheo['fit'], colormap=cmap, cbar_limits=[None, None], norm=norm,
                cbar_ticks=cbar_ticks, cbar_tick_labels=cbar_tick_labels,
                ax=axs_rheo[i], filename=filenames_rheo[i],
                title=titles_rheo[i])

        value_data_diff = value_fit - data
        export_csv(
            value_data_diff,
            save_dir_files + '_value_diff.csv',
            name='diff'
        )
        sd = calc_deviation(value_fit, data)

        if diff_map_func is not None:
            diff_map_func(
                value_fit, data, value_data_diff, sd=sd,
                axs=axs_value_vs_data, filename=filename_diff,
                #fig=fig
            )
        if extended_plot is True:
            plt.tight_layout()
            plt.show()
