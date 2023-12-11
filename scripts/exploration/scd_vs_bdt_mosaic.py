import functools
import numpy as np
from cmcrameri import cm
from litho.plots import diff_map
from litho.utils import makedir, makedir_from_filename, calc_deviation
from litho.utils import export_csv, import_csv
from scripts.exploration.parameter_variations import rheo_variation
from scripts.exploration.data_extractors import get_model_data, extract_bdt_data
from scripts.exploration.rheo_mosaic import rheo_mosaic

def main():
    # Create the output directory
    save_dir = 'output/' + 'bdt_vs_scd/'
    makedir(save_dir)

    scd_dict = {
        'nearest': {
             'file': 'data/SCD_nearest.txt',
             'dir': 'scd_nearest/'
        }
    }

    # BDT eq. vs SCD ef. / Mosaics #########################################

    #########################################################################
    # LC & UC with same rheology

    save_dir = save_dir + 'crust/'
    makedir(save_dir)

    #crust_params = [
    #    1, 2, 3, 4, 5,
    #    6, 7, 8, 9, 10,
    #    11, 12, 13, 14,
    #    15, 16, 17, 18,
    #    19, 20, 21, 22
    #]

    crust_params = [1, 3, 5, 7, 8, 11, 12, 13, 16, 17, 19, 22]

    print('looping over rheo variations')
    results = rheo_variation(
        bdt_scd_results(
            scd_dict,
            save_dir = save_dir,
            plot=True
            #plot=False
        ),
        crust_params=crust_params
    )

    print('loop finished')
    bdts = [result['bdt'] for result in list(results.values())]
    bdts_in_ucs = [result['bdt_in_uc'] for result in list(results.values())]
    bdts_scds_diffs = [
        result['bdt_scds_diffs'] for result in list(results.values())
    ]
    crust_rheos = [result['crust'] for result in list(results.values())]
    #########################################################################

    print('detemining rheo mosaics')
    rheo_mosaic(
        scd_dict, bdts, bdts_scds_diffs, bdts_in_ucs,
        uc_rheos=None, lc_rheos=lc_rheos, lm_rheos=None,
        data_keys=['nearest'],
        data_name='scd', value_name='bdt',
        save_dir=save_dir + 'mosaic/',
        extended_plot=False,
        diff_map_func=diff_map_func,
    )

    print('detemining rheo average')
    bdts_prom = (sum(bdts)/len(bdts))
    bdts_prom_filename = save_dir + 'files/bdt_prom.csv'
    makedir_from_filename(bdts_prom_filename)
    export_csv(bdts_prom, bdts_prom_filename, 'bdt')
    plot_bdt_vs_scd(
        scd_dict, bdts_prom,
        save_dir=save_dir + 'maps/', name='prom_diff'
    )

    print('done!')

diff_map_func = functools.partial(
    diff_map,
    #colormap=cm.grayC_r, colormap_diff=get_elevation_diff_cmap(100),
    a1_name='BDT', a2_name='SCD',
    colormap=cm.hawaii, colormap_diff=cm.cork,
    cbar_limits=[-100,10], cbar_limits_diff=[-100,100],
    cbar_label='Profundidad [km.]', cbar_label_diff='Dif. Profundidad [km.]',
    title_1='BDT',
    title_2='SCD',
    title_3=f'(BDT - SCD) Diff.',
    labelpad=-48, labelpad_diff=-56,
)

def plot_bdt_vs_scd(
        scd_dict, bdt, save_dir='SCD_BDT', name='bdt_scd_diff'
    ):
    for scdv in scd_dict.values():
        scd = import_csv(scdv['file'])['scd'].reindex_like(bdt)
        bdt_scd_diff = bdt - scd
        sd = calc_deviation(bdt, scd)
        diff_map_func(
            bdt, scd, bdt_scd_diff, sd=sd,
            filename=save_dir + scdv['dir'] + name
        )

def bdt_scd_results(scd_dict, save_dir='scd_bdt', plot=False):
    save_dir_maps = save_dir + 'maps/'
    save_dir_files = save_dir + 'files/'
    def results(TM, MM, name):
        bdt_scds_diffs = {}
        print(f'...extracting bdt data')
        data = get_model_data(TM, MM, extract_bdt_data)
        filename_bdt = save_dir_files + name
        makedir_from_filename(filename_bdt)
        export_csv(data['bdt'], filename_bdt + '_bdt.csv', name='bdt')
        export_csv(data['bdt_in_uc'], filename_bdt + '_bdt_in_uc.csv', name='uc')
        uc_array = np.ones(data['bdt'].data.shape) * MM.uc
        lc_array = np.ones(data['bdt'].data.shape) * MM.lc
        crust_array = uc_array if MM.uc == MM.lc else None
        for key, scdv in zip(
                scd_dict.keys(), scd_dict.values()
            ):
            print(f'...comparing against {key}')
            bdt = data['bdt'] #.reindex_like(scd)
            bdt_in_uc = data['bdt_in_uc']
            #bdt = data['bdt'].where(scd.notnull()) #TODO: needed?
            scd = import_csv(scdv['file'])['scd'].reindex_like(bdt)
            bdt_scd_diff = bdt - scd
            makedir_from_filename(save_dir_files + scdv['dir'] + name)
            export_csv(
                bdt_scd_diff,
                save_dir_files + scdv['dir'] + name + '_diff.csv',
                name='diff'
            )
            sd = calc_deviation(scd, bdt)
            if plot is True:
                print(f'...ploting')
                diff_map_func(
                    bdt, scd, bdt_scd_diff, sd=sd,
                    filename=save_dir_maps + scdv['dir'] + name + '_diff'
                )
            bdt_scds_diffs[key] = bdt_scd_diff
        return {
            'bdt_scds_diffs': bdt_scds_diffs,
            'bdt': bdt,
            'bdt_in_uc': bdt_in_uc,
            'uc': uc_array,
            'lc': lc_array,
            'crust': crust_array
        }
    return results

main()
