#import multiprocessing as mp
#import resource
import sys
import ast
#import os
import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import shutil
from itertools import product
from pathlib import Path
from litho import Lithosphere
from litho.thermal import TM0, TM1, TM2, TM3
from litho.plots import estimator_plot
from litho.utils import makedir
from inputs import (
    thermal_inputs,
    thermal_conf,
    input_path
)


# python vars.py var_name '[start, end, step]'

def TM(thermal_model, **kwargs):
    if thermal_model == 'TM1':
        return TM1(**kwargs)
    elif thermal_model == 'TM2':
        return TM2(**kwargs)
    elif thermal_model == 'TM3':
        return TM3(**kwargs)
    else:
        raise ValueError(
            "{} is not a valid thermal model.".format(thermal_model)
        )
    return

def parse_input():
    args = sys.argv[1:]
    couples = len(args) // 2
    vnames = []; vranges = [] #; vtuples = []
    #print("Input:")
    for couple in range(couples):
        vname = args[couple*2]
        vrange = ast.literal_eval(args[couple*2+1])
        #print("    ", vname, vrange)
        #vtuple = [vname] + vrange
        vnames.append(vname)
        vranges.append(vrange)
    vaxes = [get_var_axis(vranges[i]) for i in range(len(vranges))]
    mtotal = np.prod([len(vax) for vax in vaxes])
    mplot = mtotal
    print("Total number of models to run: {}".format(mtotal))
    ctrl_vnames = []; ctrl_vranges = []; ctrl_vaxes = []
    vitems = [vnames, vranges, vaxes]
    if len(vnames) > 2:
        ctrl_vnames, ctrl_vranges, ctrl_vaxes = [item[2:] for item in vitems]
        vnames, vranges, vaxes = [item[0:2] for item in vitems]
        mplot = np.prod([len(vax) for vax in vaxes])
        print("Total number of models per plot: {}".format(mplot))
        np.savetxt(save_dir + 'ctrl_vnames.txt', ctrl_vnames, fmt='%s')
        np.savetxt(save_dir + 'ctrl_vranges.txt', ctrl_vranges, fmt='%.2e')
    pt = int(mtotal/mplot)
    print(
        f'Making {pt} plot(s)'
        f' of {vnames[0] if len(vnames)<2 else f"{vnames[0]} vs {vnames[1]}"}'
    )
    np.savetxt(save_dir_files + 'vars_names.txt', vnames, fmt='%s')
    np.savetxt(save_dir_files + 'vars_ranges.txt', vranges, fmt='%.2e')
    return vnames, vranges, vaxes, ctrl_vnames, ctrl_vranges, ctrl_vaxes, pt

def get_var_name(name):
    var_name = [name]
    return var_name

def get_var_axis(range):
    if len(range) == 3:
        step = range[2]
        first_v = range[0]
        last_v = range[1]
        var_axis = np.linspace(
            first_v, last_v,
            num=int((abs(last_v-first_v))/step+1),
            endpoint=True
        )
    else:
        var_axis = range
    return var_axis

def get_model_estimators(t_input, m_input):
    L.set_thermal_state(
        TM(thermal_conf['default_model'],**t_input), age=None
    )
    estimators, _ = L.stats()
    return [estimators['rmse'], estimators['mse']]

def vars_estimators(t_input, m_input, var_names, var_ranges, var_type='thermal'):
    #t_input, m_input = input_setup()
    var_name_1 = get_var_name(var_names[0])
    var_axis_1 = get_var_axis(var_ranges[0])
    var_name_2 = None
    var_axis_2 = None
    if len(var_names) > 1:
        var_name_2 = get_var_name(var_names[1])
        var_axis_2 = get_var_axis(var_ranges[1])
    rmses = []
    mses = []
    i = 0
    for var_1 in var_axis_1:
        for vn in var_name_1:
            t_input[vn] = var_1
        if var_name_2 is None:
            rmses.extend(get_model_estimators(t_input, m_input))
            mses.append(rmses.pop())
        else:
            for var_2 in var_axis_2:
                for vn in var_name_2:
                    t_input[vn] = var_2
                #mem()
                i += 1
                #print(i)
                rmses.extend(get_model_estimators(t_input, m_input))
                mses.append(rmses.pop())
    if var_name_2 is not None:
        rmses = np.array(rmses).reshape(len(var_axis_1), len(var_axis_2))
        mses = np.array(mses).reshape(len(var_axis_1), len(var_axis_2))
    return {'rmses': rmses, 'mses': mses}

def get_results(t_input, m_input, vnames, vranges, vaxes, filename=''):
    estimators = vars_estimators(t_input, m_input, vnames, vranges)
    # Save
    np.savetxt(
        save_dir_files + 'vars_rmses' + filename + '.txt', estimators['rmses'])
    np.savetxt(
        save_dir_files + 'vars_mses' + filename + '.txt', estimators['mses'])
    estimator_plot(
        vnames, vaxes, estimators['rmses'], label='RMSE',
        filename=save_dir+'RMSE'+filename,
        #cbar_limits=[60,20]
        cbar_limits=[30,20]
    )
    estimator_plot(
        vnames, vaxes, estimators['mses'], signed=True, label='MSE',
        filename=save_dir+'MSE'+filename,
        #cbar_limits=[60,-60]
        cbar_limits=[30,-30]
    )
    return estimators['rmses'], estimators['mses']

def load_and_plot_results(vnames, vaxes, filename):
    rmses = np.loadtxt(save_dir_files + 'vars_rmses' + filename + '.txt')
    mses = np.loadtxt(save_dir_files + 'vars_mses' + filename + '.txt')
    estimator_plot(
        vnames, vaxes, rmses, label='RMSE',
        filename=save_dir+'RMSE'+filename)
    estimator_plot(
        vnames, vaxes, mses, signed=True, label='MSE',
        filename=save_dir+'MSE'+filename)
    #print("Use: python vars.py var_name '[start, end, step]'")
    #return vnames, vaxes, rmses, mses
    return rmses, mses

def load_and_plot_results_bak(save_dir, filename):
    save_dir_files = save_dir + 'Archivos/'
    makedir(save_dir_files)
    rmses = np.loadtxt(save_dir_files + 'vars_rmses' + filename + '.txt')
    mses = np.loadtxt(save_dir_files + 'vars_mses' + filename + '.txt')
    vranges = np.loadtxt(
        save_dir_files + 'vars_ranges' + filename + '.txt', ndmin=2)
    vnames = np.loadtxt(
        save_dir_files + 'vars_names' + filename + '.txt', dtype='str', ndmin=2)
    vaxes = [get_var_axis(vranges[i]) for i in range(len(vranges))]
    estimator_plot(
        vnames, vaxes, rmses, label='RMSE',
        filename=save_dir+'RMSE'+filename)
    estimator_plot(
        vnames, vaxes, mses, signed=True, label='MSE',
        filename=save_dir+'MSE'+filename)
    #print("Use: python vars.py var_name '[start, end, step]'")
    return vnames, vaxes, rmses, mses

def add_estimators_to_data_array(
        avnames, vaxes, ctrl_vars, rmses, mses, da_rmses, da_mses
    ):

    ctrl_vars = np.array(ctrl_vars).reshape(len(ctrl_vars),1).tolist()
    vaxes = vaxes + ctrl_vars

    ndim = len(avnames)
    padded_shape = (rmses.shape + (1,)*ndim)[:ndim]
    rmses = rmses.reshape(padded_shape)
    mses = mses.reshape(padded_shape)

    da_rmses = da_rmses.combine_first(
        xr.DataArray(
            rmses,
            coords=list(zip(avnames,vaxes)),
            name='RMSES'
        )
    )
    da_mses = da_mses.combine_first(
        xr.DataArray(
            mses,
            coords=list(zip(avnames,vaxes)),
            name='MSES'
        )
    )
    return da_rmses, da_mses

def make_parallel_coordinates_plot(df_rmses, df_mses, vnamess):
    fig = px.parallel_coordinates(
        df_rmses, color="RMSES",
        #dimensions=vnames,
        dimensions=vnamess+["RMSES"],
        color_continuous_scale='viridis',#px.colors.diverging.Tealrose,
        color_continuous_midpoint=None,
        #range_color=[20,60]
    )
    fig2 = px.parallel_coordinates(
        df_mses, color="MSES",
        #dimensions=vnames,
        dimensions=vnamess+["MSES"],
        color_continuous_scale='RdBu',#px.colors.diverging.Tealrose,
        color_continuous_midpoint=0,
        #range_color=[-60,60]
    )
    #fig.show()
    #fig2.show()
    fig.write_image(save_dir + 'pc_rmses.svg')
    fig2.write_image(save_dir + 'pc_mses.svg')


if __name__ == '__main__':
    L = Lithosphere()
    #TODO: implement better path logic with pathlib
    thermal_output_path = makedir(thermal_conf['output_path'])
    save_dir = makedir(thermal_output_path + '/' + Path(__file__).stem + '/')
    save_dir_files = makedir(save_dir + 'Archivos/')
    shutil.copy(input_path, save_dir)
    t_input = thermal_inputs[thermal_conf['default_model']]
    t_input = dict(**t_input.pop('constants'),**t_input)
    m_input = {}
    if len(sys.argv) > 2:
        # Create files and Plot
        inp = parse_input()
        vnames, vranges, vaxes, ctrl_vnames, ctrl_vranges, ctrl_vaxes, pt = inp
        avnames = vnames + ctrl_vnames
        if ctrl_vnames:
            pn = 1
            da_rmses = xr.DataArray(name='RMSES')
            da_mses = xr.DataArray(name='MSES')
            for ctrl_vars in product(*ctrl_vaxes):
                filename = ""
                ctrl_vars_str = ""
                for var_idx in range(len(ctrl_vars)):
                    ctrl_vname = ctrl_vnames[var_idx]
                    ctrl_var = ctrl_vars[var_idx]
                    t_input[ctrl_vname] = ctrl_var
                    previous = f"{ctrl_vars_str}{', ' if ctrl_vars_str else ''}"
                    ctrl_vars_str = f"{previous}{ctrl_vname}={ctrl_var}"
                    filename = (
                        f"{filename}_{ctrl_vname}_"
                        f"{ctrl_var:{'.2f' if .01<ctrl_var<99 else '.2e'}}"
                    )
                print(f"Plot {pn}/{pt} ({ctrl_vars_str})")
                rmses, mses = get_results(
                    t_input, m_input, vnames, vranges, vaxes, filename
                )
                print()
                da_rmses, da_mses = add_estimators_to_data_array(
                    avnames, vaxes, ctrl_vars,
                    rmses, mses, da_rmses, da_mses
                )
                pn += 1
            df_rmses = da_rmses.to_dataframe().reset_index()
            df_mses = da_mses.to_dataframe().reset_index()
            df_rmses.to_csv(save_dir + 'df_rmses.csv')
            df_mses.to_csv(save_dir + 'df_mses.csv')
            print("Making parallel coordinates plot...")
            make_parallel_coordinates_plot(df_rmses, df_mses, avnames)
        else:
            get_results(t_input, m_input, vnames, vranges, vaxes)
            print()

    else:
        # Load files and Plot
        vnames = np.loadtxt(
            save_dir_files+'vars_names.txt',dtype='str',ndmin=1
        ).tolist()
        vranges = np.loadtxt(
            save_dir_files+'vars_ranges.txt',ndmin=2
        ).tolist()
        vaxes = [get_var_axis(vranges[i]) for i in range(len(vranges))]
        ctrl = Path(save_dir + 'ctrl_vnames.txt')
        if ctrl.is_file():
            print(
                f"Found {len(list(Path(save_dir_files).rglob('_rmses_*')))}",
                "files to plot"
            )
            print("Plotting...")
            ctrl_vnames = np.loadtxt(
                save_dir+'ctrl_vnames.txt',dtype='str',ndmin=1
            ).tolist()
            ctrl_vranges = np.loadtxt(
                save_dir + 'ctrl_vranges.txt',ndmin=2
            ).tolist()
            ctrl_vaxes = [
                get_var_axis(ctrl_vranges[i]) for i in range(len(ctrl_vranges))
            ]
            avnames = vnames + ctrl_vnames
            for ctrl_vars in product(*ctrl_vaxes):
                filename = ''
                for var_idx in range(len(ctrl_vars)):
                    ctrl_vname = ctrl_vnames[var_idx]
                    ctrl_var = ctrl_vars[var_idx]
                    filename = (
                        f"{filename}_{ctrl_vname}_"
                        f"{ctrl_var:{'.2f' if .01<ctrl_var<99 else '.2e'}}"
                    )
                rmses, mses = load_and_plot_results(vnames, vaxes, filename)
            df_rmses = pd.read_csv(save_dir + 'df_rmses.csv')
            df_mses = pd.read_csv(save_dir + 'df_mses.csv')
            print("Making parallel coordinates plot...")
            make_parallel_coordinates_plot(df_rmses, df_mses, avnames)

        else:
            print("Found 1 file to plot")
            print("Plotting...")
            load_and_plot_results(vnames, vaxes, '')
