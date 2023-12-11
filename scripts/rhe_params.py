import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

import xarray as xr
import numpy as np
from litho import Lithosphere
from litho.thermal import TM1, TM2, TM3
from litho.mechanic import rhe_data, MM
from scripts.inputs import thermal_inputs, thermal_conf, mechanic_constants, MM_input
from litho.colormaps import categorical_cmap
import pandas as pd

def read_rheo(filename):
    f = open(filename)
    dic = {}
    line = f.readline()
    while line:
        if line[0] != '#':
            id_rh, name, h, n, a, ref = line.split()
            dic[id_rh] = {
                'name': name,
                'n': float(n),
                'A': float(a),
                'H': float(h),
                'ref': ref
            }
        line = f.readline()
    f.close()
    return dic

#rhe_data = read_rheo('RheParams.dat')
rhe_data = read_rheo('RheParams_initial.dat')

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

def default_TM():
    return TM(
        thermal_conf['default_model'],
        **thermal_inputs[thermal_conf['default_model']]
    )

def main():
    # MODELO
    L = Lithosphere()
    L.set_thermal_state(default_TM())
    L.set_mechanic_state(MM(**MM_input))

    #lon_axis = L.state.coords['lon']
    depth_axis = L.get_depths_array({}).coords['depth'].values
    depth_axis_2 = [*depth_axis[::-1], *depth_axis]

    #lat = -30.0
    #lon = -62.0
    lat = -24.6
    lon = -67.4

    # SECCIONES 1D
    geotherm_1D = L.get_geotherm({'lat': lat, 'lon': lon})

    bys_t_1D, bys_c_1D = L.get_brittle_yield_strength_envelope(
        {'lat': lat, 'lon': lon}
    )

    bys_1D = np.concatenate(
        (bys_c_1D.squeeze().values[::-1], bys_t_1D.squeeze().values)
    )
    #df = pd.DataFrame()
    #df['bys_1D'] = bys_1D
    #df['depth_axis_2'] = depth_axis_2
    #print(df.to_string())

    #bys_1D = xr.concat(
    #    [bys_c_1D.squeeze(), bys_t_1D.squeeze()], dim='depth'
    #).squeeze().values

    topo_depth = L.get_topo().sel(lat=lat,lon=lon).values

    dys_list = []
    uc_params = []
    lc_params = []
    lm_params = []
    #uc_yield_temps_interp_dic = {}
    #lc_yield_temps_interp_dic = {}
    #lm_yield_temps_interp_dic = {}
    uc_yield_temps = {}
    lc_yield_temps = {}
    lm_yield_temps = {}

    yield_temps_1 = []
    yield_depths_1 = []
    for key, value in rhe_data.items():
        print(key)
        #if key in set(['1','14','30']):
        #if key in set(map(str,np.arange(2,30,2))):
        if key:
            #dys = L.get_ductile_yield_strength_envelope(
            #    {'lat': lat, 'lon': lon}
            #).squeeze().values

            dys = MM.calc_ductile_yield_strength(
                None,
                mechanic_constants['e'],
                value['n'], value['A'], value['H'], 
                mechanic_constants['R'],
                geotherm_1D.squeeze().values 
            )
            dys_list.append({'name': value['name'], 'value': dys, 'id': key})
            yield_temp = MM.calc_ductile_yield_temperature(
                None,
                mechanic_constants['e'],
                value['n'], value['A'], value['H'], 
                mechanic_constants['R'],
                mechanic_constants['s_max']
            )
            yield_depth = np.interp(
                yield_temp,
                geotherm_1D.squeeze().values,
                geotherm_1D.coords['depth'].values
            )
            yield_depths_1.append(yield_depth)
            yield_temps_1.append(yield_temp)

            if 0 <= int(key) < 11:
                uc_params.append(key)
                uc_yield_temps[value['name']] = yield_temp
            elif 11 <= int(key) < 23:
                lc_params.append(key)
                lc_yield_temps[value['name']] = yield_temp
            else:
                lm_params.append(key)
                lm_yield_temps[value['name']] = yield_temp

    fig = plt.figure(figsize=(12,7)) #12,7
    min_z = geotherm_1D.idxmax(dim='depth').squeeze().values
    print(min_z)
    max_z = 5

    major_z_ticks = np.arange(0, min_z+1, -25)
    minor_z_ticks = np.arange(5, min_z+1, -5)

    gs = gridspec.GridSpec(1,3)
    ax = fig.add_subplot(gs[0,1:])
    ax.set_xlim(-1000,1000)
    ax.set_ylim(min_z,max_z)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_yticks(major_z_ticks)
    ax.set_yticks(minor_z_ticks, minor=True)
    ax.set_title('YSEs')
    ax.plot(bys_1D, depth_axis_2, 'k')
    colors = categorical_cmap(3, [len(uc_params),len(lc_params),len(lm_params)],
        desaturated_first=True)(np.linspace(0,1,len(dys_list)))
    colors_iterator = iter(colors)
    for i, dys in enumerate(dys_list):
        #print(dys)
        color = next(colors_iterator)
        dys_1D = [*-dys['value'][::-1], *dys['value']]
        #print(dys_1D)
        #print(depth_axis_2)
        #print(dys['name'])
        #ax.plot(dys_1D, depth_axis_2, color=color, label=dys['name'])
        ax.plot(dys_1D, depth_axis_2, color=color, label=dys['id'])
    ax.axvline(x=-200, color='r', linestyle='dashed')
    ax.axhline(y=topo_depth, color='k')
    ax.axvline(x=0, color='k')
    
    ax2 = fig.add_subplot(gs[0,0])
    ax2.set_xlim(0,1400)
    #ax2.set_ylim(-180,20)
    ax2.set_ylim(min_z,max_z)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.set_yticks(major_z_ticks)
    ax2.set_yticks(minor_z_ticks, minor=True)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=False)
    #ax2.set_yticks([])
    ax2.set_title('Temperatura')
    ax2.plot(geotherm_1D.squeeze().values, depth_axis)
    ax2.axhline(y=topo_depth, color='k')
    ax2.axvline(x=0, color='k')
    plot_intersections = True
    if plot_intersections is True:
        #colors = np.vstack((np.array([[0, 0, 0, 1]]), colors))
        colors = colors
        colors_iterator = iter(colors)
        for yield_depth, yield_temp in zip(yield_depths_1, yield_temps_1):
            color = next(colors_iterator)
            #ax.plot([-2000,-200], [yield_depth, yield_depth], 
            #    color=color, linestyle='dashed')
            #ax2.plot([yield_temp, 1300], [yield_depth, yield_depth], 
            #    color=color, linestyle='dashed')
            ax2.plot([yield_temp, yield_temp], [yield_depth, -200], 
                color=color, linestyle='dashed')

    #legend = ax.legend(loc=2, bbox_to_anchor=(1.05, 1.00))
    suptitle = fig.suptitle('Lat: {}, Lon: {}'.format(lat,lon))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.tight_layout(pad=7)
    extra_artists = [suptitle]
    #extra_artists = [suptitle, legend]
    plt.savefig('rhe_params.svg', bbox_extra_artists=extra_artists,
        bbox_inches='tight', transparent=True)

    df_temps = pd.DataFrame()
    df_temps['rheo_id'] = list(rhe_data.keys())
    df_temps['rheo_name'] = [value['name'] for value in rhe_data.values()]
    df_temps['yield_temp'] = yield_temps_1
    df_temps['yield_depth'] = yield_depths_1
    df_temps.to_csv('rhe_params.csv', sep=' ', na_rep='nan', float_format='%.2f')

main()
