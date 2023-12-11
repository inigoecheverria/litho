import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from litho.utils import MidPointNorm, round_to_1, get_magnitude, makedir_from_filename
from litho.colormaps import (
    jet_white_r, jet_white_r_2, get_diff_cmap, get_elevation_diff_cmap)
from litho.data import Data, data_setup
from cmcrameri import cm

import matplotlib as mpl
#mpl.rcParams['figure.figsize'] = [20., 10.]
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)

data = Data(*data_setup())


boundaries = xr.Dataset(
    {
        'Z0': (
            ('lon', 'lat'),
            data.topo,
            {'name': 'Topography level', 'units': 'kilometers'}
        ),
        'Zi': (
            ('lon', 'lat'),
            data.icd,
            {'name': 'ICD', 'units': 'kilometers'}
        ),
        'Zm': (
            ('lon', 'lat'),
            data.moho,
            {'name': 'Moho', 'units': 'kilometers'}
        ),
        'Zb': (
            ('lon', 'lat'),
            data.slab_lab,
            {'name': 'Lithosphere base', 'units': 'kilometers'}
        ),
        'lab_domain': (
            ('lat', 'lon'),
            data.lab_domain,
            {'name': 'LAB domain'}
        )
    },
    coords={
        'lon': data.lon_axis,
        'lat': data.lat_axis,
    },
)


def plot(
    darray, ax=None, figsize=None, vmin=None, vmax=None,
    cmap=None, trim=True, shf_data=None, diff=False, estimators=None,
    xlim=None, ylim=None, margin=True, boundaries=boundaries,
    boundaries_color=None, fill=True,
    topo=True, border=True, coastline=True, colored=False,
    filename=None, format='svg', #format='png',
    earthquakes=None, levels=None, fontsize=None, eq_size=None, eq_color=None,
    **kwargs
):
    if fontsize:
        mpl.rcParams.update({'font.size': fontsize})
    #TODO: do not repeat trim logic everywhere
    if cmap is None:
        if 'units' in darray.attrs:
            if darray.units == 'celsius':
                cmap = get_cmap('coolwarm')
            elif darray.units == 'milliwatts':
                cmap = get_cmap('afmhot')
            elif darray.units == 'kilometers':
                cmap = get_cmap('elevation_diff')
            elif darray.units == 'MPa':
                cmap = get_cmap('jet_white_r')
                vmin = 0
                vmax = 200
            elif darray.units == 'integrated MPa':
                cmap = get_cmap('viridis')
            elif darray.units == 'ET':
                cmap = get_cmap('jet_white_r_ET')
                vmin = 0
                vmax = 100
            #elif darray.units == 'Km. diff.':
            #    cmap = get_cmap('elevation_diff')
            else:
                cmap = get_cmap('viridis')
        else:
            cmap = get_cmap('viridis')
    if cmap == 'no':
        cmap = None
    fixed_coords = {}
    for coord_key, coord_val in darray.indexes.items():
        if coord_val.size == 1:
            fixed_coords[coord_key] = coord_val
    # Line plots (1D)
    if darray.squeeze().ndim == 1:
        # Depth plots
        #if 'depth' not in fixed_coords:
        #x_dim = 'depth'
        ##x_dim = [
        ##    k for k in ('lat', 'lon', 'depth') if k not in fixed_coords
        ##][0]
        x_dim = [
            k for k in darray.dims if k not in fixed_coords
        ][0]
        if ax is None:
            if figsize is None:
                figsize = (15, 5)
            if trim is True:
                xlim=(
                    darray.dropna(x_dim)[x_dim].min(),
                    darray.dropna(x_dim)[x_dim].max()
                )
            else:
                xlim=(darray[x_dim].min(), darray[x_dim].max())
            #if ylim is None:
            #    ylim = (0, None)
            if x_dim == 'depth':
                #if 'depth' in darray.coords and 'depth' not in fixed_coords:
                xlim = xlim[::-1]
                if boundaries is not None and len(fixed_coords) == 2:
                    boundaries = [
                        boundaries['Zb'].loc[fixed_coords],
                        boundaries['Zm'].loc[fixed_coords],
                        boundaries['Zi'].loc[fixed_coords],
                        boundaries['Z0'].loc[fixed_coords]
                    ]
                else:
                    boundaries = None
        #print('1D_profile')
        ax = depth_profile(
            darray, x_dim, ax=ax, figsize=figsize,
            xlim=xlim, ylim=ylim, boundaries=boundaries
        )
        return ax
        #
    # Pcolormesh plots
    if darray.squeeze().ndim == 2:
        #Cross section plots (2D)
        if 'lat' in fixed_coords or 'lon' in fixed_coords:
            x_dim = 'lon' if 'lat' in fixed_coords else 'lat'
            y_dim = 'depth'
            if ax is None:
                if figsize is None:
                    figsize = (15, 5)
                if trim is True:
                    lonlat_margin = 0
                    depth_margin = 0
                    if margin is True:
                        lonlat_margin = data.lonlat_step*2
                        depth_margin = data.depth_step*5
                        #lonlat_margin = 0.2*2
                        #depth_margin = 1*5
                    xlim=(
                        darray.dropna(x_dim, how='all')[x_dim].min()-lonlat_margin,
                        darray.dropna(x_dim, how='all')[x_dim].max()+lonlat_margin
                    )
                    ylim=(
                        darray.dropna(y_dim, how='all')[y_dim].min()-depth_margin,
                        darray.dropna(y_dim, how='all')[y_dim].max()+depth_margin
                    )
                else:
                    if xlim is None:
                        xlim=(darray[x_dim].min(), darray[x_dim].max())
                    if ylim is None:
                        ylim=(darray[y_dim].min(), darray[y_dim].max())
            if boundaries is not None:
                boundaries = [
                    boundaries['Zb'].loc[fixed_coords],
                    boundaries['Zm'].loc[fixed_coords],
                    boundaries['Zi'].loc[fixed_coords],
                    boundaries['Z0'].loc[fixed_coords]
                ]
            #print('cross_section')
            ax = cross_section(
                darray, x_dim, y_dim, ax=ax, figsize=figsize,
                xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax, cmap=cmap,
                boundaries=boundaries, boundaries_color=boundaries_color,
                fill=fill, earthquakes=earthquakes,
                eq_color=eq_color, eq_size=eq_size
            )
            if filename:
                makedir_from_filename(filename)
                filename = filename + '.' + format
                plt.tight_layout()
                plt.savefig(
                    filename, bbox_inches='tight', format=format)
                    #dpi=1200, transparent=True)
                    #format="eps", dpi=1200)
                    #format='svg')
                    #dpi=1200, format='pdf')
                    #dpi='figure', format='pdf')
                plt.close()
            return ax
        # Map plots (2D)
        if all (k not in fixed_coords for k in ('lat', 'lon')):
            if ax is None: # New addition careful!
                if figsize is None:
                    figsize = (8, 10)
                ax = base_map(
                    topo=topo,
                    figsize=figsize,
                    border=border,
                    coastline=coastline,
                    colored=colored
                )
            if vmin is not None or vmax is not None:
                cbar_limits = [vmax, vmin]
            else:
                cbar_limits = [np.nanmax(darray), np.nanmin(darray)]
            heatmap_map(
                darray, ax=ax, colormap=cmap, cbar_limits=cbar_limits,
                alpha=1, earthquakes=earthquakes, levels=levels, **kwargs
                )
            if shf_data is not None:
                if diff is False:
                    if 'units' in darray.attrs:
                        if darray.units == 'milliwatts':
                            cbar_limits = cbar_limits
                        else:
                            cbar_limits = None
                    data_scatter_map(
                        shf_data['SHF'], shf_data['Long'], shf_data['Lat'],
                        data_types=shf_data['Type'],
                        colormap=cmap, cbar_limits=cbar_limits, ax=ax
                    )
                else:
                    diff_scatter_map(
                        shf_data['SHF_diff'], shf_data['Long'], shf_data['Lat'],
                        data_types=shf_data['Type'], estimators=estimators,
                        weights=shf_data['SHF_weights'],
                        ax=ax)
            if filename:
                makedir_from_filename(filename)
                filename = filename + '.' + format
                plt.tight_layout()
                plt.savefig(
                    filename, bbox_inches='tight', format=format)
                    #dpi=1200, transparent=True)
                    #format="eps", dpi=1200)
                    #format='svg')
                    #dpi=1200, format='pdf')
                    #dpi='figure', format='pdf')
                plt.close()
            return ax

    if darray.squeeze().ndim == 3:
        print("can't plot 3D array")

def get_cmap(colormap):
    if colormap == 'elevation_diff':
        colormap = get_elevation_diff_cmap(100)
    if colormap == 'jet_white_r':
        colormap = jet_white_r_2
    if colormap == 'jet_white_r_ET':
        colormap = jet_white_r
    return colormap

def depth_profile(
    darray, x_dim,
    ax=None, figsize=(15,5),
    xlim=None, ylim=None,
    boundaries=None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    darray.plot(
        ax=ax, x=x_dim, #xincrease=False,
        xlim=xlim, ylim=ylim
    )
    if boundaries is not None:
        for boundary in boundaries:
            pass
            ax.axvline(boundary)
    return ax

def cross_section(
    darray, x_dim, y_dim,
    ax=None, figsize=(15, 5),
    xlim=None, ylim=None,
    vmin=None, vmax=None,
    cmap='viridis',
    boundaries=None,
    boundaries_color=None,
    fill=None,
    earthquakes=None,
    filename=None,
    eq_color=None,
    eq_size=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if fill is True:
        darray.plot(
            ax=ax,
            x=x_dim, y=y_dim,
            xlim=xlim, ylim=ylim,
            vmin=vmin, vmax=vmax, cmap=cmap,
            rasterized=True,
        )
    if boundaries is not None:
        for boundary in boundaries:
            boundary.plot(ax=ax, color=boundaries_color)
    if earthquakes is not None:
        s = 30 if eq_size is None else eq_size
        color = 'orange' if eq_color is None else eq_color
        lat_coords = darray.coords['lat']
        if lat_coords and len(lat_coords) == 1:
            eqs = earthquakes[
                earthquakes['latitude_bin'] == lat_coords.data[0]
            ]
            ax.scatter(
                eqs['longitude'], -eqs['depth'], color=color,
                s=s, edgecolors='black', linewidth=0.2,
                zorder=1000, rasterized=True
            )
            #eqs_big = eqs[eqs['mag'] > 8.5]
            #if len(eqs_big) > 0:
            #    plt.scatter(
            #        eqs_big['longitude'], -eqs_big['depth'], color='red',
            #        s=30, edgecolors='black', linewidth=0.2,
            #        zorder=1000
            #    )
    if filename:
        makedir_from_filename(filename)
        #filename = filename + '.png'
        #plt.savefig(filename, transparent=True)
        filename = filename + '.svg'
        plt.savefig(filename, transparent=True, format='svg')
        plt.close()
    return ax


def base_map(
        extent=None , xticks=None, yticks=None, linewidth=0.2,
        topo=False, border=True, coastline=True, colored=False,
        figsize=None, gs=None
    ):
    if gs is not None:
        ax = plt.subplot(gs, projection=ccrs.PlateCarree())
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
    if extent is None:
        ax.set_extent([-80, -60, -45, -10])
    if xticks is None and yticks is None:
        xticks = np.arange(-80, -60, 5.0)
        yticks = np.arange(-45, -10, 5.0)
    if figsize is None:
        figsize=(8,8)
    if topo is True:
        ax.stock_img()
    if colored is True:
        land = cfeature.NaturalEarthFeature('physical', 'land', '10m')
        ax.add_feature(
            land, facecolor='#4d4c4c', edgecolor='face',zorder=-1,
            rasterized=True
        )
        ocean = cfeature.NaturalEarthFeature('physical','ocean','10m')
        ax.add_feature(
            ocean, facecolor='#808080', edgecolor='face',zorder=-1,
            rasterized=True
        )
    if coastline is True:
        coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
        ax.add_feature(
            coastline, facecolor='None', edgecolor='k',
            rasterized=True
        )
    if border is True:
        border = cfeature.NaturalEarthFeature(
            'cultural',
            'admin_0_boundary_lines_land',
            '10m'
        )
        ax.add_feature(
            border, facecolor='None', edgecolor='gray', alpha=0.7,
            rasterized=True
        )
    gl = ax.gridlines(draw_labels=True, linewidth=linewidth, color='gray', alpha=0.8)
    #gl.top_label = False
    #gl.right_label = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    return ax

def get_map_scatter_function(map_lon, map_lat, data_types, map):
    markers = ['o']
    data_markers = np.array(markers * len(map_lon))
    m_scatter = {}
    if data_types is not None:
        _, indices = np.unique(data_types, return_inverse=True)
        markers = np.array(['s', 'o', '^', 'p'])
        labels = np.array(['Shallow Marine Probe', 'Deep Marine Drillhole',
                           'Deep Land Borehole', 'Geochemestry of Hotspring'])
        data_markers = markers[indices]
    # Map Scatter Function
    def map_scatter(scatter_data, data_weights=None, **kwargs):
        scatter_data = np.ma.masked_invalid(scatter_data)
        if data_weights is not None:
            #normalize weights
            data_weights = (
                (data_weights - np.min(data_weights))
                / (np.max(data_weights) - np.min(data_weights))
            )
            data_weights = np.ma.masked_invalid(data_weights)
        for m, l in zip(markers, labels):
            mask = data_markers == m
            m_scatter_data = scatter_data[mask]
            m_data_weights = data_weights[mask] if data_weights is not None else None
            m_lon, m_lat = map_lon[mask], map_lat[mask]
            scatter = map.scatter(
                m_lon, m_lat,
                c=m_scatter_data, marker=m, label=l, alpha=m_data_weights,
                **kwargs)
            if data_types is not None:
                m_scatter[m] = scatter
        return m_scatter, scatter
    return map_scatter


def heatmap_map(
        array_2D, colormap=None, cbar_limits=None, ax=None, alpha=1,
        filename=None, return_width_ratio=False,
        cbar_label=None, cbar_ticks=None, cbar_tick_labels=None,
        title=None, norm=None, earthquakes=None, levels=None):
    if ax is None:
        ax = base_map(topo=False)
    cbar_max, cbar_min = np.nanmax(array_2D), np.nanmin(array_2D)
    if cbar_limits is not None:
        cbar_max, cbar_min = cbar_limits[0], cbar_limits[1]
    kwargs = {}
    if colormap is not None:
        kwargs['cmap'] = colormap
    cbar_kwargs = {}
    if cbar_label is not None:
        cbar_kwargs['label'] = cbar_label
    if cbar_ticks is not None:
        cbar_kwargs['ticks'] = cbar_ticks
    if cbar_tick_labels is not None: # No way to set tick labels in cbar_kwargs
        add_colorbar = False
        cbar_dict = cbar_kwargs
        cbar_kwargs = None
    else:
        add_colorbar=True
    if levels is not None:
        im = array_2D.plot(
            ax=ax, x='lon', alpha=alpha, vmin=cbar_min, vmax=cbar_max, norm=norm,
            add_colorbar=add_colorbar, cbar_kwargs=cbar_kwargs, rasterized=True,
            levels=levels,
            **kwargs
        )
    else:
        im = array_2D.plot(
            ax=ax, x='lon', alpha=alpha, vmin=cbar_min, vmax=cbar_max, norm=norm,
            add_colorbar=add_colorbar, cbar_kwargs=cbar_kwargs, rasterized=True,
            **kwargs
        )
    if add_colorbar is False:
        #cb = plt.colorbar(im, rasterized=True, **cbar_dict)
        cb = plt.colorbar(im, **cbar_dict)
        if cbar_tick_labels is not None:
            cb.ax.set_yticklabels(cbar_tick_labels)
    if earthquakes is not None:
        ax.scatter(
            earthquakes['longitude'], earthquakes['latitude'], s=2.0,
            facecolor=earthquakes['color'],#, zorder=1000
            rasterized=True
        )
    if title is not None:
        ax.set_title(title)
    if return_width_ratio:
        width_ratio = 1 + 0.05 + 0.12
        return width_ratio
    if filename:
        makedir_from_filename(filename)
        #filename = filename + '.png'
        #plt.savefig(filename, transparent=True)
        filename = filename + '.svg'
        plt.savefig(filename, transparent=True, format='svg')
        plt.close()

def diff_map(
        a1, a2, diff, sd=None,
        a1_name=None, a2_name=None,
        colormap=jet_white_r, colormap_diff=get_elevation_diff_cmap(100),
        cbar_label=None, cbar_label_diff=None,
        cbar_limits=None, cbar_limits_diff=None,
        title_1=None, title_2=None, title_3=None,
        axs=None, filename=None, labelpad=-45, labelpad_diff=-45
    ):

    if axs is None:
        fig = plt.figure(figsize=(12,6))
        nc = 3
        gs = gridspec.GridSpec(1,nc) #gs = fig.add_gridspec(1,2)
        axs = [base_map(gs=gs[0,n]) for n in np.arange(nc)]
    if cbar_limits is None:
        cbar_max = max(np.nanmax(a1), np.nanmax(a2))
        cbar_min = min(np.nanmin(a1), np.nanmin(a2))
        cbar_limits = [cbar_min, cbar_max]
    if cbar_limits_diff is None:
        cbar_max_abs = max(np.nanmax(diff), abs(np.nanmin(diff)))
        cbar_limits_diff = [-cbar_max_abs, cbar_max_abs]
    if a1_name is None: a1_name = 'Array 1'
    if a2_name is None: a2_name = 'Array 2'
    if title_1 is None: title = a1_name
    if title_2 is None: title = a2_name
    if title_3 is None: title = f'{a1_name} - {a2_name} Diff.'
    wr1 = heatmap_map(
        a1, colormap=colormap, cbar_label=cbar_label,
        cbar_limits=cbar_limits, #labelpad=labelpad,
        title=title_1, ax=axs[0], return_width_ratio=True
    )
    wr2 = heatmap_map(
        a2, colormap=colormap, cbar_label=cbar_label,
        cbar_limits=cbar_limits, #labelpad=labelpad,
        title=title_2, ax=axs[1], return_width_ratio=True
    )
    axs[1].set_yticks([])
    sd_string = ' S.D.: {:.2f}'.format(sd) if sd is not None else ''
    title_3 = title_3 + sd_string if title_3 is not None else None
    wr3 = heatmap_map(
        diff, colormap=colormap_diff, cbar_label=cbar_label_diff,
        cbar_limits=cbar_limits_diff, #labelpad=labelpad_diff,
        title=title_3, ax=axs[2], return_width_ratio=True
    )
    axs[2].set_yticks([])
    plt.tight_layout()
    if filename:
        makedir_from_filename(filename)
        #filename = filename + '.png'
        #plt.savefig(filename, transparent=True)
        filename = filename + '.svg'
        plt.savefig(filename, transparent=True, format='svg')
        plt.close()


def data_scatter_map(
        data, lon, lat, cbar_limits=None, data_types=None,
        colormap=None,
        ax=None, cbar=True, legend=True,
        rmse=None, return_width_ratio=False,
        filename=None):
    # Scatter map
    cbar_max, cbar_min = np.nanmax(data), np.nanmin(data)
    if cbar_limits is not None:
        cbar_max, cbar_min = cbar_limits[0], cbar_limits[1]
    map_scatter = get_map_scatter_function(lon, lat, data_types, ax)
    m_scatter, scatter = map_scatter(
        data, cmap='afmhot',
        vmin=cbar_min, vmax=cbar_max, s=150,
        edgecolors='black', linewidths=0.5, rasterized=True)
    if colormap != 'afmhot':
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', '5%', '12%', axes_class=plt.Axes)
        cbar = plt.colorbar(scatter, cax=cbar_ax)# pad=0.2)
        cbar.set_label('Heat Flow [mW/m?]', rotation=90, labelpad=-45)
    if legend is True:
        fig = ax.get_figure()
        ax.legend(bbox_to_anchor=(0.5, 0.0), loc='upper center',
                           ncol=2, bbox_transform=fig.transFigure)

def diff_scatter_map(
        diff, lon, lat, data_types=None, ax=None,
        rmse=None, legend=True, return_width_ratio=False,
        filename=None,
        mse=None, sigmas=None, sd=None, estimators=None,
        weights=None):
    if estimators is not None:
        rmse=estimators['rmse']
        mse=estimators['mse']
        sigmas=estimators['sigmas']
        sd=estimators['sd']
    # Scatter map
    diff_max = np.nanmax(diff)
    diff_min = np.nanmin(diff)
    diff_limit = np.nanmax([abs(diff_max), abs(diff_min)])
    diff_limit = round_to_1(diff_limit, 'ceil')
    #diff_step = 10**get_magnitude(diff_limit)
    diff_step = 5
    divisions = np.arange(-80, 80+diff_step, diff_step)
    ticks = np.arange(-80, 80+diff_step, 10)
    #divisions = np.arange(-diff_limit, diff_limit+diff_step, diff_step)
    #ticks = np.arange(-diff_limit, diff_limit+diff_step, 2*diff_step)
    bins = len(divisions) - 1
    diff_cmap = get_diff_cmap(bins)
    norm = MidPointNorm(midpoint=0, vmin=-80, vmax=80)
    map_scatter = get_map_scatter_function(lon, lat, data_types, ax)
    m_scatter, scatter = map_scatter(
        diff, cmap=diff_cmap, norm=norm,
        edgecolors='k', linewidths=.3,
        data_weights=weights
    )
    # Colorbar
    cbar_pad = '15%'
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes('right', '5%', pad=cbar_pad, axes_class=plt.Axes)
    plt.sca(ax)
    cbar = plt.colorbar(scatter, cax=cbar_ax)# pad=0.2)
    #cbar.set_label('Diff [mW/m?]', rotation=90, labelpad=-60)
    cbar.set_ticks([])
    ## Colorbar histogram
    hist_ax = divider.append_axes('right', '30%', pad='0%', axes_class=plt.Axes)
    plt.sca(ax)
    N, bins, patches = hist_ax.hist(
        diff, bins=divisions,
        orientation='horizontal',
        weights=weights
    )
    hist_ax.set_yticks(ticks)
    #hist_ax.set_ylim([-diff_limit, diff_limit])
    hist_ax.set_ylim([-80, 80])
    hist_ax.yaxis.tick_right()
    if mse is not None:
        # MAE
        mse_text = plt.figtext(
            0.4, 0, f'MSE: {mse:0.2f}',
            fontweight='bold')
        #extra_artists.append(mse_text)
    if rmse is not None:
        # RMSE
        rmse_text = plt.figtext(
            0.4, -0.03, f'RMSE: {rmse:0.2f}',
            fontweight='bold')
        #extra_artists.append(rmse_text)
    if sd is not None:
        # RMSE
        rmse_text = plt.figtext(
            0.4, -0.06, f'SD: {sd:0.2f}',
            fontweight='bold')
        #extra_artists.append(rmse_text)
    if sigmas is not None:
        hist_ax.axhline(y=sigmas['n_1_sigma'])
        hist_ax.text(15,sigmas['n_1_sigma']-0.5*diff_step,r'-$\sigma$',size='small')
        hist_ax.axhline(y=sigmas['p_1_sigma'])
        hist_ax.text(15,sigmas['p_1_sigma']+0.2*diff_step,r'+$\sigma$',size='small')
        hist_ax.axhline(y=sigmas['n_2_sigma'])
        hist_ax.text(15,sigmas['n_2_sigma']-0.5*diff_step,r'-2$\sigma$',size='small')
        hist_ax.axhline(y=sigmas['p_2_sigma'])
        hist_ax.text(15,sigmas['p_2_sigma']+0.2*diff_step,r'+2$\sigma$',size='small')
    norm = Normalize(bins.min(), bins.max())
    for bin, patch in zip(bins, patches):
        color = diff_cmap(norm(bin))
        patch.set_facecolor(color)

def estimator_plot(
    vnames, vaxes, estimator, signed=False, label='', filename=None,
    cbar_limits=None
):
    # x axis
    x_name = vnames[0]
    x_axis = vaxes[0]
    plt.xlabel(''.join(x_name))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if len(vnames) > 1:
        # y axis
        y_name = vnames[1]
        y_axis = vaxes[1]
        plt.ylabel(''.join(y_name))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # 2D matrix
        vmin = np.floor(np.min(estimator))
        vmax = np.ceil(np.max(estimator))
        v = np.linspace(vmin, vmax, 100)
        #plt.contourf(
        #    x_axis, y_axis, rmses.T, v, norm=colors.PowerNorm(gamma=1./2.))
        #plt.pcolormesh(
        #    x_axis, y_axis, rmses.T, norm=colors.PowerNorm(gamma=1./2.))
        x_step = x_axis[1] - x_axis[0]
        y_step = y_axis[1] - y_axis[0]
        if signed is True:
            norm = None
            #vmax = np.nanmax([abs(vmax), abs(vmin)])
            #vmin = -vmax
            if cbar_limits is None:
                vmax = 60
                vmin = -60
            else:
                vmax = cbar_limits[0]
                vmin = cbar_limits[1]
            #cmap = get_diff_cmap(vmax * 2 + 1)
            cmap = 'bwr'
        else:
            #norm = colors.PowerNorm(gamma=1./2.)
            norm = None
            if cbar_limits is None:
                vmin = 20
                vmax = 60
            else:
                vmax = cbar_limits[0]
                vmin = cbar_limits[1]
            #cmap = 'viridis'
            #cmap = 'bone_r'
            cmap = cm.navia_r
        plt.imshow(
            estimator.T, origin='lower', aspect='auto',
            norm=norm, cmap=cmap, vmin=vmin, vmax=vmax,
            extent=[
                x_axis[0] - x_step/2, x_axis[-1] + x_step/2,
                y_axis[0] - y_step/2, y_axis[-1] + y_step/2])
        #xx, yy = np.meshgrid(x_axis, y_axis)
        #xx = xx.flatten()
        #yy = yy.flatten()
        #rmses_f = rmses.T.flatten()
        #plt.scatter(xx, yy, c=rmses_f, cmap='rainbow')
        plt.colorbar()
        name = filename + '_2D'
    else:
        index = np.arange(len(x_axis))
        plt.plot(x_axis, estimator, '-r', linewidth=1.)
        plt.plot(x_axis, estimator, 'or', linewidth=1.)
        #plt.bar(x_axis, estimator, alpha=.4, width=(x_axis[1]-x_axis[0])*0.8)
        diff = max(estimator) - min(estimator)
        plt.ylim(min(estimator)-0.2*diff, max(estimator)+0.2*diff)
        plt.ylabel(label)
        plt.grid(True)
        name = filename
    plt.tight_layout()
    if filename:
        makedir_from_filename(filename)
        #filename = filename + '.png'
        #plt.savefig(filename)#,dpi='figure', format='pdf')
        filename = filename + '.svg'
        plt.savefig(filename, format='svg')#,dpi='figure', format='pdf')
        plt.close()

def get_axes(nc):
    gs = gridspec.GridSpec(1,nc) #gs = fig.add_gridspec(1,2)
    axs = [base_map(gs=gs[0,n]) for n in np.arange(nc)]
    return axs

#def get_axes(nc=2,nr=1, sharex=None, sharey=None):
#    gs = gridspec.GridSpec(nr,nc) #gs = fig.add_gridspec(1,2)
#    ax0 = base_map(gs=gs[0,0])
#    if sharex == 'first':
#        sharex = ax0
#    if sharey == 'first':
#        sharey = ax0
#    axs = []
#    for r in len(nr):
#        for c in len(nc):
#            if r == 0 and c == 0:
#                axs.append(ax0)
#            else:
#                axs.append(base_map(gs=gs[r,n], sharex=sharex, sharey=sharey))
#    return axs
