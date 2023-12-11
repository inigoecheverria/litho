import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from litho.colormaps import eet_tassara_07
from litho.utils import export_csv, import_csv
from litho.plots import base_map, heatmap_map

# Grilla EET Tassara et al., 2007
te_tassara = np.loadtxt('Tassara/te.SA.xyz')
x_values = te_tassara[:,0]
y_values = te_tassara[:,1]
eet = te_tassara[:,2]

# Lons
lon_max= np.nanmax(x_values)
lon_min = np.nanmin(x_values)
lon_step = 1/12
# Lats
lat_max = np.nanmax(y_values)
lat_min = np.nanmin(y_values)
lat_step = 1/12

# Axes
x_axis = np.linspace(lon_min, lon_max,
    num=int(round(abs(lon_max-lon_min)/lon_step+1)),
    endpoint=True)
y_axis = np.linspace(lat_max, lat_min,
    num=int(round(abs(lat_min-lat_max)/lat_step+1)),
    endpoint=True)

# Grids
x_grid = x_values.reshape(len(y_axis), len(x_axis))
y_grid = y_values.reshape(len(y_axis), len(x_axis))
eet_grid = eet.reshape(len(y_axis), len(x_axis))
xx, yy = np.meshgrid(x_axis, y_axis)

# Interpolation
y_axis_model = np.linspace(-10, -45,
    num=int(round(abs(-45+10)/0.2+1)),
    endpoint=True)
x_axis_model = np.linspace(-80, -60,
    num=int(round(abs(-60+80)/0.2+1)),
    endpoint=True)

xx_model, yy_model = np.meshgrid(x_axis_model,y_axis_model)
eet_interpolator = RectBivariateSpline(x_axis, y_axis[::-1], eet_grid[::-1].T)
eet_interpolated = eet_interpolator(x_axis_model, y_axis_model[::-1])
eet_interpolated = eet_interpolated[:,::-1].T
#print(eet_interpolated.shape)

def save_array():
    np.savetxt('Interpolados/Te_Tassara.txt', eet_interpolated.T)

def load_array():
    eet_interpolated = np.loadtxt('Interpolados/Te_Tassara.txt').T

###############################################################################

eet_grid_xa = xr.DataArray(
    eet_grid,
    dims=('lat', 'lon'),
    coords={
        'lat': y_axis,
        'lon': x_axis,
    }
)

#eet_interpolated_xa = eet_grid_xa.interp(lon=x_axis_model, lat=y_axis_model)

eet_interpolated_xa = xr.DataArray(
    eet_interpolated,
    dims=('lat', 'lon'),
    coords={
        'lat': y_axis_model,
        'lon': x_axis_model,
    }
)

def save_df():
    export_csv(
        eet_interpolated_xa,
        'Interpolados/Te_Tassara_df.csv',
        name='eet'
    )

def load_df():
    import_csv('Interpolados/Te_Tassara_df.csv')

if __name__ == '__main__':

    save_array()
    save_df()

    # Plot
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)

    # Original Map
    ax1 = base_map(
        extent=[min(x_axis), max(x_axis), min(y_axis), max(y_axis)],
        xticks=x_axis[::10], yticks=y_axis[::10], topo=False, gs=gs[0,0]
    )
    heatmap_map(
        eet_grid_xa, ax=ax1, cbar_limits=[0,100], colormap=eet_tassara_07
    )
    rect1 = Rectangle((-80,-45), 20, 35, facecolor='none', edgecolor='red')
    ax1.add_patch(rect1)

    # Interpolated map
    ax2 = base_map(
        extent=[
            min(x_axis_model), max(x_axis_model),
            min(y_axis_model), max(y_axis_model)
        ],
        xticks=x_axis_model, yticks=y_axis_model, topo=False, gs=gs[0,1]
    )
    heatmap_map(
        eet_interpolated_xa, ax=ax2, cbar_limits=[0,100], colormap=eet_tassara_07
    )
    rect2 = Rectangle((-80,-45), 20, 35, facecolor='none', edgecolor='red')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()
    plt.close()
