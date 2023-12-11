import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.io import netcdf
from litho.colormaps import eet_pg_07
from litho.utils import export_csv, import_csv
from litho.plots import base_map, heatmap_map

# Grilla EET Perez-Gussinye et al., 2007
res=[400, 600, 800]
res=res[1]

te_pg = netcdf.netcdf_file(f'Perez-Gussinye/Te_{res}.grd',
    mmap=False)

# Axes
x_axis = te_pg.variables['x'][:]
y_axis = te_pg.variables['y'][::-1]

# Grids
eet_grid = te_pg.variables['z'][::-1, :]
xx, yy = np.meshgrid(x_axis, y_axis)

# Interpolation
y_axis_model = np.linspace(-10, -45,
    num=int(round(abs(-45+10)/0.2+1)),
    endpoint=True)
x_axis_model = np.linspace(-80, -60,
    num=int(round(abs(-60+80)/0.2+1)),
    endpoint=True)
xx_model, yy_model = np.meshgrid(x_axis_model,y_axis_model)
eet_grid_masked = eet_grid.copy()
eet_grid[np.isnan(eet_grid)] = 1.e-10000000000000
eet_interpolator = RectBivariateSpline(x_axis, y_axis[::-1], eet_grid[::-1].T)
eet_interpolated = eet_interpolator(x_axis_model, y_axis_model[::-1])
eet_interpolated = eet_interpolated[:,::-1].T

def save_array():
    np.savetxt(f'Interpolados/Te_PG_{res}.txt', eet_interpolated.T)

def load_array():
    eet_interpolated = np.loadtxt(f'Interpolados/Te_PG_{res}.txt').T

###############################################################################

eet_grid_xa = xr.DataArray(
    eet_grid,
    dims=('lat', 'lon'),
    coords={
        'lon': x_axis,
        'lat': y_axis,
    }
)

#eet_interpolated_xa = eet_grid_xa.interp(lon=x_axis_model, lat=y_axis_model)

eet_interpolated_xa = xr.DataArray(
    eet_interpolated,
    dims=('lat', 'lon'),
    coords={
        'lon': x_axis_model,
        'lat': y_axis_model
    }
)

def save_df():
    export_csv(
        eet_interpolated_xa,
        f'Interpolados/Te_PG_{res}_df.csv',
        name='eet'
    )

def load_df():
    import_csv(f'Interpolados/Te_PG_{res}_df.txt')


if __name__ == '__main__':

    save_array()
    save_df()

    #Plot
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)

    # Original Map
    ax1 = base_map(
        extent=[min(x_axis), max(x_axis), min(y_axis), max(y_axis)],
        xticks=x_axis, yticks=y_axis, topo=False, gs=gs[0,0]
    )
    heatmap_map(
        eet_grid_xa, ax=ax1, cbar_limits=[0,100], colormap=eet_pg_07
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
        eet_interpolated_xa, ax=ax2, cbar_limits=[0,100], colormap=eet_pg_07
    )
    rect2 = Rectangle((-80,-45), 20, 35, facecolor='none', edgecolor='red')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.show()
    plt.close()
