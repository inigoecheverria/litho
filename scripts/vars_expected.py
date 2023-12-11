import importlib
import pandas as pd
import numpy as np
import xarray as xr
import shutil
from pathlib import Path
from litho.utils import makedir
import litho.equations as eqs
from inputs import (
    TM1_input,
    TM2_input,
    TM3_input,
    thermal_constants,
    thermal_conf,
    input_path
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

##### Importar modelo de geometrias

model_path = importlib.resources.files('litho').joinpath('data/Model.dat')

model_df = pd.read_csv(
        model_path,
        sep='\s+',
        #TODO: eliminate or use nil column (boolean for slab/lab)
        names=['lon','lat','Zb','Zm','Zi','Z0', 'nil'],
        index_col=[1,0])
model = model_df.to_xarray()


##### Agregar indicador de dominio al modelo (slab / lab)

areas_path = importlib.resources.files('litho').joinpath('data/Areas.dat')

model['D'] = xr.DataArray(
        np.loadtxt(areas_path)+1,
        dims=('lat','lon'),
        coords={
            'lat': model.coords['lat'][::-1],
            'lon': model.coords['lon']})


##### Agregar edades de la fosa al modelo

trench_age_path = importlib.resources.files('litho').joinpath('data/TrenchAge.dat')

model['age'] = xr.DataArray(
        np.loadtxt(trench_age_path)[:,1],
        dims=('lat'),
        coords={'lat':np.loadtxt(trench_age_path)[:,0]})


##### Importar datos de surface heat flow

shf_data_path = importlib.resources.files('litho').joinpath('data/shf_data.dat')

data_df = pd.read_csv(
        shf_data_path,
        sep='\s+',
        header=0,
        names=['Long', 'Lat', 'SHF', 'Error'],
        index_col=0)
data = data_df.to_xarray()


##### Interpolar variables del modelo en coordenadas de los datos

data['Zb'] = model['Zb'].interp(lon=data['Long'], lat=data['Lat'])
data['Zm'] = model['Zm'].interp(lon=data['Long'], lat=data['Lat'])
data['Zi'] = model['Zi'].interp(lon=data['Long'], lat=data['Lat'])
data['Z0'] = model['Z0'].interp(lon=data['Long'], lat=data['Lat'])
data['D'] = model['D'].interp(
        lon=data['Long'],
        lat=data['Lat'],
        method='nearest')
data['age'] = model['age'].interp(lat=data['Lat'])


##### Obtener profundidad de la interseccion slab/lab del modelo e interpolar
##### en coordenadas de los datos

sli_lon = model['D'].idxmax(dim='lon')
data['sli_Zb'] = model['Zb'].sel(lon=sli_lon).interp(lat=data['Lat'])
data['sli_Z0'] = model['Z0'].sel(lon=sli_lon).interp(lat=data['Lat'])


##### Dividir datos segun su dominio

slab_data = data.where(data['D']==1)
lab_data = data.where(data['D']==2)


##### Calcular la temperatura en el lab

lab_temp = eqs.calc_lab_temp(
    thermal_constants['Tp'],
    thermal_constants['G'],
    lab_data['Zb'],
    lab_data['Z0']
)


##### Calcular la temperatura en el slab (utilizando k y klcm)

sli_temp = eqs.calc_lab_temp(
    thermal_constants['Tp'],
    thermal_constants['G'],
    slab_data['sli_Zb'],
    slab_data['sli_Z0']
)

#def slab_temp(k):
#     return eqs.calc_slab_temp_from_sli_temp(
#        sli_temp,
#        slab_data['sli_Zb'],
#        slab_data['sli_Z0'],
#        slab_data['Zb'],
#        slab_data['Z0'],
#        slab_data['age'],
#        k,
#        thermal_constants['kappa'],
#        thermal_constants['alpha'],
#        thermal_constants['V'],
#        thermal_constants['Tp'],
#        thermal_constants['b'],
#    )
#
#slab_temp_M1 = slab_temp(TM1_input['k'])
#slab_temp_M2 = slab_temp(TM2_input['k'])
#slab_temp_M3 = slab_temp(TM3_input['klcm'])

# ks: unique thermal conductivty used for slab temp calculation
slab_temp = eqs.calc_slab_temp_from_sli_temp(
        sli_temp,
        slab_data['sli_Zb'],
        slab_data['sli_Z0'],
        slab_data['Zb'],
        slab_data['Z0'],
        slab_data['age'],
        thermal_constants['ks'],
        thermal_constants['kappa'],
        thermal_constants['alpha'],
        thermal_constants['V'],
        thermal_constants['Tp'],
        thermal_constants['b'],
)


##### Obtener temperatura basal combinando los dominios

#Tb = {
#    'M1': slab_temp_M1.combine_first(lab_temp),
#    'M2': slab_temp_M2.combine_first(lab_temp),
#    'M3': slab_temp_M3.combine_first(lab_temp)
#}

Tb = slab_temp.combine_first(lab_temp)


#####  Calcular expresiones en funcion de SHF para las distintas variables
#####  de los modelos analiticos

H0_M1 = eqs.calc_H0_M1(
        TM1_input['k'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M1'],
        Tb
        )

k_M1 = eqs.calc_k_M1(
        TM1_input['H0'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M1'],
        Tb
        )

Tp_M1 = eqs.calc_Tp_M1(
        TM1_input['k'],
        TM1_input['H0'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        thermal_constants['G']
        )

Huc_M2 = eqs.calc_Huc_M2(
        TM2_input['Hlc'],
        TM2_input['k'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M2']
        Tb
        )

Hlc_M2 = eqs.calc_Hlc_M2(
        TM2_input['Huc'],
        TM2_input['k'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M2']
        Tb
        )

k_M2 = eqs.calc_k_M2(
        TM2_input['Huc'],
        TM2_input['Hlc'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M2']
        Tb
        )

Tp_M2 = eqs.calc_Tp_M2(
        TM2_input['k'],
        TM2_input['Huc'],
        TM2_input['Hlc'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        thermal_constants['G']
        )

Huc_M3 = eqs.calc_Huc_M3(
        TM3_input['Hlc'],
        TM3_input['kuc'],
        TM3_input['klcm'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M3']
        Tb
        )

Hlc_M3 = eqs.calc_Hlc_M3(
        TM3_input['Huc'],
        TM3_input['kuc'],
        TM3_input['klcm'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M3']
        Tb
        )

kuc_M3 = eqs.calc_kuc_M3(
        TM3_input['Huc'],
        TM3_input['Hlc'],
        TM3_input['klcm'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M3']
        Tb
        )

klcm_M3 = eqs.calc_klcm_M3(
        TM3_input['Huc'],
        TM3_input['Hlc'],
        TM3_input['kuc'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        #Tb['M3']
        Tb
        )

Tp_M3 = eqs.calc_Tp_M3(
        TM3_input['kuc'],
        TM3_input['klcm'],
        TM3_input['Huc'],
        TM3_input['Hlc'],
        data['Z0'],
        data['Zi'],
        data['Zm'],
        data['Zb'],
        thermal_constants['G']
        )


##### Convertir xarray dataset a pandas dataframe (tabla de datos)

dataframe = data.to_dataframe()
dataframe = dataframe.drop(['lon', 'lat', 'sli_Zb', 'sli_Z0'], axis='columns')

##### Agregar valores de Tb para cada modelo a la tabla

#dataframe['Tb_M1'] = Tb['M1']
#dataframe['Tb_M2'] = Tb['M2']
#dataframe['Tb_M3'] = Tb['M3']

dataframe['Tb'] = Tb

##### Agregar valores maximos y minimos de variables a la tabla

Q0_min = data['SHF'] - data['Error']/100*data['SHF']
Q0_max = data['SHF'] + data['Error']/100*data['SHF']

dataframe['H0_M1'] = H0_M1(data['SHF']).data
dataframe['H0_M1_min'] = H0_M1(Q0_min).data
dataframe['H0_M1_max'] = H0_M1(Q0_max).data

dataframe['k_M1'] = k_M1(data['SHF']).data
dataframe['k_M1_min'] = k_M1(Q0_min).data
dataframe['k_M1_max'] = k_M1(Q0_max).data

dataframe['Tp_M1'] = Tp_M1(data['SHF']).where(data['D']==2).data
dataframe['Tp_M1_min'] = Tp_M1(Q0_min).where(data['D']==2).data
dataframe['Tp_M1_max'] = Tp_M1(Q0_max).where(data['D']==2).data

dataframe['Huc_M2'] = Huc_M2(data['SHF']).data
dataframe['Huc_M2_min'] = Huc_M2(Q0_min).data
dataframe['Huc_M2_max'] = Huc_M2(Q0_max).data

dataframe['Hlc_M2'] = Hlc_M2(data['SHF']).data
dataframe['Hlc_M2_min'] = Hlc_M2(Q0_min).data
dataframe['Hlc_M2_max'] = Hlc_M2(Q0_max).data

dataframe['k_M2'] = k_M2(data['SHF']).data
dataframe['k_M2_min'] = k_M2(Q0_min).data
dataframe['k_M2_max'] = k_M2(Q0_max).data

dataframe['Tp_M2'] = Tp_M2(data['SHF']).where(data['D']==2).data
dataframe['Tp_M2_min'] = Tp_M2(Q0_min).where(data['D']==2).data
dataframe['Tp_M2_max'] = Tp_M2(Q0_max).where(data['D']==2).data

dataframe['Huc_M3'] = Huc_M3(data['SHF']).data
dataframe['Huc_M3_min'] = Huc_M3(Q0_min).data
dataframe['Huc_M3_max'] = Huc_M3(Q0_max).data

dataframe['Hlc_M3'] = Hlc_M3(data['SHF']).data
dataframe['Hlc_M3_min'] = Hlc_M3(Q0_min).data
dataframe['Hlc_M3_max'] = Hlc_M3(Q0_max).data

dataframe['kuc_M3'] = kuc_M3(data['SHF']).data
dataframe['kuc_M3_min'] = kuc_M3(Q0_min).data
dataframe['kuc_M3_max'] = kuc_M3(Q0_max).data

dataframe['klcm_M3'] = klcm_M3(data['SHF']).data
dataframe['klcm_M3_min'] = klcm_M3(Q0_min).data
dataframe['klcm_M3_max'] = klcm_M3(Q0_max).data

dataframe['Tp_M3'] = Tp_M3(data['SHF']).where(data['D']==2).data
dataframe['Tp_M3_min'] = Tp_M3(Q0_min).where(data['D']==2).data
dataframe['Tp_M3_max'] = Tp_M3(Q0_max).where(data['D']==2).data

##### Agregar estadisticos a la tabla

dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

dataframe.loc['mean'] = dataframe.mean(skipna=True)
dataframe.loc['median'] = dataframe.median(skipna=True)
dataframe.loc['std'] = dataframe.std(skipna=True)


##### Imponer formato a las columnas de la tabla

formats = {
        'Lat': '{:8.3f}',
        'Long': '{:8.3f}',
        'SHF': '{:7.2f}',
        'Error': '{:5.1f}',
        'Zb': '{:10.4f}',
        'Zm': '{:10.4f}',
        'Zi': '{:10.4f}',
        'Z0': '{:10.4f}',
        'D': '{:3.0f}',
        'age': '{:7.2e}',
        #'Tb_M1': '{:7.1f}',
        #'Tb_M2': '{:7.1f}',
        #'Tb_M3': '{:7.1f}',
        'Tb': '{:7.1f}',
        'H0_M1': '{:10.2e}',
        'H0_M1_min': '{:+10.2e}',
        'H0_M1_max': '{:+10.2e}',
        'k_M1': '{:6.2f}',
        'k_M1_min': '{:+6.2f}',
        'k_M1_max': '{:+6.2f}',
        'Tp_M1': '{:7.1f}',
        'Tp_M1_min': '{:7.1f}',
        'Tp_M1_max': '{:7.1f}',
        'Huc_M2': '{:10.2e}',
        'Huc_M2_min': '{:+10.2e}',
        'Huc_M2_max': '{:+10.2e}',
        'Hlc_M2': '{:10.2e}',
        'Hlc_M2_min': '{:+10.2e}',
        'Hlc_M2_max': '{:+10.2e}',
        'k_M2': '{:6.2f}',
        'k_M2_min': '{:+6.2f}',
        'k_M2_max': '{:+6.2f}',
        'Tp_M2': '{:7.1f}',
        'Tp_M2_min': '{:7.1f}',
        'Tp_M2_max': '{:7.1f}',
        'Huc_M3': '{:10.2e}',
        'Huc_M3_min': '{:+10.2e}',
        'Huc_M3_max': '{:+10.2e}',
        'Hlc_M3': '{:10.2e}',
        'Hlc_M3_min': '{:+10.2e}',
        'Hlc_M3_max': '{:+10.2e}',
        'kuc_M3': '{:6.2f}',
        'kuc_M3_min': '{:+6.2f}',
        'kuc_M3_max': '{:+6.2f}',
        'klcm_M3': '{:6.2f}',
        'klcm_M3_min': '{:+6.2f}',
        'klcm_M3_max': '{:+6.2f}',
        'Tp_M3': '{:7.1f}',
        'Tp_M3_min': '{:7.1f}',
        'Tp_M3_max': '{:7.1f}',
        }

for col, f in formats.items():
    dataframe[col] = dataframe[col].map(lambda x: f.format(x))

##### Guardar tabla de datos completa

#save_dir = 'output/vars_base/'
#makedir(save_dir)
thermal_output_path = makedir(thermal_conf['output_path'])
save_dir = makedir(thermal_output_path + '/' + Path(__file__).stem + '/')
shutil.copy(input_path, save_dir)
dataframe.to_csv(save_dir + 'variables.csv', sep=',')


##### Guardar tablas de datos individuales para cada variable

#common_cols = ['Long','Lat','SHF','Error','Zb','Zm','Zi','Z0','D','age']
common_cols = ['Long','Lat','SHF','Error','Zb','Zm','Zi','Z0','D','age','Tb']

##### Para H0_M1:
dataframe[common_cols+[
    #'Tb_M1',
    'H0_M1',
    'H0_M1_min',
    'H0_M1_max']].to_csv(save_dir + 'variable_H0_M1.csv')

##### Para k_M1:
dataframe[common_cols+[
    #'Tb_M1',
    'k_M1',
    'k_M1_min',
    'k_M1_max']].to_csv(save_dir + 'variable_k_M1.csv')

##### Para Tp_M1:
dataframe[common_cols+[
    #'Tb_M1',
    'Tp_M1',
    'Tp_M1_min',
    'Tp_M1_max']].to_csv(save_dir + 'variable_Tp_M1.csv')

##### Para Huc_M2:
dataframe[common_cols+[
    #Tb_M2,
    'Huc_M2',
    'Huc_M2_min',
    'Huc_M2_max']].to_csv(save_dir + 'variable_Huc_M2.csv')

##### Para Hlc_M2:
dataframe[common_cols+[
    #Tb_M2,
    'Hlc_M2',
    'Hlc_M2_min',
    'Hlc_M2_max']].to_csv(save_dir + 'variable_Hlc_M2.csv')

##### Para k_M2:
dataframe[common_cols+[
    #'Tb_M2',
    'k_M2',
    'k_M2_min',
    'k_M2_max']].to_csv(save_dir + 'variable_k_M2.csv')

##### Para Tp_M2:
dataframe[common_cols+[
    #'Tb_M2',
    'Tp_M2',
    'Tp_M2_min',
    'Tp_M2_max']].to_csv(save_dir + 'variable_Tp_M2.csv')

##### Para Huc_M3:
dataframe[common_cols+[
    #'Tb_M3',
    'Huc_M3',
    'Huc_M3_min',
    'Huc_M3_max']].to_csv(save_dir + 'variable_Huc_M3.csv')

##### Para Hlc_M3:
dataframe[common_cols+[
    #'Tb_M3',
    'Hlc_M3',
    'Hlc_M3_min',
    'Hlc_M3_max']].to_csv(save_dir + 'variable_Hlc_M3.csv')

##### Para kuc_M3:
dataframe[common_cols+[
    #'Tb_M3',
    'kuc_M3',
    'kuc_M3_min',
    'kuc_M3_max']].to_csv(save_dir + 'variable_kuc_M3.csv')

##### Para klcm_M3:
dataframe[common_cols+[
    #'Tb_M3',
    'klcm_M3',
    'klcm_M3_min',
    'klcm_M3_max']].to_csv(save_dir + 'variable_klcm_M3.csv')

##### Para Tp_M3:
dataframe[common_cols+[
    #'Tb_M3',
    'Tp_M3',
    'Tp_M3_min',
    'Tp_M3_max']].to_csv(save_dir + 'variable_Tp_M3.csv')
