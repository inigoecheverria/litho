import multiprocessing as mp
import xarray as xr
import numpy as np
from litho import Lithosphere

def get_model_data(TM, MM, data_extractors):
    dexts = [data_extractors] if not isinstance(
        data_extractors, list
    ) else data_extractors
    out_q = mp.Queue()
    def mp_termomecanico(TM, MM, data_extractor, queue):
        L = Lithosphere()
        L.set_thermal_state(TM)
        L.set_mechanic_state(MM)
        data = []
        for data_extractor in dexts:
            data.append(data_extractor(L))
        data = data[0] if len(data) == 1 else data
        queue.put(data)
        return
    proc = mp.Process(
        target=mp_termomecanico, args=(TM, MM, data_extractors, out_q))
    proc.start()
    data = out_q.get()
    proc.join()
    return data

#def get_model_data(TM, MM, data_extractor):
#    L = Lithosphere()
#    L.set_thermal_state(TM)
#    L.set_mechanic_state(MM)
#    data = data_extractor(L)
#    return data

def extract_boundary_data(L):
    boundaries = L.get_boundaries()
    #crust_base = boundaries['Zm'].combine_first(boundaries['Zb'])
    crust_base = np.maximum(boundaries['Zm'], boundaries['Zb'] + 1)
    return {
        **boundaries,
        'crust_base': crust_base
    }

def extract_thermal_data(L, input_bounds={}):
    estimators, shf_df = L.stats()
    return {
        'geotherm': L.get_geotherm(input_bounds=input_bounds),
        'surface_heat_flow': L.get_surface_heat_flow(),
        'estimators': estimators,
        'shf_df': shf_df
    }
