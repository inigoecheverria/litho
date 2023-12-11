import multiprocessing as mp
import xarray as xr
from litho import Lithosphere

def get_model_data(TM, MM, data_extractor):
    out_q = mp.Queue()
    def mp_termomecanico(TM, MM, data_extractor, queue):
        L = Lithosphere()
        L.set_thermal_state(TM)
        L.set_mechanic_state(MM)
        data = data_extractor(L)
        queue.put(data)
        return
    proc = mp.Process(
        target=mp_termomecanico, args=(TM, MM, data_extractor, out_q))
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

def extract_bdt_data(L):
    #print('get_icd_started')
    yse = L.get_yield_strength_envelope({})
    #print('get_icd_finished')
    yse_rs = yse.shift(depth=1)
    yse_ls = yse.shift(depth=-1)
    ysemax = yse.max(dim='depth')
    c = (((yse > yse_rs) & (yse > yse_ls)) | (yse == ysemax))
    yse_i = c.argmax(dim='depth')  #índices
    v = yse.where(c)
    L.set_depths_array({})
    bdt_absolute = L.space['Z'][:, :, yse_i]
    bdt_absolute = bdt_absolute.where(bdt_absolute !=5)
    topo = L.get_topo()
    bdt = bdt_absolute-topo
    icd_relative = (L.get_icd() - L.get_topo())
    bdt_icd_diff = bdt - icd_relative
    bdt_in_uc = xr.ones_like(bdt_icd_diff).where(bdt_icd_diff > 0, other=0)
    return {'bdt': bdt, 'bdt_in_uc': bdt_in_uc}

def extract_eet_data(L):
    return {'eet': L.get_eet({})}


# Example usage: get_model_data(TM, MM, extract_bdt_data)
