import importlib
from copy import deepcopy
from litho.thermal import TM1, TM2, TM3
from litho.mechanic import MM, rhe_data
#from litho.utils import var_range
from scripts.inputs_javi import (
    thermal_conf,
    thermal_inputs,
    MM_input,
    input_path
)

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

def thermal_model_variation(
    results_function, TM=None, MM=None, dir_name=None
):
    # Check arguments
    if TM is None: TM = default_TM()
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    TM1_a = TM1(**thermal_inputs['TM1'])
    TM2_a = TM2(**thermal_inputs['TM2'])
    TM3_a = TM3(**thermal_inputs['TM3'])
    results = {}
    print(type(results_function))
    results['TM1'] = results_function(TM1_a, MM, 'TM1')
    results['TM2'] = results_function(TM2_a, MM, 'TM2')
    results['TM3'] = results_function(TM3_a, MM, 'TM3')
    # Return the dictionary
    return results

def TM_Tp_variation(
        results_function, TM=None, MM=None, dir_name=None,
        Tp_params=None,
):
    # Check arguments
    if TM is None: TM = default_TM()
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if k_params is None: kuc_params = var_range(1.5, 4.5, 1)
    if Tp_params is None: Tp_params = [TM.Tp]
    # Loop to modify variables
    results = {}
    i=0
    print(Tp_params)
    for Tp in Tp_params:
        TM.Tp = Tp
        print(f'iteration number: {i}')
        print(f'Tp:{TM.Tp}')
        # Store results for each iteration
        name = dir_name + f'k_{TM.Tp:.0f}'
        results[name] = results_function(TM, MM, name)
        i+=1
    return results

###########################################################################
# TM1

def TM1_H0_variation(
        results_function, TM=None, MM=None, dir_name=None,
        H0_params=None, icd_scaling_factor=None
):
    # Check arguments
    if TM is None: TM = TM1(thermal_inputs['TM1'])
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if H0_params is None: Huc_params = var_range(2.5e-6, 3e-6, 2e-7)
    #if icd_scale_factor is None: icd_scaling_factor = var_range(0.6, 1.4, 0.2)
    if H0_params is None: H0_params = [TM.H0]
    # Loop to modify variables
    results = {}
    i=0
    for H0 in H0_params:
        TM.H0 = H0
        print(f'iteration number: {i}')
        print(f'H0:{TM.H0}')
        # Store results for each iteration
        name = dir_name + f'H0_{H0:.2E}'
        results[name] = results_function(TM, MM, name)
        i+=1
    return results

def TM1_k_variation(
        results_function, TM=None, MM=None, dir_name=None,
        k_params=None,
):
    # Check arguments
    if TM is None: TM = TM1(thermal_inputs['TM1'])
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if k_params is None: kuc_params = var_range(1.5, 4.5, 1)
    if k_params is None: k_params = [TM.k]
    # Loop to modify variables
    results = {}
    i=0
    print(k_params)
    for k in k_params:
        TM.k = k
        print(f'iteration number: {i}')
        print(f'k:{TM.k}')
        # Store results for each iteration
        name = dir_name + f'k_{TM.k:.2f}'
        results[name] = results_function(TM, MM, name)
        i+=1
    return results

def TM1_delta_variation(
        results_function, TM=None, MM=None, dir_name=None,
        delta_params=None,
):
    # Check arguments
    if TM is None: TM = TM1(thermal_inputs['TM1'])
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if delta_params is None: delta_params = var_range(5, 20, 5)
    if delta_params is None: delta_params = [TM.delta]
    # Loop to modify variables
    results = {}
    i=0
    for delta in delta_params:
        TM.delta = delta 
        print(f'iteration number: {i}')
        print(f'delta:{TM.delta}')
        # Store results for each iteration
        name = dir_name + f'delta_{delta:.0f}'
        results[name] = results_function(TM, MM, name)
        i+=1
    return results

###############################################################################
# TM2

def TM2_H_variation(
        results_function, TM=None, MM=None, dir_name=None,
        Huc_params=None, Hlc_params=None
):
    # Check arguments
    if TM is None: TM = TM2(thermal_inputs['TM2'])
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if Huc_params is None: Huc_params = var_range(1.5e-6, 2e-6, 1e-7)
    #if Hlc_params is None: Hlc_params = var_range(5e-7, 1e-6, 1e-7)
    if Huc_params is None: Huc_params = [TM.Huc]
    if Hlc_params is None: Hlc_params = [TM.Hlc]
    # Loop to modify variables
    results = {}
    i=0
    for Huc in Huc_params:
        TM.Huc = Huc
        for Hlc in Hlc_params:
            TM.Hlc = Hlc
            print(f'iteration number: {i}')
            print(f'Huc:{TM.Huc}, Hlc:{TM.Hlc}')
            # Store results for each iteration
            name = dir_name + f'Huc_{Huc:.2E}_Hlc_{Hlc:.2E}'
            results[name] = results_function(TM, MM, name)
            i+=1
    return results

###############################################################################
# TM3

def TM3_k_variation(
        results_function, TM=None, MM=None, dir_name=None,
        k_params=None, kuc_params=None, klcm_params=None
):
    # Check arguments
    if TM is None: TM = TM3(thermal_inputs['TM3'])
    else: TM = deepcopy(TM)
    #if MM is None: MM = default_MM()
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    #if kuc_params is None: kuc_params = var_range(2.0, 4.0, 1)
    #if klcm_params is None: klcm_params = var_range(2.0, 2.5, 0.1)
    if kuc_params is None: kuc_params = [TM.kuc]
    if klcm_params is None: klcm_params = [TM.klcm]
    if k_params is not None:
        # Treat lc and uc as single layer
        klcm_params = crust_params
        kuc_params = [TM.klcm]
    # Loop to modify variables
    results = {}
    i=0
    for klcm in klcm_params:
        TM.klcm = klcm
        klcm_name = f'klcm_{TM.klcm:.2f}'
        for kuc in kuc_params:
            TM.kuc = kuc
            kuc_name = f'kuc_{TM.kuc:.2f}'
            print(f'iteration number: {i}')
            print(f'kuc:{TM.kuc}, klcm:{TM.klcm}')
            if k_params is not None:
                TM.kuc = klcm
                kuc_name = klcm_name
            # Store results for each iteration
            name = dir_name + f'{dir_name}_{kuc_name}_{klcm_name}'
            results[name] = results_function(TM, MM, name)
            i+=1
    return results
