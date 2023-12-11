import importlib
from copy import deepcopy
from litho.thermal import TM1, TM2, TM3
from litho.mechanic import MM, rhe_data
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

def default_MM():
    return MM(**MM_input)

def parameter_variation_template(
        results_function, TM=None, MM=None, dir_name=None
):
    # Check arguments
    if TM is None: TM = default_TM()
    else: TM = deepcopy(TM)
    if MM is None: MM = default_MM()
    else: MM = deepcopy(MM)
    if dir_name is not None: dir_name = dir_name + '/'
    # Loop to modify variables
    results = {}
    # Store results for each iteration
    name = 'default_vars'
    results[name] = results_function(TM, MM, name)
    # Return the dictionary
    return results

def rheo_variation(
        results_function, TM=None, MM=None, dir_name=None,
        uc_params=None, lc_params=None, lm_params=None, flm_params=None,
        crust_params=None
):
    # Check arguments
    if TM is None: TM = default_TM()
    else: TM = deepcopy(TM)
    if MM is None: MM = default_MM()
    else: MM = deepcopy(MM)
    if dir_name is not None: dir_name = dir_name + '/'
    else: dir_name = ''
    if uc_params is None: uc_params = [MM.uc]
    if lc_params is None: lc_params = [MM.lc]
    if lm_params is None: lm_params = [MM.lm]
    if flm_params is None: flm_params = [MM.serp]
    if crust_params is not None:
        # Treat lc and uc as single layer
        lc_params = crust_params
        uc_params = [MM.uc]
    # Loop to modify variables
    results = {}
    i=0
    for flm_param in flm_params:
        MM.serp = flm_param
        if MM.serp is not None:
            flm_string = '__' + rhe_data[str(flm_param)]['name']
        else:
            flm_string = ''
        for lm_param in lm_params:
            MM.lm = lm_param
            for lc_param in lc_params:
                MM.lc = lc_param
                for uc_param in uc_params:
                    MM.uc = uc_param
                    uc_name = rhe_data[str(uc_param)]['name']
                    if crust_params is not None:
                        MM.uc = lc_param
                        uc_name = rhe_data[str(lc_param)]['name']

                    print(f'iteration number: {i}')
                    print(f'uc:{MM.uc}, lc:{MM.lc}, lm:{MM.lm}')
                    # Store results for each iteration
                    name = (dir_name +
                        uc_name + '__'
                        + rhe_data[str(lc_param)]['name'] + '__'
                        + rhe_data[str(lm_param)]['name'] + flm_string)
                    results[name] = results_function(TM, MM, name)
                    i+=1
                    #print(results)
    return results


def applied_stress_variation(
        results_function, TM=None, MM=None, dir_name=None):
    # Check arguments
    if TM is None: TM = default_TM()
    else: TM = deepcopy(TM)
    if MM is None: MM = default_MM()
    else: MM = deepcopy(MM)
    if dir_name is not None: dir_name = dir_name + '/'
    results = {}
    for s_max in np.linspace(100, 200, 5, endpoint=True):
        MM.s_max = s_max
        # Store results for each iteration
        name = dir_name + 's_max_{:.0f}'.format(s_max)
        results[name] = results_function(TM, MM, name)
    return results
