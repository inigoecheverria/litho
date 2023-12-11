def initial_thermal_vars(t_input):
    #k:3.5, h:2.2e-6
    t_input['k_cs'] = 3.0
    t_input['k_ci'] = 3.0
    t_input['k_ml'] = 3.0
    t_input['H_cs'] = 1.8e-6
    t_input['H_ci'] = 1.8e-6
    t_input['H_ml'] = 1.8e-6
    t_input['delta_icd'] = False
    t_input['t_lat'] = False
    t_input['delta'] = 10
    t_input['t'] = 39.13

def thermal_exploration(
        results_function, t_input=None, m_input=None, dir_name=None):
    if t_input is None and m_input is None:
        t_input, m_input = input_setup()
    if dir_name is not None:
        dir_name = dir_name + '/'
    else:
        dir_name = ''
    results = {}
    ###### Modelo Inicial (Modelo 1)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_1')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_' + '{:.1f}'.format(t_input['k_cs']) +
        #'__delta_' + '{:.1f}'.format(t_input['delta']) +
        #'__t_' + '{:.2f}'.format(t_input['t']))
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    ###### t Variable (Modelo 2)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_2')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_' + '{:.1f}'.format(t_input['k_cs']) +
        #'__delta_' + '{:.1f}'.format(t_input['delta']) +
        #'__t_var')
    t_input['t_lat'] = True
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    ###### K Variable (Modelo 4)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_4')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_var' +
        #'__delta_' + '{:.1f}'.format(t_input['delta']) +
        #'__t_' + '{:.2f}'.format(t_input['t']))
    t_input['k_cs'] = 3.0
    t_input['k_ci'] = 2.5
    t_input['k_ml'] = 3.5
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    ###### Delta Variable (Modelo 3)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_3')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_' + '{:.1f}'.format(t_input['k_cs']) +
        #'__delta_icd' +
        #'__t_' + '{:.2f}'.format(t_input['t']))
    t_input['delta_icd'] = True
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    ###### Delta y t variable (Modelo 5)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_5')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_' + '{:.1f}'.format(t_input['k_cs']) +
        #'__delta_icd' +
        #'__t_var')
    t_input['t_lat'] = True
    t_input['delta_icd'] =True
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    ###### Modelo mas complejo (Modelo 6)
    initial_thermal_vars(t_input)
    name = dir_name + (
        'model_6')
        ##'h_' + '{:.1f}'.format(t_input['H_cs']) +
        #'__k_var' +
        #'__delta_icd' +
        #'__t_var')
    t_input['t_lat'] = True
    t_input['k_cs'] = 3.0
    t_input['k_ci'] = 2.5
    t_input['k_ml'] = 3.5
    t_input['delta_icd'] = True
    #print(name)
    #print(t_input)
    results[name] = results_function(t_input, m_input, name)
    return results

def thermal_H_exploration(
        results_function, t_input=None, m_input=None, dir_name=None):
    if t_input is None and m_input is None:
        t_input, m_input = input_setup()
    if dir_name is not None:
        dir_name = dir_name + '/'
    else:
        dir_name = ''
    results = {}
    initial_thermal_vars(t_input)
    for h in np.linspace(0, 5.e-6, 6):
        t_input['H_cs'] = h 
        t_input['H_ci'] = h 
        t_input['H_ml'] = h 
        name = dir_name + 'h_{:.2E}'.format(h)
        results[name] = results_function(t_input, m_input, name)
    return results

def thermal_K_exploration(
        results_function, t_input=None, m_input=None, dir_name=None):
    if t_input is None and m_input is None:
        t_input, m_input = input_setup()
    if dir_name is not None:
        dir_name = dir_name + '/'
    else:
        dir_name = ''
    results = {}
    initial_thermal_vars(t_input)
    for k in np.linspace(1,5,5):
        t_input['k_cs'] = k
        t_input['k_ci'] = k
        t_input['k_ml'] = k
        name = dir_name + 'k_{:.2E}'.format(k)
        results[name] = results_function(t_input, m_input, name)
    return results

def thermal_delta_exploration(
        results_function, t_input=None, m_input=None, dir_name=None):
    if t_input is None and m_input is None:
        t_input, m_input = input_setup()
    if dir_name is not None:
        dir_name = dir_name + '/'
    else:
        dir_name = ''
    results = {}
    initial_thermal_vars(t_input)
    for delta in np.linspace(5,15,6):
        t_input['delta'] = delta 
        name = dir_name + 'delta_{:.0f}'.format(delta)
        results[name] = results_function(t_input, m_input, name)
    return results

def thermal_hot_and_cold_exploration(
        results_function, t_input=None, m_input=None, dir_name=None):
    if t_input is None and m_input is None:
        t_input, m_input = input_setup()
    if dir_name is not None:
        dir_name = dir_name + '/'
    else:
        dir_name = ''
    results = {}
    initial_thermal_vars(t_input)
    name = dir_name + 'normal'
    results[name] = results_function(t_input, m_input, name)
    name = dir_name + 'hot'
    t_input['delta'] = 15
    t_input['H_cs'] = 4.e-6
    t_input['H_ci'] = 4.e-6
    t_input['H_ml'] = 4.e-6
    #t_input['k_cs'] = 3.
    #t_input['k_ci'] = 3.
    #t_input['k_ml'] = 3.
    results[name] = results_function(t_input, m_input, name)
    name = dir_name + 'cold'
    t_input['delta'] = 5
    t_input['H_cs'] = 1.e-7
    t_input['H_ci'] = 1.e-7
    t_input['H_ml'] = 1.e-7
    #t_input['k_cs'] = 5.
    #t_input['k_ci'] = 5.
    #t_input['k_ml'] = 5.
    results[name] = results_function(t_input, m_input, name)
    return results
