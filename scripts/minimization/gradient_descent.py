import numpy as np
from litho import Lithosphere
from litho.thermal import TM0, TM1, TM2, TM3
from inputs import thermal_inputs, thermal_conf, input_path

def gradient_descent(objective_function, initial_params, learning_rates, num_iterations, constraints=None, epsilons=None):
    """
    Perform gradient descent to optimize parameters.

    Args:
    objective_function (function): The objective function to minimize.
    initial_params (list or numpy.ndarray): Initial parameter values.
    learning_rates (list or numpy.ndarray): Learning rates for each parameter.
    num_iterations (int): The number of iterations.
    constraints (list of tuples, optional): Parameter constraints in the form of (min_value, max_value).

    Returns:
    optimized_params (numpy.ndarray): Optimal parameter values.
    min_value (float): Minimum value of the objective function.
    """

    params = np.array(initial_params, dtype=float)

    for _ in range(num_iterations):
        gradient = numerical_gradient(objective_function, params, epsilons)
        params -= learning_rates * gradient
        if constraints is not None:
            params = apply_constraints(params, constraints)
        print("\ngradient:", gradient)
        print("params:", params)
        print("\n")



    min_value = objective_function(params)

    return params, min_value

def numerical_gradient(objective_function, params, epsilons=None):
    """
    Compute the numerical gradient of the objective function with custom epsilon values for each parameter.

    Args:
    objective_function (function): The objective function to minimize.
    params (numpy.ndarray): Parameter values.
    epsilons (list, optional): List of epsilon values for each parameter.

    Returns:
    gradient (numpy.ndarray): Numerical gradient.
    """
    num_params = len(params)
    gradient = np.zeros(num_params)

    if epsilons is None:
        epsilons = [1e-6] * num_params  # Default epsilon for each parameter

    for i in range(num_params):
        perturbed_params = params.copy()
        perturbed_params[i] += epsilons[i]
        gradient[i] = (objective_function(perturbed_params) - objective_function(params)) / epsilons[i]

    return gradient

def apply_constraints(params, constraints):
    """
    Apply constraints to parameter values.

    Args:
    params (numpy.ndarray): Parameter values.
    constraints (list of tuples): Parameter constraints in the form of (min_value, max_value).

    Returns:
    params (numpy.ndarray): Parameter values with constraints applied.
    """
    for i, (min_value, max_value) in enumerate(constraints):
        if params[i] < min_value:
            params[i] = min_value
        elif params[i] > max_value:
            params[i] = max_value
    return params


def rmse_TM1_function(params):
    L = Lithosphere()
    #TM1_a = TM1(H0=params[0], delta=None, k=2.25)#, constants=tc)
    #TM1_a = TM1(H0=2.5e-6, delta=None, k=params[0])#, constants=tc)
    #TM1_a = TM1(H0=2.5e-6, delta=params[0], k=2.25)#, constants=tc)
    #
    #TM1_a = TM1(H0=params[0], delta=None, k=params[1])#, constants=tc)
    TM1_a = TM1(2.5e-6, delta=params[1], k=params[0])#, constants=tc)
    #
    #TM1_a = TM1(H0=params[0], delta=params[1], k=params[2])#, constants=tc)
    #
    #TM2_a = TM2(Huc=1.65e-6, Hlc=4e-7, k=2.0, constants=tc)
    #TM3_a = TM3(Huc=1.65e-6, Hlc=4e-7, kuc=3.0, klcm=1.0, constants=tc)
    L.set_thermal_state(TM1_a)
    estimators, df = L.stats()
    return estimators['rmse']

#initial_params = [3.e-6, 10., 2.]
#learning_rates = [5.e-13, 1e-2, 1e-2]
#epsilons = [1.e-10, 1e-3, 1e-4]
#num_iterations = 500
#constraints = None
##constraints = [(1.e-6, 5.e-6), (1, 30), (1, 5)]

# 1 parameter: H0 (T01)
#initial_params = [5.e-6]
#learning_rates = [5.e-13]
#epsilons = [1.e-10]
#num_iterations = 100
#constraints = None

# 1 parameter: k (T02)
#initial_params = [7]
#learning_rates = [5e-2]
#epsilons = [1.e-10]
#num_iterations = 100
#constraints = None

# 1 parameter: delta (T03)
#initial_params = [40]
#learning_rates = [1]
#epsilons = [1e-10]
#num_iterations = 200
#constraints = None

# 2 parameters: H & k (T04)
#initial_params = [5e-6, 7]
#learning_rates = [5e-13, 5e-2]
#epsilons = [1e-10, 1e-10]
#num_iterations = 500
#constraints = [(1e-7, 6e-6), (0.1,9)]

# 2 parameters: H & delta (T06)
#initial_params = [5e-6, 40]
#learning_rates = [5e-13, 1]
#epsilons = [1e-10, 1e-10]
#num_iterations = 500
#constraints = [(1e-7, 6e-6), (1,50)]

# 2 parameters: k & delta (T07)
#initial_params = [7, 40]
#learning_rates = [5e-2, 1]
#epsilons = [1e-10, 1e-10]
#num_iterations = 500
#constraints = [(0.1, 9), (1,50)]

# 3 parameters: H, k & delta (T05, T08, T09, T10, T11, T12, 
#          H.delta   H.k    k.delta
# Centers: T12,      T11,   T10
# Thirds:  T09,      T13,   T14
initial_params = [5e-7, 6, 40]
learning_rates = [5e-13, 5e-2, 1]
epsilons = [1e-10, 1e-10, 1e-10]
num_iterations = 500
constraints = [(1e-7, 5e-6), (0.1,7), (1, 50)]

optimized_params, min_value = gradient_descent(
    rmse_TM1_function, initial_params, learning_rates, num_iterations,
    constraints, epsilons
)
print("Optimal Parameters:", optimized_params)
print("Minimum Value:", min_value)

## Example usage:
#def example_objective_function(params):
#    return params[0] ** 2 + params[1] ** 2
#
#initial_params = [2.0, 3.0]
#learning_rate = 0.1
#num_iterations = 100
#constraints = [(0, 5), (0, 5)]
#
#optimized_params, min_value = gradient_descent(example_objective_function, initial_params, learning_rate, num_iterations, constraints)
#
#print("Optimal Parameters:", optimized_params)
#print("Minimum Value:", min_value)


#def gradient_descent(
#    gradient, start, learn_rate, n_iter=50, tolerance=1e-06
#):
#    vector = start
#    for _ in range(n_iter):
#        diff = -learn_rate * gradient(vector)
#        if np.all(np.abs(diff) <= tolerance):
#            break
#        vector += diff
#    return vector
