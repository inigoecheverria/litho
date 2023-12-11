import numpy as np
import pandas as pd

weigh_errors = True

def evaluate_model(
    df, weigh_errors=weigh_errors, return_dataframe=False, save_dir=None
):
    data = df['SHF']
    model = df['SHF_model']
    p_errors = df['Error'] # percentages
    diff = model - data
    if weigh_errors is True:
        errors = p_errors/100*data

        # min error to avoid the overrepresentation of values with very small errors
        # min_error equal to 1sd left of mean
        min_error = errors.mean() - errors.std()
        errors = errors.clip(lower=min_error)

        mean_error = errors.mean()
        #errors = errors.fillna(mean_error) # No values without errors

        weights = mean_error / errors
        #weights = 1 / errors**2 # inverse-variance weighting

        #n_weights = weights / np.sum(weights)
        # Weighted estimators
        mse = calc_mse_weighted(diff, weights)
        sd = calc_sd_weighted(diff, weights, mse=mse)
        sigmas = calc_sigmas_weighted(mse=mse, sd=sd)
        msqe = calc_msqe_weighted(diff, weights)
        rmse = calc_rmse_weighted(msqe=msqe)
    else:
        # Estimators
        weights = np.nan*np.empty(len(diff))
        mse = calc_mse(diff)
        sd = calc_sd(diff)
        sigmas = calc_sigmas(mse=mse, sd=sd)
        msqe = calc_msqe(diff)
        rmse = calc_rmse(msqe=msqe)
    estimators = {'rmse': rmse, 'mse': mse, 'msqe':msqe, 'sigmas': sigmas, 'sd': sd}
    #print_estimators_table(estimators, save_dir + 'estimadores')
    return_value = estimators
    if return_dataframe:
        df = df.assign(
            **{
                'SHF_diff': diff,
                'SHF_weights': weights
            }
        )
        return_value = [estimators, df]
    return return_value

def calc_mse(values):
    # Mean Signed Error
    #mse = sum(values) / len(values)
    mse = values.mean()
    return mse

def calc_sd(values):
    # Standard Deviation
    #sd = np.sqrt(((values-mse)**2).mean())
    sd = np.std(values)
    return sd

def calc_sigmas(values=None, mse=None, sd=None):
    # Standard deviations from the mean
    if mse is None or sd is None:
        assert values is not None
        if sd is None:
            sd = calc_sd(values)
        if mse is None:
            mse = calc_mse(values)
    n_1_sigma = mse - sd
    p_1_sigma = mse + sd
    n_2_sigma = mse - 2*sd
    p_2_sigma = mse + 2*sd
    sigmas = {
        'p_1_sigma': p_1_sigma, 'n_1_sigma': n_1_sigma,
        'p_2_sigma': p_2_sigma, 'n_2_sigma': n_2_sigma}
    #moda = np.nanmax(abs(values))
    return sigmas

def calc_msqe(values):
    # Mean Squared Error
    msqe = (values**2).mean()
    return msqe

def calc_rmse(values=None, msqe=None):
    # Root Mean Squared Error
    if msqe is None:
        assert values is not None
        msqe = calc_msqe(values)
    rmse = np.sqrt(msqe)
    return rmse

def calc_mse_weighted(values, weights):
    # Mean Signed Error
    #mse = np.average(values, weights = n_weights)
    #mse = sum(n_weights*values)
    mse = sum(weights*values)/sum(weights)
    return mse

def calc_sd_weighted(values, weights, mse=None):
    # Standard Deviation
    if mse is None:
        mse = calc_mse_weighted(values, weights)
    V1 = sum(weights)
    V2 = sum(weights**2)
    sd = np.sqrt(sum(weights*((values-mse)**2))/(V1-(V2/V1)))
    return sd

def calc_sigmas_weighted(values=None, weights=None, mse=None, sd=None):
    # Standard deviations from the mean
    if mse is None or sd is None:
        assert values is not None and weights is not None
        if mse is None:
            mse = calc_mse_weighted(values, weights)
        if sd is None:
            sd = calc_sd_weighted(values, weights, mse=mse)
    n_1_sigma = mse - sd
    p_1_sigma = mse + sd
    n_2_sigma = mse - 2*sd
    p_2_sigma = mse + 2*sd
    sigmas = {
        'p_1_sigma': p_1_sigma, 'n_1_sigma': n_1_sigma,
        'p_2_sigma': p_2_sigma, 'n_2_sigma': n_2_sigma}
    return sigmas

def calc_msqe_weighted(values, weights):
    # Mean Squared Error
    #msqe_weighted = sum(n_weights*(values**2))
    msqe = sum(weights*(values**2)) / sum(weights)
    return msqe

def calc_rmse_weighted(
        values=None, weights=None, msqe=None
    ):
    # Root Mean Squared Error
    if msqe is None:
        assert values is not None and weights is not None
        msqe = calc_msqe_weighted(values, weights)
    rmse = np.sqrt(msqe)
    return rmse

def print_estimators_table(estimators, filename):
    if filename == None:
        filename = 'Output/estimadores'
    df = pd.DataFrame.from_dict(estimators)
    writer = pd.ExcelWriter(filename + '.xlsx')
    df.to_excel(writer, 'Estimadores')
    writer.save()
