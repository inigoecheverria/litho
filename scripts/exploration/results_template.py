from litho.utils import makedir, makedir_from_filename, export_csv
from scripts.exploration.parameter_variations import rheo_variations


def iteration_results_template(some_argument, save_dir='results_dir', plot=False):
    save_dir_maps = save_dir + 'Mapas/'
    save_dir_files = save_dir + 'Archivos/'
    def iteration_results(TM, MM, name):
        # Calculate the result for each iteration
        data = get_model_data(TM, MM, extract_bdt_data)
        some_result = data['bdt'] + some_argument
        # Save some files for each iteration
        filename = save_dir_files + name
        makedir_from_filename(filename)
        np.savetxt(filename + '_data.txt', data['bdt'])
        np.savetxt(filename + '_some_result.txt', some_result)
        if plot is True:
            mapname = save_dir_maps + name
            # Plot something for each iteration
            heatmap_map(data['bdt'], filename=mapname + '_data')
            heatmap_map(some_result, filename=mapname + '_some_result')
        # Return the results of each iteration
        return {
            'bdt': data['bdt'],
            'some_result': some_result,
        }
    # Return the function to compute iteration results
    return iteration_results

# Create the output directory
save_dir = 'Exploration/'
makedir(save_dir)

# Define parameters to variate

#uc_params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#lc_params = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
#lm_params = [23,24,25,26,27,28,29,30]
#flm_params = [23]
lc_params = [1, 3, 5, 7, 8, 11, 12, 13, 16, 17, 19, 22]

# Apply a parameter variation with the previous parameters
# Returning the results of the target iteration_results function

results = rheo_variations(
    iteration_results_template(
        scd_dict, save_dir = save_dir + 'scd_bdt/',
        plot=True
    ),
    lc_params=lc_params
)
#results = rheo_exploration(
#    functools.partial(
#        rheo_exploration,
#        scd_bdt_results(
#            scd_dict, save_dir=save_dir + 'scd_bdt/lc_uc/', plot=True
#        ),
#        uc_params=uc_params
#    ),
#    lc_params=lc_params
#)

# Extract results as you prefer
bdts = [results['bdt'] for result in list(results.values())]
some_results = [results['some_result'] for result in list(results.values())]

# Do whatever with them
