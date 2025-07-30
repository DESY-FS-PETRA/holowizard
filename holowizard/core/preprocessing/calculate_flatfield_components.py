import numpy as np
from sklearn.decomposition import FastICA

from holowizard.core.utils.remove_outliers import remove_outliers_multiprocess_wrapper


def calculate_flatfield_components(ref_data, num_components, model=FastICA, log_mode=True):
    if log_mode:
        filtered_ref_data = np.array([remove_outliers_multiprocess_wrapper(x) for x in np.log(ref_data)])
    else:
        filtered_ref_data = np.array([remove_outliers_multiprocess_wrapper(x) for x in ref_data])

    components_model = model(n_components=num_components)
    components_input_dims = [
        filtered_ref_data.shape[0],
        filtered_ref_data.shape[2] * filtered_ref_data.shape[1],
    ]
    flatfields = filtered_ref_data.reshape(components_input_dims)
    components_model.fit(flatfields)

    return components_model
