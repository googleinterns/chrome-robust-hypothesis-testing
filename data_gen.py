import numpy as np
import pandas as pd


def generate_raw_data(number_of_users, gamma_shape=5, gamma_scale=500):
    """
    Function that is used to generate Metric 1; 
    full description can be found in the supporting notebook

    Parameters
    ----------
        number_of_users: int
            number of users in given treatment/control group

        gamma_shape: float
            shape paramter, or k^star, for the gamma distribution

        gamma_scale: float
            scale paramter, or theta^star, for the gamma distribution

    Parameters
    ----------
        raw_data: Dataframe
            Dataframe with two columns:
                - Client ID
                - Metric value

    """
    # create array to store the results
    raw_data = np.empty(shape=[0, 2])

    # sample number of observations per user
    number_of_observations_per_user = np.random.geometric(
        p=0.05, size=number_of_users)

    for client_id, cur_num_of_obs in enumerate(number_of_observations_per_user):
        # for each user sample sample k_i and theta_i
        shape_array = np.maximum(np.random.normal(
            loc=gamma_shape, scale=2, size=cur_num_of_obs), 2)
        scale_array = np.maximum(np.random.normal(
            loc=gamma_scale / cur_num_of_obs, scale=100, size=cur_num_of_obs), 300)
        # sample metric values
        cur_observations = np.random.gamma(shape_array, scale_array)
        # add observations to the array
        to_add = np.vstack(
            [np.repeat(client_id, cur_num_of_obs), cur_observations]).transpose()
        raw_data = np.concatenate([raw_data, to_add])

    return pd.DataFrame(raw_data, columns=['Client ID', 'Metric Value'])


def get_binned_data(raw_data, bins_boundaries):
    """
    Function that bins raw data

    Parameters
    ----------
        raw_data: Dataframe
            raw data to be used to create binned data

        bins_boundaries: array_like
            array of bins' boundaries

    Returns
    ----------
        binned_data: Dataframe
            binned data at a Client level
    """
    # get number of bins
    num_of_hist_bins = len(bins_boundaries)

    # get arrays for both client IDs and metric values
    ids = raw_data['Client ID'].values.astype('int')
    metric_values = raw_data['Metric Value'].values

    # get unique IDs' of all clients
    unique_ids = np.unique(ids)
    num_of_ids = len(unique_ids)

    # create an array that stores histograms/counts for all clients
    binned_data = np.zeros(shape=[num_of_ids, num_of_hist_bins+1])
    binned_data[:, 0] = unique_ids

    # determine which bins/buckets observations belong to
    # Side note: np.histogram would compute counts of a flattened array ->
    # here histograms would be computed essentially for each client
    indices_of_bins = bins_boundaries.searchsorted(metric_values)

    # add counts to the corresponding (user, bin) locations
    np.add.at(binned_data, (ids, indices_of_bins+1), 1)

    # convert an array to output Dataframe
    cols = ['Client ID'] + ['Bin ' + str(i) for i in range(num_of_hist_bins)]
    binned_data = pd.DataFrame(binned_data, columns=cols)

    return binned_data
