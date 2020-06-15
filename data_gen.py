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
    # create dataframe to store the results
    raw_data = pd.DataFrame(columns=['Client ID', 'Metric Value'])

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
        cur_observations = np.random.gamma(shape_array, scale_array).tolist()
        # add observations to the dataframe
        for cur_obs in cur_observations:
            raw_data = raw_data.append(
                {'Client ID': client_id, 'Metric Value': cur_obs}, ignore_index=True)
    return raw_data


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

    # create a dataframe with corresponding columns
    binned_data = pd.DataFrame()

    binned_data['Client ID'] = 0
    for i in range(num_of_hist_bins):
        binned_data['Bin ' + str(i)] = 0

    # get unique IDs' of all clients
    unique_ids = raw_data['Client ID'].unique().astype('int')

    for cur_cliend_id in unique_ids:
        # consider all observations for each client
        cur_data = raw_data[raw_data['Client ID']
                            == cur_cliend_id]['Metric Value'].values
        # find bin those observations fall into
        items_to_add = bins_boundaries.searchsorted(cur_data)
        # Add Client to the dataframe
        binned_data = binned_data.append(
            {'Client ID': cur_cliend_id}, ignore_index=True)
        for cur_item in items_to_add:
            # for each observations check whether the bin was empty, 
            # if empty replace the NaN by 1,
            # otherwise increase count by 1 in the bin
            if pd.isnull(binned_data.iloc[cur_cliend_id]['Bin ' + str(cur_item)]):
                binned_data.iloc[cur_cliend_id]['Bin ' + str(cur_item)] = 1
            else:
                binned_data.iloc[cur_cliend_id]['Bin ' + str(cur_item)] += 1
    # replace NaNs by count = 0
    binned_data = binned_data.fillna(0)
    return binned_data
