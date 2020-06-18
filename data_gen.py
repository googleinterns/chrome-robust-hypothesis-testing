import numpy as np
import pandas as pd


def generate_seq_from_gamma(num_of_obs, gamma_shape=25000, gamma_scale=0.1):
    """
    Function that samples metric 1 for a given user

    Parameters
    ----------
        num_of_obs: int
            number of observations for a given user

        gamma_shape: float
            shape paramter, or k^star, for the gamma distribution

        gamma_scale: float
            scale paramter, or theta^star, for the gamma distribution

    Returns
    ----------
        cur_observations: array_like
            sampled metric 1 values for a given user
    """
    # standard deviations for parameters of the gamma
    shape_std_value = 2*gamma_shape
    scale_std_value = 2*gamma_scale

    # sample parameters
    shape_array = np.random.normal(
        loc=gamma_shape, scale=shape_std_value, size=num_of_obs)
    scale_array = np.random.normal(
        loc=gamma_scale, scale=scale_std_value, size=num_of_obs)

    # truncate parameters
    shape_array = np.maximum(shape_array, gamma_shape/3)
    scale_array = np.maximum(scale_array, gamma_scale/3)

    # sample metric values
    cur_observations = np.random.gamma(shape_array, scale_array)

    return cur_observations


def generate_raw_data(number_of_users, gamma_shape=25000, gamma_scale=0.1):
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

    # sample number of observations per user
    number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users)

    # sample metric values for each user
    raw_data = [generate_seq_from_gamma(num_of_obs, gamma_shape, gamma_scale)
                for num_of_obs in number_of_observations_per_user]

    ids = [np.repeat(client_id, num_of_obs) for client_id,
           num_of_obs in enumerate(number_of_observations_per_user)]

    # stack IDs and observations
    ids_array = np.hstack(ids)
    metric_values = np.hstack(raw_data)

    return pd.DataFrame(np.vstack([ids_array, metric_values]).transpose(), columns=['Client ID', 'Metric Value'])


def get_binned_data(raw_data, bins_boundaries):
    """
    Function that bins raw data

    Parameters
    ----------
        raw_data: Dataframe
            raw data to be used to create binned data

        bins_boundaries: array_like
            array of bins' boundaries
            (With the right-most boundary for the overflow bins 
            and left-most boundary for underflow bin included)
            Example: [0,1,2,3,4] would correspond to bins:
                (0,1], (1,2], (2,3], (3,4]

    Returns
    ----------
        binned_data: Dataframe
            binned data at a Client level
    """
    # get number of bins
    num_of_hist_bins = len(bins_boundaries) - 1

    binned_data = [[[cur_id] + np.histogram(cur_group_of_metric_values['Metric Value'].values,
                                          bins=bins_boundaries)[0].tolist()]
                   for cur_id, cur_group_of_metric_values in raw_data.groupby('Client ID')]

    # convert an array to output Dataframe
    cols = ['Client ID'] + ['Bin ' + str(i) for i in range(num_of_hist_bins)]
    binned_data = pd.DataFrame(np.vstack(binned_data), columns=cols)

    return binned_data
