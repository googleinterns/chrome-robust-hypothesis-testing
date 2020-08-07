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
            scale parameter, or theta^star, for the gamma distribution

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
            shape parameter, or k^star, for the gamma distribution

        gamma_scale: float
            scale parameter, or theta^star, for the gamma distribution

    Returns
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


def generate_raw_data_exponential(number_of_users, scale_param=1):
    """
    Function that generates per user data from exponential distribution

    Parameters
    ----------
        number_of_users: int
            number of users in given treatment/control group

        scale_param: float
            scale parameter, or theta^star, for the gamma distribution

    Returns
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
    raw_data = [np.random.exponential(scale=scale_param, size=num_of_obs)
                for num_of_obs in number_of_observations_per_user]

    ids = [np.repeat(client_id, num_of_obs) for client_id,
           num_of_obs in enumerate(number_of_observations_per_user)]

    # stack IDs and observations
    ids_array = np.hstack(ids)
    metric_values = np.hstack(raw_data)

    return pd.DataFrame(np.vstack([ids_array, metric_values]).transpose(), columns=['Client ID', 'Metric Value'])


def generate_raw_data_lognormal(number_of_users, mean_param=0, sigma_param=1):
    """
    Function that generates per user data from exponential distribution

    Parameters
    ----------
        number_of_users: int
            number of users in given treatment/control group

        mean_param: float
            mean parameter for the lognormal distribution

        sigma_param: float
            std parameter for the lognormal distribution
    Returns
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
    raw_data = [np.random.lognormal(mean=mean_param, sigma=sigma_param, size=num_of_obs)
                for num_of_obs in number_of_observations_per_user]

    ids = [np.repeat(client_id, num_of_obs) for client_id,
           num_of_obs in enumerate(number_of_observations_per_user)]

    # stack IDs and observations
    ids_array = np.hstack(ids)
    metric_values = np.hstack(raw_data)

    return pd.DataFrame(np.vstack([ids_array, metric_values]).transpose(), columns=['Client ID', 'Metric Value'])


def generate_raw_data_mixture_of_lognormal(number_of_users, vec_of_means, vec_of_stds, weights):
    """
    Function that generates per user data from exponential distribution
    """

    # sample number of observations per user
    number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users)

    cluster_asgn = [np.random.choice([0, 1], size=num_of_obs, p=weights)
                    for num_of_obs in number_of_observations_per_user]

    # sample metric values for each user
    raw_data = [np.random.lognormal(mean=vec_of_means[cur_cluster_assignment],
                                    sigma=vec_of_stds[cur_cluster_assignment],
                                    size=len(cur_cluster_assignment))
                for cur_cluster_assignment in cluster_asgn]

    ids = [np.repeat(client_id, num_of_obs) for client_id,
           num_of_obs in enumerate(number_of_observations_per_user)]

    # stack IDs and observations
    ids_array = np.hstack(ids)
    metric_values = np.hstack(raw_data)

    return pd.DataFrame(np.vstack([ids_array, metric_values]).transpose(), columns=['Client ID', 'Metric Value'])


def get_binned_data_client_level(raw_data, bins_boundaries):
    """
    Function that bins raw data

    Parameters
    ----------
        raw_data: Dataframe
            raw data to be used to create binned data;
            Dataframe has two columns:
                - Client ID
                - Metric value


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

    # using list comprehension, create list of lists with first entry being the client ID,
    # followed by the histogram for the client
    binned_data = [[[cur_id] + np.histogram(cur_group_of_metric_values['Metric Value'].values,
                                            bins=bins_boundaries)[0].tolist()]
                   for cur_id, cur_group_of_metric_values in raw_data.groupby('Client ID')]

    # convert an array to output Dataframe
    cols = ['Client ID'] + ['Bin ' + str(i) for i in range(num_of_hist_bins)]
    binned_data = pd.DataFrame(np.vstack(binned_data), columns=cols)

    return binned_data


def get_binned_data_cookie_bucket_level(raw_data, number_of_buckets, bins_boundaries):
    """
    Function that bins raw data and aggregates histograms at a cookie bucket level

    Parameters
    ----------
        raw_data: Dataframe
            raw data to be used to create binned data;
            Dataframe has two columns:
                - Client ID
                - Metric value

        number_of_buckets: int
            number of cookie buckets to be used
            Bucketing is performed based on the remained of
            the ID when divided by number_of_cookie_buckets

        bins_boundaries: array_like
            array of bins' boundaries
                (With the right-most boundary for the overflow bins 
                and left-most boundary for underflow bin included)
                Example: [0,1,2,3,4] would correspond to bins:
                    (0,1], (1,2], (2,3], (3,4]

    Returns
    ----------
        binned_data: Dataframe
            Dataframe with each row corresponding to the histogram
                of a given cookie buckets

    """
    # get number of bins
    num_of_hist_bins = len(bins_boundaries) - 1

    # get Clients' buckets
    buckets = raw_data['Client ID'] % number_of_buckets

    binned_data = [np.histogram(cur_group_of_metric_values['Metric Value'].values,
                                bins=bins_boundaries)[0]
                   for _, cur_group_of_metric_values in raw_data.groupby(buckets)]

    # convert an array to output Dataframe
    cols = ['Bin ' + str(i) for i in range(num_of_hist_bins)]
    binned_data = pd.DataFrame(np.vstack(binned_data), columns=cols)

    return binned_data


def generate_data_mixture_exp_bucket_level(number_of_users, scale_params, weights, num_of_cookie_buckets, bins_boundaries):
    """
    Function that generates data from mixture of exponential distribution, but at a cookie bucket level
    i.e. data pts in a given cookie bucket is generated from the same distribution, but shift might occur
    in some (small number of buckets)
    -- might be generalized to more components than two
    """

    # sample number of observations per user
    treatment_number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users[0])

    control_number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users[1])

    # get IDs to compute number of observations in each cookie bucket
    treat_user_ids = np.arange(number_of_users[0])
    control_user_ids = np.arange(number_of_users[1])

    # match clients with cookie buckets
    treat_bucket_assignment = treat_user_ids % num_of_cookie_buckets
    control_bucket_assignment = control_user_ids % num_of_cookie_buckets

    # compute number of observations per cookie bucket
    pos_cookie_bucket_indices = np.arange(num_of_cookie_buckets)

    num_of_obs_cookie_bucket_treat = [treatment_number_of_observations_per_user[treat_bucket_assignment
                                                                                == cur_cookie_bucket].sum()
                                      for cur_cookie_bucket in pos_cookie_bucket_indices]
    num_of_obs_cookie_bucket_control = [control_number_of_observations_per_user[control_bucket_assignment
                                                                                == cur_cookie_bucket].sum()
                                        for cur_cookie_bucket in pos_cookie_bucket_indices]

    # define which component to sample from for each cookie bucket
    comp_assgn = np.random.choice(
        np.arange(len(scale_params)), size=num_of_cookie_buckets, p=weights)

    # sample data
    raw_obs_treat = [np.random.exponential(scale=scale_params[comp_assgn[cur_cookie_bucket]],
                                           size=num_of_obs_cookie_bucket_treat[cur_cookie_bucket])
                     for cur_cookie_bucket in pos_cookie_bucket_indices]
    raw_obs_control = [np.random.exponential(scale=scale_params[comp_assgn[cur_cookie_bucket]],
                                             size=num_of_obs_cookie_bucket_control[cur_cookie_bucket])
                       for cur_cookie_bucket in pos_cookie_bucket_indices]

    # get binned data
    binned_data_treat = [np.histogram(cur_bucket, bins=bins_boundaries)[0]
                         for cur_bucket in raw_obs_treat]
    binned_data_control = [np.histogram(cur_bucket, bins=bins_boundaries)[0]
                           for cur_bucket in raw_obs_control]

    return np.stack(binned_data_treat), np.stack(binned_data_control)


def generate_data_single_corrupted_exp_bucket_level(number_of_users, scale_params, num_of_cookie_buckets, bins_boundaries):
    """
    Function that generates data from mixture of exponential distribution, but at a cookie bucket level
    i.e. data pts in a given cookie bucket is generated from the same distribution, but shift might occur
    in one cookie bucket irrespectively to the total number of cookie buckets
    -- might be generalized to more components than two
    """

    # sample number of observations per user
    treatment_number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users[0])

    control_number_of_observations_per_user = np.random.geometric(
        p=0.03, size=number_of_users[1])

    # get IDs to compute number of observations in each cookie bucket
    treat_user_ids = np.arange(number_of_users[0])
    control_user_ids = np.arange(number_of_users[1])

    # match clients with cookie buckets
    treat_bucket_assignment = treat_user_ids % num_of_cookie_buckets
    control_bucket_assignment = control_user_ids % num_of_cookie_buckets

    # compute number of observations per cookie bucket
    pos_cookie_bucket_indices = np.arange(num_of_cookie_buckets)

    num_of_obs_cookie_bucket_treat = [treatment_number_of_observations_per_user[treat_bucket_assignment
                                                                                == cur_cookie_bucket].sum()
                                      for cur_cookie_bucket in pos_cookie_bucket_indices]
    num_of_obs_cookie_bucket_control = [control_number_of_observations_per_user[control_bucket_assignment
                                                                                == cur_cookie_bucket].sum()
                                        for cur_cookie_bucket in pos_cookie_bucket_indices]

    # define which component to sample from for each cookie bucket
    comp_assgn = np.zeros(num_of_cookie_buckets, dtype='int')
    comp_assgn[0] = 1
    np.random.shuffle(comp_assgn)

    # sample data
    raw_obs_treat = [np.random.exponential(scale=scale_params[comp_assgn[cur_cookie_bucket]],
                                           size=num_of_obs_cookie_bucket_treat[cur_cookie_bucket])
                     for cur_cookie_bucket in pos_cookie_bucket_indices]
    raw_obs_control = [np.random.exponential(scale=scale_params[comp_assgn[cur_cookie_bucket]],
                                             size=num_of_obs_cookie_bucket_control[cur_cookie_bucket])
                       for cur_cookie_bucket in pos_cookie_bucket_indices]

    # get binned data
    binned_data_treat = [np.histogram(cur_bucket, bins=bins_boundaries)[0]
                         for cur_bucket in raw_obs_treat]
    binned_data_control = [np.histogram(cur_bucket, bins=bins_boundaries)[0]
                           for cur_bucket in raw_obs_control]

    return np.stack(binned_data_treat), np.stack(binned_data_control)
