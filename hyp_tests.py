import numpy as np
import pandas as pd
from tools import get_alpha_quantile_student, \
    compute_mean_hist_data, compute_quantile_hist_data
from scipy.stats import norm
import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
import pdb


def jackknife_t_test(treatment_mv_all_buckets, treatment_mv_all_expect_given,
                     control_mv_all_buckets, control_mv_all_expect_given,
                     ci_level):
    """
    Jackknife t-test for data in cookie buckets

    Parameters
    ----------
      treatment_mv_all_buckets: float
        aggregated metric over all buckets 
          for treatment group

      treatment_mv_all_expect_given: array_like
        aggregated metric over all buckets expect one 
          for treatment group

      control_mv_all_buckets: float
        aggregated metric over all buckets 
          for control group

      control_mv_all_expect_given:
        aggregated metric over all buckets expect one 
          for control group

      ci_level: float in (0,1)
        nominal coverage of the CI 

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 
    """

    # compute number of buckets
    number_of_buckets = len(treatment_mv_all_expect_given.ravel())

    # obtain correct quantile of the Student t-distribution corresponding
    # to (N-1) degrees of freedom
    qq = get_alpha_quantile_student(ci_level, number_of_buckets-1)

    # check that treatment and control have the same number of buckets
    assert number_of_buckets == len(control_mv_all_expect_given.ravel())

    # compute percent change for total change of metric value (mv)
    overall_percent_change = 100 * \
        (treatment_mv_all_buckets / control_mv_all_buckets - 1)

    # compute percent change for all buckets except given
    percent_change_for_all_but_given = 100 * \
        (treatment_mv_all_expect_given / control_mv_all_expect_given - 1)

    # compute ps_{-j} according to the definition
    ps_j = number_of_buckets * overall_percent_change - \
        (number_of_buckets - 1) * percent_change_for_all_but_given

    # compute center and radius of the CI
    ci_center = ps_j.mean()
    std_ps = ps_j.std(ddof=1)

    ci_size = qq * std_ps / np.sqrt(number_of_buckets)

    return ci_center, ci_size


def jackknife_cookie_bucket_mean(treatment_group, control_group,
                                 number_of_buckets, ci_level):
    """"
    Perform jackknife cookie bucket test (for not binned data)

    Parameters
    ----------
      treatment_group: pd.Dataframe
        dataframe corresponding to treatment group that contains two columns:
          - Client ID
          - Metric Value

      control_group: pd.Dataframe
        dataframe corresponding to control group that contains two columns:
          - Client ID
          - Metric Value

      number_of_buckets: int
        number of buckets to be used for performing the test

      ci_level: (0,1)
        nominal coverage of the CI 

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 

    """

    # temporarily add bucket id
    treatment_buckets = (
        treatment_group['Client ID'] % number_of_buckets).astype('int')
    control_buckets = (
        control_group['Client ID'] % number_of_buckets).astype('int')

    # list of all bucket ids
    bucket_ids = np.arange(number_of_buckets).tolist()

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    for cur_bucket in bucket_ids:
        # leave one bucket out and compute the mean
        treatment_group_mv += [treatment_group.loc[treatment_buckets
                                                   != cur_bucket]['Metric Value'].mean()]
        control_group_mv += [control_group.loc[control_buckets
                                               != cur_bucket]['Metric Value'].mean()]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute mean metric value in both groups
    treatment_all = treatment_group['Metric Value'].mean()
    control_all = control_group['Metric Value'].mean()

    ci_center, ci_size = jackknife_t_test(treatment_all, treatment_group_mv,
                                          control_all, control_group_mv,
                                          ci_level)

    return ci_center, ci_size


def jackknife_cookie_bucket_quantile(treatment_group, control_group,
                                     number_of_buckets, ci_level, quantile_to_test):
    """"
    Perform jackknife cookie bucket test (for not binned data)

    Parameters
    ----------
      treatment_group: pd.Dataframe
        dataframe corresponding to treatment group that contains two columns:
          - Client ID
          - Metric Value

      control_group: pd.Dataframe
        dataframe corresponding to control group that contains two columns:
          - Client ID
          - Metric Value

      number_of_buckets: int
        number of buckets to be used for performing the test

      ci_level: (0,1)
        nominal coverage of the CI 

      quantile_to_test: (0,1)
        quantile for which Jackknife Cookie Bucket test is to be performed

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 

    """

    # temporarily add bucket id
    treatment_buckets = (
        treatment_group['Client ID'] % number_of_buckets).astype('int')
    control_buckets = (
        control_group['Client ID'] % number_of_buckets).astype('int')

    # list of all bucket ids
    bucket_ids = np.arange(number_of_buckets).tolist()

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    for cur_bucket in bucket_ids:
        # leave one bucket out and compute the quantile
        treatment_group_mv += [treatment_group.loc[treatment_buckets
                                                   != cur_bucket][
            'Metric Value'].quantile(q=quantile_to_test)]
        control_group_mv += [control_group.loc[control_buckets
                                               != cur_bucket][
            'Metric Value'].quantile(q=quantile_to_test)]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute overall quantiles in both groups
    treatment_all = treatment_group['Metric Value'].quantile(
        q=quantile_to_test)
    control_all = control_group['Metric Value'].quantile(q=quantile_to_test)

    ci_center, ci_size = jackknife_t_test(treatment_all, treatment_group_mv,
                                          control_all, control_group_mv,
                                          ci_level)

    return ci_center, ci_size


def jackknife_cookie_bucket_quantile_binned(treatment_binned_data, control_binned_data,
                                            bins_boundaries, number_of_buckets,
                                            ci_level, quantile_to_test):
    """"
    Perform jackknife cookie bucket test (for binned data)

    Parameters
    ----------
      treatment_binned_data: pd.Dataframe
        dataframe corresponding to treatment group that contains two types of columns:
          - Client ID
          - Count of observations in Bin "j" for a given client

      control_binned_data: pd.Dataframe
        dataframe corresponding to control group that contains two types of columns:
          - Client ID
          - Count of observations in Bin "j" for a given client

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      number_of_buckets: int
        number of buckets to be used for performing the test

      ci_level: (0,1)
        nominal coverage of the CI 

      quantile_to_test: (0,1)
        quantile for which Jackknife Cookie Bucket test is to be performed

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 

    """

    # pdb.set_trace()

    # list of all bucket ids
    bucket_ids = np.arange(number_of_buckets).tolist()

    # compute buckets corresponding to each client
    treatment_buckets = (treatment_binned_data['Client ID'] %
                         number_of_buckets).astype('int')

    control_buckets = (control_binned_data['Client ID'] %
                       number_of_buckets).astype('int')

    # compute histogram data / total counts within each bucket
    treatment_bucket_data = treatment_binned_data.groupby(
        treatment_buckets).sum()
    control_bucket_data = control_binned_data.groupby(control_buckets).sum()

    # drop client ID from bucketed data
    treatment_bucket_data = treatment_bucket_data.drop(
        axis=1, columns=['Client ID'])
    control_bucket_data = control_bucket_data.drop(
        axis=1, columns=['Client ID'])

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    for cur_bucket in bucket_ids:
        # leave one bucket out and compute the total counts for the left buckets
        cur_hist_treatment = treatment_bucket_data[treatment_bucket_data.index
                                                   != cur_bucket].sum().values
        cur_hist_control = control_bucket_data[control_bucket_data.index
                                               != cur_bucket].sum().values
        # compute approximate quantiles based on binned data
        treatment_group_mv += [compute_quantile_hist_data(
            cur_hist_treatment, bins_tuples, quantile=quantile_to_test)]
        control_group_mv += [compute_quantile_hist_data(
            cur_hist_control, bins_tuples, quantile=quantile_to_test)]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute approximate quantiles for all buckets and both groups
    treatment_all = compute_quantile_hist_data(
        treatment_bucket_data.sum().values, bins_tuples, quantile=quantile_to_test)

    control_all = compute_quantile_hist_data(
        control_bucket_data.sum().values, bins_tuples, quantile=quantile_to_test)

    ci_center, ci_size = jackknife_t_test(treatment_all, treatment_group_mv,
                                          control_all, control_group_mv,
                                          ci_level)

    return ci_center, ci_size


def jackknife_cookie_bucket_mean_binned(treatment_binned_data, control_binned_data,
                                        bins_boundaries, number_of_buckets,
                                        ci_level):
    """"
    Perform jackknife cookie bucket test (for binned data)

    Parameters
    ----------
      treatment_binned_data: pd.Dataframe
        dataframe corresponding to treatment group that contains two types of columns:
          - Client ID
          - Count of observations in Bin "j" for a given client

      control_binned_data: pd.Dataframe
        dataframe corresponding to control group that contains two types of columns:
          - Client ID
          - Count of observations in Bin "j" for a given client

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      number_of_buckets: int
        number of buckets to be used for performing the test

      ci_level: (0,1)
        nominal coverage of the CI 

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 

    """

    # list of all bucket ids
    bucket_ids = np.arange(number_of_buckets).tolist()

    # compute buckets corresponding to each client
    treatment_buckets = (treatment_binned_data['Client ID'] %
                         number_of_buckets).astype('int')

    control_buckets = (control_binned_data['Client ID'] %
                       number_of_buckets).astype('int')

    # compute histogram data / total counts within each bucket
    treatment_bucket_data = treatment_binned_data.groupby(
        treatment_buckets).sum()
    control_bucket_data = control_binned_data.groupby(control_buckets).sum()

    # drop client ID from bucketed data
    treatment_bucket_data = treatment_bucket_data.drop(
        axis=1, columns=['Client ID'])
    control_bucket_data = control_bucket_data.drop(
        axis=1, columns=['Client ID'])

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    for cur_bucket in bucket_ids:
        # leave one bucket out and compute the total counts for the left buckets
        cur_hist_treatment = treatment_bucket_data[treatment_bucket_data.index
                                                   != cur_bucket].sum().values
        cur_hist_control = control_bucket_data[control_bucket_data.index
                                               != cur_bucket].sum().values
        # compute approximate means based on binned data
        treatment_group_mv += [compute_mean_hist_data(
            cur_hist_treatment, bins_tuples)]
        control_group_mv += [compute_mean_hist_data(
            cur_hist_control, bins_tuples)]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute approximate means for all buckets and both groups
    treatment_all = compute_mean_hist_data(
        treatment_bucket_data.sum().values, bins_tuples)

    control_all = compute_mean_hist_data(
        control_bucket_data.sum().values, bins_tuples)

    ci_center, ci_size = jackknife_t_test(treatment_all, treatment_group_mv,
                                          control_all, control_group_mv,
                                          ci_level)

    return ci_center, ci_size


def jackknife_cookie_bucket_quantile_bucketed(treatment_cookie_buckets, control_cookie_buckets,
                                              bins_boundaries, ci_level, quantile_to_test, return_interval=False):
    """"
    Perform jackknife cookie bucket test (for binned data in cookie buckets)

    Parameters
    ----------
      treatment_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      control_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      ci_level: (0,1)
        nominal coverage of the CI 

      quantile_to_test: (0,1)
        quantile for which Jackknife Cookie Bucket test is to be performed

    Returns
    ----------
      ci_center:
        center of the confidence interval

      ci_size:
        size/radius of the confidence interval 

    """

    # # convert to arrays for treatment and control groups
    # data_treat = treatment_cookie_buckets.values
    # data_control = control_cookie_buckets.values

    # compute number of  cookie buckets
    num_of_cookie_buckets = treatment_cookie_buckets.shape[0]
    assert num_of_cookie_buckets == control_cookie_buckets.shape[0]

    # list of all bucket ids
    indices = np.arange(num_of_cookie_buckets)

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    for cur_bucket in indices:
        # drop current bucket
        ind_treat = np.delete(indices, cur_bucket)
        ind_control = np.delete(indices, cur_bucket)
        # get corresponding histograms
        cur_hist_treatment = treatment_cookie_buckets[ind_treat, :].sum(axis=0)
        cur_hist_control = control_cookie_buckets[ind_control, :].sum(axis=0)
        # compute approximate quantiles based on binned data
        treatment_group_mv += [compute_quantile_hist_data(
            cur_hist_treatment, bins_tuples, quantile=quantile_to_test)]
        control_group_mv += [compute_quantile_hist_data(
            cur_hist_control, bins_tuples, quantile=quantile_to_test)]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute approximate quantiles for all buckets and both groups
    treatment_all = compute_quantile_hist_data(
        treatment_cookie_buckets.sum(axis=0), bins_tuples, quantile=quantile_to_test)

    control_all = compute_quantile_hist_data(
        control_cookie_buckets.sum(axis=0), bins_tuples, quantile=quantile_to_test)

    ci_center, ci_radius = jackknife_t_test(treatment_all, treatment_group_mv,
                                            control_all, control_group_mv,
                                            ci_level)

    left_bound = ci_center - ci_radius
    right_bound = ci_center + ci_radius

    if return_interval:
        # in case we want to return the estimators for analyses
        if 0 < left_bound or right_bound < 0:
            return True, [left_bound, right_bound]
        else:
            return False, [left_bound, right_bound]

    else:
        # in case we just need to perform the test
        if 0 < left_bound or right_bound < 0:
            return True
        else:
            return False


def percentile_bootstrap_ci_ratio_cookie_buckets(treatment_cookie_buckets, control_cookie_buckets,
                                                 bins_boundaries, num_of_boot_samples, ci_level,
                                                 estimator_type='quantile',
                                                 quantile=0.5, paired=False, return_bootstrap_est=False, return_interval=False):
    """
    Function that computes a bootstrap CI for the effect size in terms of ratio

    Parameters
    ----------
      treatment_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      control_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      num_of_boot_samples: int
        number of bootstrap samples to use

      ci_level: float in (0,1)
        level at which to construct the confidence interval

      estimator_type: 'quantile' or 'mean'
        whether to perform a test when the effect size is in terms of mean or quantile

      quantile: (0,1)
        quantile for which the test is to be performed

      return_bootstrap_est: Bool
        whether to return estimators corresponding to bootstrap samples

      return_interval: Bool
        whether to return left and right boundary of the confidence interval

    Returns
    ----------
      test_result: Bool
        whether the test rejects (True) the null hypothesis

      If return_bootstrap_est is True, then estimators computed on bootstrap samples are returned
      (might worth further to return the interval itself-for length and shape comparison)

    """
    # convert to arrays for treatment and control groups

    # !!! some notebooks have to be updates to pass arrays

    # data_treat = treatment_cookie_buckets.values
    # data_control = control_cookie_buckets.values

    # stacking boostrap estimators
    bootstrap_est = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    # get number of cookie buckets for treatment and check that
    # it matches control
    num_of_cookie_buckets = treatment_cookie_buckets.shape[0]
    assert num_of_cookie_buckets == control_cookie_buckets.shape[0]

    # possible indices to perform resamplaing
    indices = np.arange(num_of_cookie_buckets)

    for i in range(num_of_boot_samples):
        if paired:
            # get bootstrap indices for treatment and control separately
            ind_resampled = np.random.choice(
                indices, size=num_of_cookie_buckets)
            # get an bootstrap array for treatment and control separately
            # and compute the resulting histogram
            boot_treat = treatment_cookie_buckets[ind_resampled, :].sum(axis=0)
            boot_control = control_cookie_buckets[ind_resampled, :].sum(axis=0)
            # compute estimator
        else:
          # get bootstrap indices for treatment and control separately
            ind_treat = np.random.choice(indices, size=num_of_cookie_buckets)
            ind_control = np.random.choice(indices, size=num_of_cookie_buckets)
            # get an bootstrap array for treatment and control separately
            # and compute the resulting histogram
            boot_treat = treatment_cookie_buckets[ind_treat, :].sum(axis=0)
            boot_control = control_cookie_buckets[ind_control, :].sum(axis=0)

        # compute estimator
        if estimator_type is 'quantile':
            quant_treat = compute_quantile_hist_data(
                boot_treat, bins_tuples, quantile)
            quant_control = compute_quantile_hist_data(
                boot_control, bins_tuples, quantile)
            bootstrap_est += [100*(quant_treat/quant_control-1)]
    # compute quantiles of bootstrap estiamtors
    left_bound = np.quantile(bootstrap_est, q=(1 - ci_level)/2)
    right_bound = np.quantile(bootstrap_est, q=(1 + ci_level)/2)
    if return_interval:
        if return_bootstrap_est:
            # in case we want to return the estimators for analyses
            if 0 < left_bound or right_bound < 0:
                return True, [left_bound, right_bound], bootstrap_est
            else:
                return False, [left_bound, right_bound], bootstrap_est
        else:
            # in case we just need to perform the test
            if 0 < left_bound or right_bound < 0:
                return True, [left_bound, right_bound]
            else:
                return False, [left_bound, right_bound]
    else:
        if return_bootstrap_est:
            # in case we want to return the estimators for analyses
            if 0 < left_bound or right_bound < 0:
                return True, bootstrap_est
            else:
                return False, bootstrap_est
        else:
            # in case we just need to perform the test
            if 0 < left_bound or right_bound < 0:
                return True
            else:
                return False


def bc_a_bootstrap_ratio_cookie_buckets(treatment_cookie_buckets, control_cookie_buckets,
                                        bins_boundaries, num_of_boot_samples, ci_level,
                                        estimator_type='quantile',
                                        quantile=0.5, paired=False, return_bootstrap_est=False, return_interval=False):
    """
    Function that computes BCa bootstrap CI

    Parameters
    ----------
      treatment_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      control_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      num_of_boot_samples: int
        number of bootstrap samples to use

      ci_level: float in (0,1)
        level at which to construct the confidence interval

      estimator_type: 'quantile' or 'mean'
        whether to perform a test when the effect size is in terms of mean or quantile

      quantile: (0,1)
        quantile for which the test is to be performed

      return_bootstrap_est: Bool
        whether to return estimators corresponding to bootstrap samples

    Returns
    ----------
      test_result: Bool
        whether the test rejects (True) the null hypothesis

      If return_bootstrap_est is True, then estimators computed on bootstrap samples are returned
      (might worth further to return the interval itself-for length and shape comparison)

    """
    # convert to arrays for treatment and control groups

    # !!! some notebooks have to be updates to pass arrays

    # data_treat = treatment_cookie_buckets.values
    # data_control = control_cookie_buckets.values

    # stacking boostrap estimators
    bootstrap_est = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    # get number of cookie buckets for treatment and check that
    # it matches control
    num_of_cookie_buckets = treatment_cookie_buckets.shape[0]
    assert num_of_cookie_buckets == control_cookie_buckets.shape[0]

    # possible indices to perform resamplaing
    indices = np.arange(num_of_cookie_buckets)

    for i in range(num_of_boot_samples):
        if paired:
            # get bootstrap indices for treatment and control separately
            ind_resampled = np.random.choice(
                indices, size=num_of_cookie_buckets)
            # get an bootstrap array for treatment and control separately
            # and compute the resulting histogram
            boot_treat = treatment_cookie_buckets[ind_resampled, :].sum(axis=0)
            boot_control = control_cookie_buckets[ind_resampled, :].sum(axis=0)
            # compute estimator
        else:
          # get bootstrap indices for treatment and control separately
            ind_treat = np.random.choice(indices, size=num_of_cookie_buckets)
            ind_control = np.random.choice(indices, size=num_of_cookie_buckets)
            # get an bootstrap array for treatment and control separately
            # and compute the resulting histogram
            boot_treat = treatment_cookie_buckets[ind_treat, :].sum(axis=0)
            boot_control = control_cookie_buckets[ind_control, :].sum(axis=0)

        # compute estimator
        if estimator_type is 'quantile':
            quant_treat = compute_quantile_hist_data(
                boot_treat, bins_tuples, quantile)
            quant_control = compute_quantile_hist_data(
                boot_control, bins_tuples, quantile)
            assert quant_control > 0
            bootstrap_est += [100 * (quant_treat / quant_control - 1)]

    # convert list to array for simplicity
    bootstrap_est = np.array(bootstrap_est)

    # compute estimator on original sample
    quant_treat = compute_quantile_hist_data(
        treatment_cookie_buckets.sum(axis=0), bins_tuples, quantile)
    quant_control = compute_quantile_hist_data(
        control_cookie_buckets.sum(axis=0), bins_tuples, quantile)
    est_ratio = 100 * (quant_treat / quant_control - 1)

    # compute bias correction
    pre_z = (bootstrap_est <= est_ratio).mean()
    z_0 = norm.ppf(pre_z)

    # compute leave-one-bucket-out estimators
    leave_one_out_est = list()

    for i in range(num_of_cookie_buckets):
        # leave one cookie bucket out at a time
        ind_treat = np.delete(indices, i)
        ind_control = np.delete(indices, i)
        # get corresponding histograms
        current_data_treat = treatment_cookie_buckets[ind_treat, :].sum(axis=0)
        current_data_control = control_cookie_buckets[ind_control, :].sum(
            axis=0)
        # compute estimator
        if estimator_type is 'quantile':
            quant_treat = compute_quantile_hist_data(
                current_data_treat, bins_tuples, quantile)
            quant_control = compute_quantile_hist_data(
                current_data_control, bins_tuples, quantile)
            leave_one_out_est += [100 * (quant_treat / quant_control - 1)]

    # convert list to array for simplicity
    leave_one_out_est = np.array(leave_one_out_est)

    # take the mean for further comp of infl fns
    est_mean = leave_one_out_est.mean()

    # compute influence functions
    infl_fns = (num_of_cookie_buckets-1) * (est_mean-leave_one_out_est)

    # compute acceleration factor
    num = sum(infl_fns ** 3)
    den = sum(infl_fns ** 2) ** (3/2)
    accel_factor = num / (6 * den)

    # compute left and right quantiles of standard normal
    left_q, right_q = norm.ppf([(1 - ci_level)/2, (1 + ci_level)/2])

    # transform using bias correction and acceleration
    left_bound = z_0 + (z_0 + left_q) / (1 - accel_factor * (z_0 + left_q))
    right_bound = z_0 + (z_0 + right_q) / (1 - accel_factor * (z_0 + right_q))

    # apply gaussian transform
    left_bound, right_bound = norm.cdf([left_bound, right_bound])

    # apply inverse transform using empirical cdf of bootstrap samples
    sample_edf = edf.ECDF(bootstrap_est)
    slope_changes = sorted(set(bootstrap_est))

    sample_edf_values_at_slope_changes = [
        sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    left_bound, right_bound = inverted_edf([left_bound, right_bound])

    if return_interval:
        if return_bootstrap_est:
            # in case we want to return the estimators for analyses
            if 0 < left_bound or right_bound < 0:
                return True, [left_bound, right_bound], bootstrap_est
            else:
                return False, [left_bound, right_bound], bootstrap_est
        else:
            # in case we just need to perform the test
            if 0 < left_bound or right_bound < 0:
                return True, [left_bound, right_bound]
            else:
                return False, [left_bound, right_bound]
    else:
        if return_bootstrap_est:
            # in case we want to return the estimators for analyses
            if 0 < left_bound or right_bound < 0:
                return True, bootstrap_est
            else:
                return False, bootstrap_est
        else:
            # in case we just need to perform the test
            if 0 < left_bound or right_bound < 0:
                return True
            else:
                return False


def compute_pseudovals_jackknife(treatment_cookie_buckets, control_cookie_buckets,
                                 bins_boundaries, quantile_to_test):
    """
    Given data in cookie buckets, compute Jackknife pseudovalues

    Parameters
    ----------
      treatment_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      control_cookie_buckets: pd.Dataframe
        dataframe corresponding to treatment group in which each rows corresponds to a cookie bucket
          and number of columns correspond to number of bins in a histogram

      bins_boundaries: list of bins boundaries
        Example: [1,3,5,9,16] corresponds to 4 bins

      quantile_to_test: (0,1)
        quantile for which Jackknife Cookie Bucket test is to be performed

    Returns
    ----------
      ps_j: array_like
        Jackknife pseudo-values

    """

    # convert to arrays for treatment and control groups

    # !!! some notebooks have to be updates to pass arrays

    # data_treat = treatment_cookie_buckets.values
    # data_control = control_cookie_buckets.values

    # compute number of  cookie buckets
    num_of_cookie_buckets = treatment_cookie_buckets.shape[0]
    assert num_of_cookie_buckets == control_cookie_buckets.shape[0]

    # list of all bucket ids
    indices = np.arange(num_of_cookie_buckets)

    # for stacking the results for both treatment and control groups
    treatment_group_mv = list()
    control_group_mv = list()

    # number of histogram bins
    num_of_bins = len(bins_boundaries)-1

    # obtain bins given boundaries
    bins_tuples = [(bins_boundaries[i-1],
                    bins_boundaries[i])
                   for i in range(1, num_of_bins+1)]

    for cur_bucket in indices:
        # drop current bucket
        ind_treat = np.delete(indices, cur_bucket)
        ind_control = np.delete(indices, cur_bucket)
        # get corresponding histograms
        cur_hist_treatment = treatment_cookie_buckets[ind_treat, :].sum(axis=0)
        cur_hist_control = control_cookie_buckets[ind_control, :].sum(axis=0)
        # compute approximate quantiles based on binned data
        treatment_group_mv += [compute_quantile_hist_data(
            cur_hist_treatment, bins_tuples, quantile=quantile_to_test)]
        control_group_mv += [compute_quantile_hist_data(
            cur_hist_control, bins_tuples, quantile=quantile_to_test)]

    # convert to array for further purposes
    treatment_group_mv = np.stack(treatment_group_mv)
    control_group_mv = np.stack(control_group_mv)

    # compute approximate quantiles for all buckets and both groups
    treatment_all = compute_quantile_hist_data(
        treatment_cookie_buckets.sum(axis=0), bins_tuples, quantile=quantile_to_test)

    control_all = compute_quantile_hist_data(
        control_cookie_buckets.sum(axis=0), bins_tuples, quantile=quantile_to_test)

    # compute percent change for total change of metric value (mv)
    overall_percent_change = 100 * \
        (treatment_all / control_all - 1)

    # compute percent change for all buckets except given
    percent_change_for_all_but_given = 100 * \
        (treatment_group_mv / control_group_mv - 1)

    # compute ps_{-j} according to the definition
    ps_j = num_of_cookie_buckets * overall_percent_change - \
        (num_of_cookie_buckets - 1) * percent_change_for_all_but_given

    return ps_j
