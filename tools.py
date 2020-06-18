from scipy.stats import t
import numpy as np
import pandas as pd
import math

GLOBAL_BIN_MAX = 2**31-1


def get_alpha_quantile_student(ci_level, degree_of_freedom=19):
    """
    Computing quantiles for Student t-distribution 
      at a given/target CI coverage level

    Parameters
    ----------
      ci_level: (0,1)
        target CI coverage level

      degree_of_freedom:
        degrees of freedom chosen according to number of buckets 

    Returns
    ----------
      quantile of the t distribution to be used in a test

    """

    interval_left_corner = (1 - ci_level) / 2
    interval_right_corner = (1 + ci_level) / 2
    qq = t(df=degree_of_freedom).ppf(
        (interval_left_corner, interval_right_corner))[1]

    return qq


def compute_quantile_hist_data(histogram_counts, bins,
                               quantile):
    """
    Compute the quantile value for binned data

    Parameters
    ----------
      histogram_counts: array_like
        a vector of counts (or weights) of |bins|.
      bins: list
        list of (int, int) pairs specifying bin lower and
        upper edge
      quantile: 
        the quantile to compute (float from ]0, 1[ interval).

    Returns
    ----------
      value of the desired quantile
    """

    # compute cumulative distirubtion
    cumulative_sum = histogram_counts.cumsum()

    # find cumulative value correponding to the target quantile
    count_below_quantile = quantile * cumulative_sum[-1]

    # find the index of the correponding bin
    nearest_position = cumulative_sum.searchsorted(count_below_quantile)

    # get left and right endpoints of the bin
    b_value_min, b_value_max = bins[nearest_position]

    # find values of the cum sum to perform interpolation
    if nearest_position == 0:
        cum_v_value_min, cum_v_value_max = 0, cumulative_sum[0]
    else:
        cum_v_value_min, cum_v_value_max = cumulative_sum[nearest_position -
                                                          1], cumulative_sum[nearest_position]

    # perform interpolation on a log-scale
    y = np.array([b_value_min, b_value_max])
    x = np.array([cum_v_value_min, cum_v_value_max])

    y = np.log2(y)
    q = np.interp(count_below_quantile, x, y)
    q = np.exp2(q)

    return q


def compute_mean_hist_data(histogram_counts, bins):
    """
    Compute the mean for binned data by taking
    a weighted average of midpoints of the bins

    Parameters
    ----------
      histogram_counts: array_like
        a vector of counts (or weights) of |bins|.
      bins: list
        list of (int, int) pairs specifying bin lower and
        upper edge

    Returns
    ----------
      estimate of the mean value
    """

    assert len(bins) == len(histogram_counts)

    # compute midpoints of each bin
    bin_midpoints = [np.mean(cur_bin) for cur_bin in bins]

    # compute weighted mean with weights being number of observations in a bin
    total_counts = sum(histogram_counts)
    if total_counts != 0:
        return sum((np.array(bin_midpoints) * histogram_counts)) / sum(histogram_counts)
    else:
        return 0


def get_exponential_bins(bin_min, bin_max, num_bins=100):
    """
    Function that computes boundaries used to bin the original data.
    Overflow and underflow bins are not included in the count for number of
    bins. So essentially there are number of bins specified + 2.

    Parameters
    ----------
      bin_min: 
        the minimum bin left boundary (except the underflow
        bin)
      bin_max
      maximum bin (except the overflow
        bucket).
      num_bins: The total number of bins (except for the underflow and the
        overflow bucket).

    Returns
    ----------
      The canonical exponential bin bounds for constructing histogram/binned data.
    """
    bin_min = int(bin_min) or 1
    # Do not add 0 as the underflow bin value.
    bin_bounds = [bin_min]
    for bin_index in range(2, num_bins):
        log_ratio = (math.log(bin_max) - math.log(bin_min)) / (
            num_bins - bin_index)
        log_next = math.log(bin_min) + log_ratio
        next_bin = int(math.exp(log_next) + 0.5)
        bin_min = next_bin if next_bin > bin_min else bin_min + 1
        bin_bounds.append(bin_min)
    # Add the overflow bin value.
    bin_bounds.append(GLOBAL_BIN_MAX)
    return np.array(bin_bounds)
