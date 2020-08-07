from scipy.stats import t
import numpy as np
import pandas as pd
import math
import pdb

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

    # pdb.set_trace()

    # compute cumulative distirubtion
    cumulative_sum = histogram_counts.cumsum()

    # compute number of bins
    num_of_bins = len(histogram_counts)
    assert num_of_bins == len(bins)

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

    if nearest_position == 0:
        # if the quantile falls into underflow bin
        # perform interpolation on a linear scale
        y = np.array([b_value_min, b_value_max])
        x = np.array([cum_v_value_min, cum_v_value_max])
        q = np.interp(count_below_quantile, x, y)
        return q
    elif nearest_position == num_of_bins-1:
        # if the quantile falls into the overflow bucket
        # return the left-endpoint of the last bin
        return bins[nearest_position][0]
    else:
        # perform interpolation on a log-scale
        y = np.array([b_value_min, b_value_max])
        x = np.array([cum_v_value_min, cum_v_value_max])

        y = np.log2(y)
        q = np.interp(count_below_quantile, x, y)
        q = np.exp2(q)

        return q


def alternative_quantile_hist_data(histogram_counts, bins,
                                   quantile, midpoints_comp='linear'):
    """
    Compute the quantile based using frequency polygon method

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

    # normalizing constant
    total_counts = sum(histogram_counts)

    # get normalized counts in each bin
    normed_counts = histogram_counts / total_counts

    # get sizes of each bin
    bin_sizes = np.array([abs(cur_pair[1]-cur_pair[0]) for cur_pair in bins])

    # get density estimates

    density_estimates = normed_counts / bin_sizes

    # add preceeding and following empty bins
    density_estimates = np.concatenate(
        [np.array([0]), density_estimates, np.array([0])])

    # compute difference in densities
    count_differences = np.diff(density_estimates)

    # number of bins
    num_of_bins = len(histogram_counts)

    # compute midpoints of bins
    if midpoints_comp is 'linear':
        bin_midpoints = [np.mean(cur_pair) for cur_pair in bins]
        # add two extra midpoints
        lowest_bin_left_boundary = bins[0][0]
        largest_bin_right_boundary = bins[-1][1]
        # add left-most point
        bin_midpoints = [2 * lowest_bin_left_boundary -
                         bin_midpoints[0]] + bin_midpoints
        # add right-most point
        bin_midpoints = bin_midpoints + \
            [2 * largest_bin_right_boundary - bin_midpoints[-1]]
    elif midpoints_comp is 'log':
        bin_midpoints = list()
        # first bin analyzed separately
        bin_midpoints += [np.mean(bins[0])]

        # for other bins except the last one find the midpoint on a log-scale
        # which results in geometric mean
        bin_midpoints += [np.sqrt(cur_pair[0]*cur_pair[1])
                          for cur_pair in bins[1:-1]]
        # the last midpoint should use linear scale
        bin_midpoints += [np.mean(bins[-1])]
        # add two extra midpoints
        lowest_bin_left_boundary = bins[0][0]
        largest_bin_right_boundary = bins[-1][1]
        # add left-most point
        bin_midpoints = [2 * lowest_bin_left_boundary -
                         bin_midpoints[0]] + bin_midpoints
        # add right-most point
        bin_midpoints = bin_midpoints + \
            [2 * largest_bin_right_boundary - bin_midpoints[-1]]

    # compute distance between midpoints
    midpoints_dist = np.diff(bin_midpoints)

    # compute slopes and intercepts for linear approximation
    slopes = count_differences / midpoints_dist
    intercepts = density_estimates[1:] - slopes * bin_midpoints[1:]

    assert len(slopes) == len(intercepts)

    # compute ECDF at breakpoints
    # at first breakpoint the value is zero
    cdf_vals_at_breakpoints = [0]
    for cur_breakpoint in range(num_of_bins+1):
        cdf_vals_at_breakpoints += [cdf_vals_at_breakpoints[cur_breakpoint] + intercepts[cur_breakpoint] * (
            bin_midpoints[cur_breakpoint + 1] - bin_midpoints[cur_breakpoint]) + slopes[cur_breakpoint] / 2 * (
            bin_midpoints[cur_breakpoint + 1] ** 2 - bin_midpoints[cur_breakpoint] ** 2)]

    cdf_vals_at_breakpoints = np.array(cdf_vals_at_breakpoints)

    # pdb.set_trace()

    # find the index of the correponding bin
    nearest_position = cdf_vals_at_breakpoints.searchsorted(quantile)

    F_hat = cdf_vals_at_breakpoints[nearest_position-1]

    midpoint = bin_midpoints[nearest_position]
    prev_midpoint = bin_midpoints[nearest_position - 1]

    cand_x = (-intercepts[nearest_position-1] + math.sqrt(2 * slopes[nearest_position-1] * (quantile - F_hat) +
                                                          (intercepts[nearest_position-1] +
                                                           slopes[nearest_position-1] * prev_midpoint) ** 2))/slopes[nearest_position-1]

    assert cand_x >= prev_midpoint
    assert cand_x <= midpoint
    return cand_x


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
        the minimum bin left boundary (including the underflow
        bin)
      bin_max
      maximum bin (including the overflow
        bin).
      num_bins: The total number of bins (including the underflow and the
        overflow bins).

    Returns
    ----------
      The canonical exponential bin bounds for constructing histogram/binned data.
    """
    bin_min = int(bin_min) or 1
    # Do not add 0 as the underflow bin value.
    bin_bounds = [0, bin_min]
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


def compute_interval_score(u, l, alpha, x):
    """
    Function that computes the interval score
    """
    return (u-l) + 2/alpha * (l-x) * (x < l) + 2/alpha * (x-u)*(x > u)
