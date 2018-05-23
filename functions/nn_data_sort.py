import numpy as np
import math
import copy
import random
from pandas.core.indexes.datetimes import DatetimeIndex
from functions.scalers import scaler_dict

rnn_meths = ['fixed_at_end', 'random_choice', 'simple_block', 'fixed_val_bootstrap',
             'all_bootstrap', 'train_bootstrap', 'overlapping_block']


def data_checks(features_data, targets_data, dates=None, images=False):
    """
    Checks on the formats, sizes etc. of the data types passed into the functions below
    """
    if images:
        assert len(features_data.shape) == 4, 'Images should have 4 dimensions'
        len(targets_data.shape) == 2, 'targets data should have two dimensions'

    else:
        if features_data is not None:
            assert len(features_data.shape) == len(targets_data.shape) == 2, 'data should have two dimensions'
        else:
            assert len(targets_data.shape) == 2, 'data should have two dimensions'

    if features_data is not None:
        assert len(features_data) == len(targets_data), 'features and targets should be equal length. Shape features: {}, \
                                                        Shape targets: {}'.format(features_data.shape, targets_data.shape)
        assert type(features_data) == type(targets_data) == np.ndarray, 'data should be in np arrays'
    else:
        assert type(targets_data) == np.ndarray, 'data should be np array'

    if dates is not None:
        assert len(dates) == len(targets_data), 'Dates should have the same number of observations as targets data \
                                                 shape targets: {}, shape dates: {}'.format(targets_data.shape, dates.shape)
        assert type(dates) == DatetimeIndex, 'Dates should be a pandas datetime index'


def sizes_from_ratios(train_ratio, val_ratio, test_ratio, data_size, set_batch_size=False):
    """
    Given train, val and test ratios and data_size, return a consistent split of how many
    data points in each set.

    Args
    train/val/test_ratio: ratio of data in each set
    data_size: the number of data points
    set_batch_size: if not False, the number of data points needed in each batch (so excess data
                    will be wasted)

    Returns
    train_size, val_size, test_size: the integer sizes of the three data sets
    """
    assert train_ratio + val_ratio + test_ratio == 1, 'Ratios don\'t add up to 1'

    if set_batch_size:
        assert type(set_batch_size) == int, 'set_batch_size needs to be an integer'
        if (data_size % set_batch_size) != 0:
            data_size = data_size - (data_size % set_batch_size)

        train_size = int(math.ceil(train_ratio * data_size))
        val_size = int(math.ceil(val_ratio * data_size))
        train_size = int(set_batch_size * round(float(train_size) / set_batch_size))
        val_size = int(set_batch_size * round(float(val_size) / set_batch_size))
        test_size = int(data_size - train_size - val_size)

        if test_size == 0 or val_size == 0:
            raise Exception(
                'Test or val data set has no points in it with these ratios and batch size:'
                'probably need to lower batch_size')

    else:
        train_size = int(math.ceil(train_ratio * data_size))
        val_size = int(math.ceil(val_ratio * data_size))
        test_size = int(data_size - train_size - val_size)

    return train_size, val_size, test_size


def split_data_set(sizes, features_data, targets_data, method_choice, dates=None, shuffle=False,
                   include_tv=False):
    """
    Split the data_set into train, val and test sets

    Args
    sizes: tuple of format (train_size, val_size, test_size) with the sizes of each data set
    features_data: np array of shape [data_length, num_features]
    targets_data: np array of shape [data_length, num_targets]
    method_choice: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
            'fixed_val_bootstrap' - train data is drawn with replacement from train set only. Val data is
                                    fixed, as the last val_len observations in the set.
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    shuffle: if True, shuffles data before doing any draws, meaning test set will not be the last data etc.
    include_tv: whether to add a joint train val dataset in the data dict (useful if bootstrapping)
    Returns
    data_dict: dict holding train, val and test features and targets, plus the dates for each dataset if required
    """

    # Checks on formats
    data_checks(features_data, targets_data, dates)

    train_size = sizes[0]
    val_size = sizes[1]
    test_size = sizes[2]
    data_size = train_size + val_size + test_size
    tr_val_size = train_size + val_size

    # Shuffle data if required
    if shuffle:
        indices = np.random.permutation(data_size)
        if features_data is not None:
            features_data = features_data[indices]
        targets_data = targets_data[indices]
        if dates is not None:
            dates = dates[indices]

    tr_indices, val_indices = get_val_indices(tr_val_size, val_size, method_choice)

    if features_data is not None:
        train_features = features_data[tr_indices, :]
        val_features = features_data[val_indices, :]
        test_features = features_data[tr_val_size:data_size, :]
    else:
        train_features = val_features = test_features = None

    train_targets = targets_data[tr_indices, :]
    val_targets = targets_data[val_indices, :]
    test_targets = targets_data[tr_val_size:data_size, :]

    if include_tv:
        if features_data is not None:
            tv_features = features_data[0:tr_val_size, :]
        else:
            tv_features = None
        tv_targets = targets_data[0:tr_val_size, :]

    if dates is not None:
        train_dates = dates[tr_indices]
        val_dates = dates[val_indices]
        test_dates = dates[tr_val_size:data_size]

        if include_tv:
            tv_dates = dates[0:tr_val_size]

    data_dict = {'train_features': train_features,
                 'val_features': val_features,
                 'test_features': test_features,
                 'train_targets': train_targets,
                 'val_targets': val_targets,
                 'test_targets': test_targets}

    if dates is None:
        data_dict['train_dates'] = None
        data_dict['val_dates'] = None
        data_dict['test_dates'] = None
        if include_tv:
            data_dict['tv_dates'] = None
    else:
        data_dict['train_dates'] = train_dates
        data_dict['val_dates'] = val_dates
        data_dict['test_dates'] = test_dates
        if include_tv:
            data_dict['tv_dates'] = tv_dates

    if include_tv:
        data_dict['tv_features'] = tv_features
        data_dict['tv_targets'] = tv_targets

    return data_dict


def basic_split(features_data, targets_data, dates=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                shuffle=False, method='fixed_at_end', include_tv=False):
    """
    Simple helper function for splitting data into train, val and test sets.
    Args:
    features_data: np array of shape [data_size, features_dim]
    targets_data: np array of shape [data_size, targets_dim]
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    train/val/test_ratio: the proportion of the data in each set
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
            'fixed_val_bootstrap' - train data is drawn with replacement from train set only. Val data is
                                    fixed, as the last val_len observations in the set.
    shuffle: if True, shuffles data before doing any draws, meaning test set will not be the last data etc.
    include_tv: whether to add a joint train val dataset in the data dict (useful if bootstrapping)
    Returns
    data_dict: dict holding train_features, train_targets, val_features etc.
    """

    # Checks on format
    data_checks(features_data, targets_data, dates)

    data_size = len(targets_data)

    # Get the sizes of each data set
    sizes = sizes_from_ratios(train_ratio, val_ratio, test_ratio, data_size)

    # Split the data into train, val and test sets
    data_dict = split_data_set(sizes, features_data, targets_data, method_choice=method,
                               dates=dates, shuffle=shuffle, include_tv=include_tv)

    return data_dict


def scale_data_dict(d_d, f_sc, t_sc, var_network=False):
    """
    Create the scaler functions, and scale the relevant data in the data dict
    Args
    data_dict: dict with keys 'train_targets', 'train_features', 'val_targets' etc. holding np arrays of the data
    f_sc: the feature scaling function
    t_sc: the target scaling function
    Returns
    data_dict: as above, but with the data scaled
    sc_dict: dict holding the feature_scaler (the feature scaling function for the model to use) and the
             rev_target_scaler (the function for reversing the scaling of the targets)
    """
    data_dict = copy.deepcopy(d_d)

    if f_sc is not None and d_d['train_features'] is not None:
        sc_features = np.concatenate([data_dict['train_features'], data_dict['val_features']], axis=0)
        feature_scaler, _ = f_sc(sc_features)
    else:
        feature_scaler = None
    if t_sc is not None:
        sc_targets = np.concatenate([data_dict['train_targets'], data_dict['val_targets']], axis=0)
        if var_network:
            target_scaler, rev_target_scaler, rev_var_scaler = t_sc(sc_targets, var_network=var_network)
        else:
            target_scaler, rev_target_scaler = t_sc(sc_targets)
            rev_var_scaler = None

    else:
        rev_target_scaler = None
        target_scaler = None
        rev_var_scaler = None

    keys = ['train', 'val', 'test']
    if 'tv_features' in list(data_dict.keys()):
        keys += ['tv']

    for key in keys:
        if feature_scaler is not None:
            data_dict[key + '_features'] = feature_scaler(data_dict[key + '_features'])
        if target_scaler is not None:
            data_dict[key + '_targets'] = target_scaler(data_dict[key + '_targets'])

    sc_dict = {'feature_scaler': feature_scaler,
               'rev_target_scaler': rev_target_scaler,
               'target_scaler': target_scaler,
               'rev_var_scaler': rev_var_scaler}

    return data_dict, sc_dict


def batch_sorter(features_data, targets_data, dates=None, batch_size=100, shuffle=False,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, feat_scaler=None,
                 targ_scaler=None, method='fixed_at_end', include_tv=False, return_data=False,
                 num_steps=None, var_network=False, features_lags=None, targets_lags=None):
    """
    Splits data into batches, scales, and loads into iterators ready for training

    Args
    features_data: np array of shape [data_size, features_dim]
    targets_data: np array of shape [data_size, targets_dim]
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    batch_size: the batch size used during training
    train/val/test_ratio: the proportion of the data in each set
    feat_scaler: scaler function for the features
    targ_scaler: scaler function for the targets
    shuffle: if True, shuffles all data before starting - means test data won't be last in the set etc.
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
            'fixed_val_bootstrap' - train data is drawn with replacement from train set only. Val data is
                                    fixed, as the last val_len observations in the set.
    include_tv: whether to add a joint train val dataset iterator (useful if bootstrapping)
    return_data: if True, return the data (unscaled) used for this NN
    Returns
    iter_dict: dict of format {'train': train_iterator, 'val': val_iterator, 'test':test_iterator}
    scale_dict: dict of format {'feature_scaler': feat_scaler_func, 'rev_target_scaler', rev_target_scaler_func}
                The scaler funcs are used by the model for scaling in np.
    """
    # Checks on format
    data_checks(features_data, targets_data, dates)

    data_size = len(targets_data)

    # Get the sizes of each data set
    sizes = sizes_from_ratios(train_ratio, val_ratio, test_ratio, data_size)

    # Split the data into train, val and test sets
    old_data_dict = split_data_set(sizes, features_data, targets_data, method_choice=method,
                                   dates=dates, shuffle=shuffle, include_tv=include_tv)

    print("Train data: {} observations".format(len(old_data_dict['train_features'])))
    print("Val data: {} observations".format(len(old_data_dict['val_features'])))
    print("Test data: {} observations\n".format(len(old_data_dict['test_features'])))

    # Scale data if required (getting mean and var from train and val set only - so this has
    # to be done after splitting the data, in case random_draw=True)
    data_dict, scaler_dict = scale_data_dict(old_data_dict, feat_scaler, targ_scaler, var_network)

    # Create the iterators
    train_iter = batch_iterator(data_dict['train_features'], data_dict['train_targets'], data_dict['train_dates'],
                                batch_size)
    val_iter = batch_iterator(data_dict['val_features'], data_dict['val_targets'], data_dict['val_dates'],
                              batch_size)
    test_iter = batch_iterator(data_dict['test_features'], data_dict['test_targets'], data_dict['test_dates'],
                               batch_size)

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter}

    if include_tv:
        tv_iter = batch_iterator(data_dict['tv_features'], data_dict['tv_targets'], data_dict['tv_dates'],
                                 batch_size)
        iter_dict['train_val'] = tv_iter

    if return_data:
        return iter_dict, scaler_dict, old_data_dict
    else:
        return iter_dict, scaler_dict


class batch_iterator(object):
    """
    Class which returns next batch of features and targets when self.next_batch() called

    Args
    features_data: np array of shape [data_size, features_dim]
    targets_data: np array of shape [data_size, targets_dim]
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    batch_size: the batch size used for training
    shuffle: if True, data is shuffled randomly before training begins
    """

    def __init__(self, features_data, targets_data, dates=None, batch_size=50,
                 shuffle=False, weightings=None, images=False):

        self.features_data = features_data
        self.targets_data = targets_data
        self.dates = dates
        self.batch_size = batch_size
        self.weightings = weightings
        self.images = images

        self.check_format(images=images)

        self.num_data_points = features_data.shape[0]

        if shuffle:
            indices = np.random.permutation(self.num_data_points)
            self.targets_data = self.targets_data[indices]
            self.features_data = self.features_data[indices]

            if weightings is not None:
                self.weightings = self.weightings[indices]

        self.counter = 0
        self.num_batches = int(math.ceil(float(self.num_data_points / float(batch_size))))

        self.features_dim = features_data.shape[1]
        self.targets_dim = targets_data.shape[1]

    def next_batch(self):
        # Return next batch of features and targets data
        targets_batch = self.new_batch(self.counter, self.targets_data)
        features_batch = self.new_batch(self.counter, self.features_data)

        if self.weightings is not None:
            weightings_batch = self.new_batch(self.counter, self.weightings)

        self.counter += 1

        if self.counter == self.num_batches:
            self.counter = 0

        if self.weightings is not None:
            return features_batch, targets_batch, weightings_batch

        else:
            return features_batch, targets_batch

    def new_batch(self, counter, data):
        # Return batch of data
        try:
            new_batch = data[counter * self.batch_size: (counter * self.batch_size) + self.batch_size]
        except:
            new_batch = data[counter * self.batch_size:]

        return new_batch

    def check_format(self, images=False):
        # Some checks on format
        data_checks(self.features_data, self.targets_data, self.dates, images=images)

        if self.weightings is not None:
            assert len(self.weightings.shape) == 2, 'weightings should have two dimensions'
            assert len(self.weightings) == len(self.features_data), 'weightings should be same lengths as features'
            assert type(self.weightings) == np.ndarray, 'weightings should be np array'

    def sample_targets(self, sample_size):
        # Random sample of the targets data used for GANs
        sample_indices = random.sample(list(range(self.num_data_points)), sample_size)
        sample = self.targets_data[sample_indices]
        return sample

    def sample_features(self, sample_size, index='random'):
        # Return the same set of features tiled, so can draw from distribution for the next prediction
        # Can define which features to return using the index input if required. Used for GANs
        if index == 'random':
            index = random.randint(0, len(self.features_data) - 1)
        sample = self.features_data[index]
        sample = np.tile(sample, [sample_size, 1])
        return sample

    def all_targets(self, resc=None):
        if resc:
            ts = resc(self.targets_data)
        else:
            ts = self.targets_data
        return ts


# RNN ###############################################################################################################


def add_lags(features_data=None, targets_data=None, features_lags=None, targets_lags=None,
             step_ahead=1, return_max=False):
    """
    Generates lags to be used as features for a NN model from both the feature and target data
    Args
    features/targets_data: 2d np arrays holding the features and targets data respectively
    features/targets_lags: dicts, where the keys are indices to select the columns, and the
                           values are tuples, holding the lag values to use for that column
    step_ahead: the number of steps ahead we are forecasting (directly) - lag values must all
                be greater than this
    return_max: if True, return the max number of lags used, which will also be the amount by
                which we have reduced the size of the data
    Returns
    features: 2d np array of features for the network (made up of lags)
    targets: 2d np array of targets (same as targets_data, but with the correct number of values
             removed from the beginning)
    """

    # Checks on the data
    if features_data is not None:
        assert len(features_data) == len(targets_data), 'Features and targets should be same length'
        assert len(features_data.shape) == len(targets_data.shape) == 2, 'Features and targets should be 2D arrays'

    # Function for checking the layout of the lags dicts
    def params_check(lags_dict):
        assert type(lags_dict) == dict, 'Features/targets lags should be a dict holding tuples'
        for f in list(lags_dict.keys()):
            l = lags_dict[f]
            assert type(l) == tuple, 'Features/targets lags should be a dict of tuples - if calling with single value' \
                                     'make sure put a comma after e.g. (1,)'
            assert step_ahead <= max(
                l), 'Lag values should be greater than the number of steps ahead we are forecasting'

    # Checks on the lags parameters
    if features_lags:
        params_check(features_lags)
    if targets_lags:
        params_check(targets_lags)

    # Get the maximum lag value which will be used
    max_f_lags = max_t_lags = 0
    if features_lags:
        max_f_lags = max([max(l) for l in list(features_lags.values())])
    if targets_lags:
        max_t_lags = max([max(l) for l in list(targets_lags.values())])
    max_lag = max(max_f_lags, max_t_lags)

    # Function for generating the lags
    def fill_f(lags, data):
        # Get the total number of explanatory vars
        num_feats = sum([len(f) for f in list(lags.values())])
        f_mat = np.zeros([len(targets_data) - max_lag, num_feats])
        col = 0
        for key in list(lags.keys()):
            for i in lags[key]:
                feature = data[max_lag - i:-i, key]
                f_mat[:, col] = feature
                col += 1
        return f_mat, num_feats

    # Get the lags from both the features and targets arrays
    num_t = num_f = 0
    if features_lags:
        f_f, num_f = fill_f(features_lags, features_data)
    if targets_lags:
        f_t, num_t = fill_f(targets_lags, targets_data)

    # Return the lags in a single features array
    if num_t != 0 and num_f != 0:
        features = np.concatenate([f_t, f_f], axis=1)
    elif num_t != 0:
        features = f_t
    elif num_f != 0:
        features = f_f
    else:
        raise Exception('Need to return some features for prediction!')

    # Truncate the targets data, as we have used early values as features
    targets = targets_data[max_lag:, :]

    if return_max:
        return features, targets, max_lag
    else:
        return features, targets


def dict_lags(d_d, features_lags, targets_lags, step_ahead=1):
    """
    Add lags of the data. We generally run this function after data scaling, to ensure that
    lags of the target values are scaled in the same way as the targets. This helps RNNs use the
    target lags effectively
    Arg
    d_d: a data dict of standard format
    features_lags: dict where the keys are the column indices, and the values are lists holding
                   the lags to add for this column
    targets_lags: same as above but for targets
    step_ahead: how many steps ahead we are forecasting. Lag values must all be greater than this
    Returns
    d_d: data dict of standard format, with the lagged values added in
    """

    # Join the data together before lag creation
    if d_d['train_features'] is not None:
        f = np.concatenate([d_d['train_features'], d_d['val_features'], d_d['test_features']],
                           axis=0)
    else:
        f = None
    t = np.concatenate([d_d['train_targets'], d_d['val_targets'], d_d['test_targets']], axis=0)

    # Store original lengths of the separate datasets
    tl = len(d_d['train_targets'])
    vl = len(d_d['val_targets'])

    f, t, lags = add_lags(features_data=f, targets_data=t, features_lags=features_lags,
                          targets_lags=targets_lags, step_ahead=step_ahead, return_max=True)

    # Split data back up
    tl -= lags
    d_d['train_features'] = f[:tl, :]
    d_d['val_features'] = f[tl: tl + vl, :]
    d_d['test_features'] = f[tl + vl:, :]

    # Update the train_val data if needed
    if 'tv_features' in d_d.keys():
        d_d['tv_features'] = f[:tl + vl, :]
        d_d['tv_targets'] = d_d['tv_targets'][lags:, :]

    # Update the train targets as may have changed shape from adding lags
    d_d['train_targets'] = d_d['train_targets'][lags:, :]

    # Update the train dates
    if d_d['train_dates'] is not None:
        d_d['train_dates'] = d_d['train_dates'][lags:]

    # Add the lags to the data dict
    d_d['features_lags'] = features_lags
    d_d['targets_lags'] = targets_lags

    return d_d


def boot_lags(tr_f, tr_t, te_f, te_t, features_lags=None, targets_lags=None, step_ahead=1):
    """
    Add lags for bootstrap data. We generally run this function after data scaling, to ensure that
    lags of the target values are scaled in the same way as the targets. This helps RNNs use the
    target lags effectively
    Arg
    tr_f, tr_t, te_f, te_t: train features, train targets, test features and test targets respectively.
                            All should be 2D np arrays
    features_lags: dict where the keys are the column indices, and the values are lists holding
                   the lags to add for this column
    targets_lags: same as above but for targets
    step_ahead: how many steps ahead we are forecasting. Lag values must all be greater than this
    Returns
    tr_f, tr_t, te_f, te_t: train features, train targets, test features and test_targets in np arrays
    lags: the max number of lags used. Used for chopping off void data at start
    """
    # Concatenate data back together
    if tr_f is not None:
        f = np.concatenate([tr_f, te_f], axis=0)
    else:
        f = None
    t = np.concatenate([tr_t, te_t], axis=0)

    # Store original length of the datasets
    tl = len(tr_t)

    # Add the lags
    f, t, lags = add_lags(features_data=f, targets_data=t, features_lags=features_lags,
                          targets_lags=targets_lags, step_ahead=step_ahead, return_max=True)

    # Split data back up
    tl -= lags
    tr_f = f[:tl, :]
    te_f = f[tl:, :]

    # Cut off early targets which have been used as lags
    tr_t = tr_t[lags:, :]

    return tr_f, tr_t, te_f, te_t, lags


def sort_batches(data, min_batch_size, num_steps, dtype, t_or_f, silent=True,
                 allowed_waste=0.1):
    """
    Sorts batches in most efficient way for RNN
    data: shape [num_data_points, targets/features dimension]
    min_batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    dtype: one of [train, test, 'train_inference']
    t_or_f: one of [features, targets]

    Returns:
    batches: the data sorted into batches
    seq_lengths: the corresponding sequence lengths
    """

    num_data_points = data.shape[0]

    if dtype == 'train':
        # Calculate the batch size - basically keeping the same number of batches as would be required by
        # min_batch_size, but increasing the size to avoid wasting data
        batches_required = int(math.floor(num_data_points / float(min_batch_size * num_steps)))
        assert batches_required != 0, 'Train data doesn\'t even fill one batch - reduce batch size'
        b_length = int(math.floor(num_data_points / float(batches_required * num_steps)))

    elif dtype in ['val', 'test', 'train_inference', 'tr_val']:
        b_length = 1
        if num_data_points % (num_steps * b_length) != 0:
            num_zeros = int((num_steps * b_length) - (num_data_points % (num_steps * b_length)))
            # Use NaN's at this point, so no mistakes if any of features or targets are actually 0
            filler = np.empty(shape=[num_zeros, data.shape[1]])
            filler[:] = np.NAN
            data = np.concatenate([data, filler], axis=0)

    num_batches = int(len(data) / (b_length * num_steps))

    # If train data, get rid of datapoints which will be wasted from the start of the TS,
    # so don't end up not using those at the end (i.e. keep those closest to the val data)
    if dtype == 'train':
        used_data_points = num_batches * b_length * num_steps
        data = data[-used_data_points:, :]
        # If are wasting more than 10% of the data, raise an exception
        wasted_ratio = np.true_divide((num_data_points - used_data_points), num_data_points)
        assert wasted_ratio < allowed_waste, 'Wasting more than {:.2f}% of the data - lower the batch size'.format(
            allowed_waste*100)

    batches = []
    seq_lengths = []

    for i in range(num_batches):
        input_matrix = np.zeros((b_length, num_steps, data.shape[1]))
        for j in range(b_length):
            row_start_index = (0 + (num_steps * i)) + (j * num_batches * num_steps)
            input_matrix[j] = data[row_start_index:(row_start_index + num_steps)]

        # Find where the nan's are to calculate sequence length
        seq_len = np.ones([b_length]) * num_steps
        seq_len[-1] = num_steps - np.sum(np.isnan(input_matrix[-1, :, 0]))
        seq_lengths.append(seq_len)

        # Replace nan's with zeros
        input_matrix[np.isnan(input_matrix)] = 0
        batches.append(input_matrix)

    if not silent:
        print("{} {} {} batches created, of shape {} [batch_length * num_steps * num_features]".format(len(batches),
                                                                                                       dtype, t_or_f,
                                                                                                       batches[0].shape))

    return batches, seq_lengths


def prepare_rnn_data(features_data, targets_data, data_set, num_steps, min_batch_size=1,
                     silent=False, allowed_waste=0.1):
    """
    Sorts RNN data properly (i.e. maximises the number of states which are passed on correctly)

    features_data: shape [size_data * num_features]
    targets_data: shape [size_data * num_targets]
    data_set: one of ['train', 'val', 'test', 'train_inference', 'tr_val']
    num_steps: number of steps RNN rolled out for during training
    min_batch_size: batch_size for training data (for train_inf, val and test batch_size will be one)
    silent: boolean - whether or not to print the batch sizes created
    allowed_waste: if more than this ratio of train data is wasted due to having to fit into the
                   batch_size provided, raises as exception
    Returns
    features_batches: list of the features batches
    targets_batches: list of the targets batches
    seq_lenghts: list of sequence lengths associated with each batch
    """

    # Checks on formats
    data_checks(features_data, targets_data)

    features_batches, seq_lengths = sort_batches(features_data, min_batch_size, num_steps, data_set, 'features', silent=silent,
                                                 allowed_waste=allowed_waste)
    targets_batches, _ = sort_batches(targets_data, min_batch_size, num_steps, data_set, 'targets',
                                      silent=silent, allowed_waste=allowed_waste)

    return features_batches, targets_batches, seq_lengths


class rnn_batch_iterator(object):
    """
    RNN batch iterator - holds the data, and returns the next batch of features, targets, seq_lengths when
    next_batch() is called
    features_data: list of features batches
    targets_data: list of targets_batches
    seq_lengths: list of seq_lengths
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    masks: list of masks - used for masking out validation data during training
    train_len: optional arg which records the train length - useful when training multiple nets at once
    d_joined: if True, batches are shape [num_bootstraps, batch_length, num_steps, features/targets_dim]
              If False, batches are shape [batch_length, num_steps, features/targets_dim]
    """

    def __init__(self, features_data, targets_data, seq_lengths, dates=None, masks=None, train_len=None,
                 four_d=False):

        self.features_data = features_data
        self.targets_data = targets_data
        self.seq_lengths = seq_lengths
        self.dates = dates
        self.four_d = four_d

        if masks:
            self.masks = masks
        else:
            self.masks = None

        self.train_len = train_len

        self.counter = 0
        self.num_batches = len(targets_data)

        if self.four_d:
            assert len(targets_data[0].shape) == 4, 'Should be 4d batches if four_d is True'
            self.batch_size = targets_data[0].shape[1]
            self.targets_dim = targets_data[0].shape[3]
            self.features_dim = features_data[0].shape[3]
        else:
            assert len(targets_data[0].shape) == 3, 'Should be 3d batches if four_d is False'
            self.batch_size = targets_data[0].shape[0]
            self.targets_dim = targets_data[0].shape[2]
            self.features_dim = features_data[0].shape[2]

    def next_batch(self):

        features_batch = self.features_data[self.counter]
        targets_batch = self.targets_data[self.counter]
        seq_length = self.seq_lengths[self.counter]
        if self.masks:
            mask_batch = self.masks[self.counter]

        self.counter += 1

        if self.counter == self.num_batches:
            self.counter = 0

        if self.masks:
            return features_batch, targets_batch, seq_length, mask_batch
        else:
            return features_batch, targets_batch, seq_length

    def all_targets(self, resc):
        # Return all the targets - used in the bootstrap NNs for calculating loss
        if self.four_d:
            # If we're using this then will be for the train_val_iter or test_iter, where all targets
            # data is the same for each bootstrap - so just take one instance of data
            d = [i[0, :, :, :] for i in self.targets_data]
        else:
            d = self.targets_data

        assert d[0].shape[0] == 1, 'Should be batch length of 1 if getting all targets'
        all_ts = np.squeeze(np.concatenate(d, axis=1), axis=0)
        # Get rid of the 0s padded at the end if any
        all_ts = all_ts[:int(np.sum(self.seq_lengths))]
        if resc:
            all_ts = resc(all_ts)
        return all_ts


def rnn_batch_sorter(features_data, targets_data, features_lags=None, targets_lags=None,
                     dates=None, batch_size=20, train_ratio=0.7,
                     val_ratio=0.15, test_ratio=0.15, num_steps=5, silent=False,
                     allowed_waste=0.1, feat_scaler=None, targ_scaler=None, method='fixed_at_end',
                     include_tv=False, return_data=False, step_ahead=1):
    """
    Sorts data ready for training an RNN
    Args
    features_data: 2D np array of features data
    targets_data: 2D np array of targets data
    features_lags: dict where keys are column indices, and the values are lists of lags to add
                   for that feature
    targets_lags: as above, but with the lags for each of the targets
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    batch_size: the batch size used for training the RNN
    train_ratio: ratio of the dataset to be put in the training set
    val_ratio: ratio of the dataset to be put in the validation set
    test_ratio: ratio of the dataset to be put in the test set
    num_steps: the number of steps the RNN is rolled out during training
    silent: whether or not to print the sizes of batches created, and of the train/val/test sets
    allowed_waste: the maximum ratio of the train data we can discard to fit into batch sizes - if
                   discard more than this, raises an exception
    feat_scaler: scaler function for the features data
    targ_scaler: scaler function for the targets data
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
            'fixed_val_bootstrap' - train data is drawn with replacement from train set only. Val data is
                                    fixed, as the last val_len observations in the set.
    include_tv: whether to include a joint train val set - useful for evaluating if bootstrapping data
    return_data: if True, return the unscaled data used for this network, before put in RNN format
    Returns
    iter_dict: dictionary holding iterators with the train, val, test and train_inference data
    """

    # Checks on format
    data_checks(features_data, targets_data, dates=dates)

    # Check method is appropriate for RNN
    assert method in rnn_meths, 'Haven\'t implemented this method for RNN yet'

    # Get the sizes of the train, val and test sets
    sizes = sizes_from_ratios(train_ratio, val_ratio, test_ratio, len(targets_data),
                              set_batch_size=False)

    # Split the data into train, val and test sets
    s_d_old = split_data_set(sizes, features_data, targets_data, method_choice=method,
                             dates=dates, include_tv=include_tv)

    if not silent:
        print("Train data: {} observations".format(sizes[0]))
        print("Val data: {} observations".format(sizes[1]))
        print("Test data: {} observations\n".format(sizes[2]))

    # Scale data if required (getting mean and var from train and val set only - so this has
    # to be done after splitting the data, in case random_draw=True)
    s_d, scaler_dict = scale_data_dict(s_d_old, feat_scaler, targ_scaler)

    # Add lags
    s_d = dict_lags(s_d, features_lags=features_lags, targets_lags=targets_lags, step_ahead=step_ahead)

    # Sort for RNN
    train_f, train_t, train_seq_lengths = prepare_rnn_data(s_d['train_features'], s_d['train_targets'],
                                                           data_set='train', min_batch_size=batch_size,
                                                           num_steps=num_steps, silent=silent,
                                                           allowed_waste=allowed_waste)
    val_f, val_t, val_seq_lengths = prepare_rnn_data(s_d['val_features'], s_d['val_targets'], data_set='val',
                                                     min_batch_size=batch_size, num_steps=num_steps, silent=silent,
                                                     allowed_waste=allowed_waste)
    test_f, test_t, test_seq_lengths = prepare_rnn_data(s_d['test_features'], s_d['test_targets'], data_set='test',
                                                        min_batch_size=batch_size, num_steps=num_steps,
                                                        silent=silent,
                                                        allowed_waste=allowed_waste)
    trinf_f, trinf_t, trinf_seq_lengths = prepare_rnn_data(s_d['train_features'], s_d['train_targets'],
                                                           data_set='train_inference', min_batch_size=batch_size,
                                                           num_steps=num_steps, silent=silent,
                                                           allowed_waste=allowed_waste)

    # Create the iterators
    train_iter = rnn_batch_iterator(train_f, train_t, train_seq_lengths, s_d['train_dates'])
    val_iter = rnn_batch_iterator(val_f, val_t, val_seq_lengths, s_d['val_dates'])
    test_iter = rnn_batch_iterator(test_f, test_t, test_seq_lengths, s_d['test_dates'])

    tr_inf_iter = rnn_batch_iterator(trinf_f, trinf_t, trinf_seq_lengths, s_d['train_dates'])

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter, 'tr_inf': tr_inf_iter}

    if include_tv:
        tv_f, tv_t, tv_seq_lengths = prepare_rnn_data(s_d['tv_features'], s_d['tv_targets'],
                                                      data_set='train_inference', min_batch_size=batch_size,
                                                      num_steps=num_steps, silent=silent,
                                                      allowed_waste=allowed_waste)
        tv_iter = rnn_batch_iterator(tv_f, tv_t, tv_seq_lengths, s_d['tv_dates'])
        iter_dict['train_val'] = tv_iter

    if return_data:
        return iter_dict, scaler_dict, s_d_old
    else:
        return iter_dict, scaler_dict


# NN Bootstrap Version ##############################################################################################


def get_val_indices(t_v_size, v_size, method_choice):
    """
    Get the indices of the val data
    Args
    t_v_size: the joint size of the train and val data
    v_size: the size of the val data set we want
    method_choice: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
            'fixed_val_bootstrap' - train data is drawn with replacement from train set only. Val data is
                                    fixed, as the last val_len observations in the set.
            'all_bootstrap' - all data (both train and val together) is bootstrapped, then train and
                              val set chosen randomly from the new dataset
    Returns
    tr_indices: the indices of the train data
    v_indices: the indices of the val data
    """

    if method_choice in ['simple_block', 'fixed_at_end', 'random_choice']:

        if method_choice == 'simple_block':
            # Val data is in a continuous block, but this can start anywhere in the train/val set
            v_start_index = np.random.choice(np.arange(t_v_size - v_size))
            v_indices = list(np.arange(v_size) + v_start_index)

        elif method_choice == 'fixed_at_end':
            # Val data is always the last data in the train/val batch
            v_start_index = t_v_size - v_size
            v_indices = list(np.arange(v_size) + v_start_index)

        elif method_choice == 'random_choice':
            # Val data is randomly spread amongst the train data
            v_indices = random.sample(list(range(t_v_size)), v_size)
            v_indices.sort()

        t_v_set = set(range(t_v_size))
        v_set = set(v_indices)
        tr_indices = list(t_v_set - v_set)
        # By using set we lose the ordering, so regain this here
        tr_indices.sort()

    elif method_choice == 'fixed_val_bootstrap':
        # Bootstrap draws with replacement from train set only. Val set is fixed as the last observations
        tr_size = t_v_size - v_size
        tr_indices = list(np.random.choice(tr_size, size=tr_size, replace=True))
        v_indices = list(np.arange(v_size) + tr_size)

    elif method_choice == 'all_bootstrap':
        # All data is bootstrapped. Then train and val set chosen randomly. Note this means
        # we may have some overlap between train and val datasets
        new_indices = list(np.random.choice(t_v_size, size=t_v_size, replace=True))
        v_indices = list(random.sample(new_indices, v_size))
        v_temp = copy.deepcopy(v_indices)
        tr_indices = [i for i in new_indices if not i in v_temp or v_temp.remove(i)]

    elif method_choice == 'train_bootstrap':
        # Val data is chosen randomly. Then remaining data is bootstrapped. This ensures that
        # we both have a varying val dataset, and that there is no overlap between the train
        # and val sets
        v_indices = random.sample(list(range(t_v_size)), v_size)
        t_v_set = set(range(t_v_size))
        v_set = set(v_indices)
        tr_options = list(t_v_set - v_set)
        tr_indices = list(np.random.choice(tr_options, size=t_v_size - v_size, replace=True))

    else:
        raise Exception(method_choice, 'is not a recognised method')

    assert type(tr_indices) == type(v_indices) == list, 'Indices should be lists'

    return tr_indices, v_indices


def bootstrap_batch_sorter(features_data, targets_data, net_type, min_batch_size, dates=None, num_steps=5,
                           num_networks=100, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, silent=False,
                           method='simple_block', allowed_waste=0.1, feat_scaler=None, targ_scaler=None,
                           var_red_net=False, return_data=False, features_lags=None, targets_lags=None,
                           step_ahead=1):
    """
    Sorts NN/RNN data properly (i.e. maximises the number of states which are passed on correctly). Will split into
    train/val and test data first, and then put the data in iterators to pass to the NN/RNN

    features_data: shape [size_data, num_features]
    targets_data: shape [size_data, num_targets]
    net_type: one of ['NN', 'RNN']
    min_batch_size: minimum batch_size for data (for test batch_size will be one for RNN)
    dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    num_steps: number of steps RNN rolled out for during training
    num_networks: the number of networks being trained at once
    test_ratio: proportion of data for test set. Val proportion will be determined by bootstrap draws.
    val_ratio: proportion of the train/val dataset used for val data (only applies if method = simple_block)
    silent: whether or not to print batch_sizes
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
    allowed_waste: the maximum proportion of the train data we can get rid of to ensure a set train batch size
                   without raising an exception
    feat/targ_scaler: scaler functions for the features and targets respectively
    var_red_net: whether or not the network is running variance reduction
    return_data: whether or not to return the data (unscaled) as well as the iterators
    features_lags: dict where the keys are the column indices, and the values are lists holding
                   the lags to add for this column
    targets_lags: same as above but for targets
    step_ahead: the number of steps ahead we're forecasting for time series.

    Returns:
    A dictionary with iterators holding datasets
    """
    data_checks(features_data, targets_data, dates=dates)

    num_data_points = targets_data.shape[0]

    # Using this function ensures consistency of test size across all networks
    t, v, _ = sizes_from_ratios(train_ratio, val_ratio, test_ratio, num_data_points, set_batch_size=False)

    # Split the data into train and test set - for bootstrap NN the train set includes both train and val data
    train_size = t + v

    if features_data is not None:
        train_f_data = features_data[0: train_size, :]
        test_f_data = features_data[train_size:, :]
    else:
        train_f_data = None
        test_f_data = None

    train_t_data = targets_data[0: train_size, :]
    test_t_data = targets_data[train_size:, :]

    # Store the original data to return if required
    if net_type == 'RNN' and return_data:
        data_dict = {
            'train_val_features': copy.deepcopy(train_f_data),
            'train_val_targets': copy.deepcopy(train_t_data),
            'test_features': copy.deepcopy(test_f_data),
            'test_targets': copy.deepcopy(test_t_data),
        }

    if dates is not None:
        train_dates = dates[0: train_size]
        test_dates = dates[train_size:]
    else:
        train_dates = test_dates = None

    # Scaling
    if features_data is not None:
        if feat_scaler:
            feat_scaler = scaler_dict[feat_scaler]
            feature_scaler, rev_feature_scaler = feat_scaler(train_f_data)
            train_f_data = feature_scaler(train_f_data)
            test_f_data = feature_scaler(test_f_data)
        else:
            feature_scaler = rev_feature_scaler = None
    else:
        feature_scaler = rev_feature_scaler = None

    if targ_scaler:
        targ_scaler = scaler_dict[targ_scaler]
        target_scaler, rev_target_scaler = targ_scaler(train_t_data)
        train_t_data = target_scaler(train_t_data)
        test_t_data = target_scaler(test_t_data)
    else:
        target_scaler = rev_target_scaler = None

    scale_dict = {'feature_scaler': feature_scaler, 'rev_feature_scaler': rev_feature_scaler,
                  'rev_target_scaler': rev_target_scaler, 'target_scaler': target_scaler}

    if not silent:
        print("Train/ Val data: {} observations".format(train_t_data.shape[0]))
        print("Test data: {} observations\n".format(test_t_data.shape[0]))

    # Separate the block length and method type if using one of the block bootstrap methods
    if type(method) == tuple:
        block_len = method[1]
        method = method[0]
    else:
        block_len = None

    if net_type == 'NN':

        # Add lags if required
        if features_lags is not None or targets_lags is not None:
            train_f_data, train_t_data, test_f_data, test_t_data, l = boot_lags(train_f_data, train_t_data,
                                                                                test_f_data, test_t_data,
                                                                                features_lags=features_lags,
                                                                                targets_lags=targets_lags,
                                                                                step_ahead=step_ahead)

        # Chop off the start of the train dates if have used data for lags
        if train_dates is not None:
            train_dates = train_dates[l:]

        iter_dict, data_dict = bootstrap_nn_data(train_f_data, train_t_data, test_f_data, test_t_data,
                                                 train_dates=train_dates,
                                                 test_dates=test_dates, batch_size=min_batch_size,
                                                 num_networks=num_networks,
                                                 val_size=v, method=method, var_red_net=var_red_net, sc_d=scale_dict)

    elif net_type == 'RNN':
        assert features_lags is not None or targets_lags is not None, 'Should be using lags for ts data'
        iter_dict = bootstrap_rnn_data(train_f_data, train_t_data, test_f_data, test_t_data, train_dates=train_dates,
                                       test_dates=test_dates, min_batch_size=min_batch_size, num_steps=num_steps,
                                       val_size=v, num_networks=num_networks, silent=silent, method=method,
                                       allowed_waste=allowed_waste, block_len=block_len, features_lags=features_lags,
                                       targets_lags=targets_lags, step_ahead=step_ahead)

    if return_data:
        return iter_dict, scale_dict, data_dict
    else:
        return iter_dict, scale_dict


def bootstrap_nn_data(tr_features_data, tr_targets_data, test_features_data, test_targets_data, batch_size,
                      num_networks, val_size, method, train_dates=None, test_dates=None, var_red_net=False,
                      sc_d=None):
    """
    Bootstraps NN data. Returns the data in iterators to pass to the NN
    Args:
    tr_features_data: shape [size_train_data, num_features], includes both val and train data
    tr_targets_data: shape [size_train_data, num_targets], includes both val and train data
    test_features_data: shape [size_test_data, num_features]
    test_targets_data: shape [size_test_data, num_targets]
    batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_networks: the number of networks being trained at once
    val_size: number of data points in the val set
    silent: whether or not to print batch_sizes
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
    train_/test_dates: for time series data - the dates/times associated with the data points - for plotting later etc.
    var_red_net: whether or not the network is using variance reduction - if True, we return the indices of
                 each train batch in the train iterator

    Returns:
    A dictionary with iterators holding datasets
    """

    data_checks(tr_features_data, tr_targets_data, dates=train_dates)
    data_checks(test_features_data, test_targets_data, dates=test_dates)

    train_val_size = tr_targets_data.shape[0]

    train_f_list = []
    train_t_list = []

    val_f_list = []
    val_t_list = []

    # For bootstrapping will have different size val sets - need to keep track of this
    len_list = []
    train_i_list = []

    for b in range(num_networks):

        train_indices, val_indices = get_val_indices(train_val_size, val_size, method_choice=method)

        train_i_list.append(np.expand_dims(np.array(train_indices), axis=0))

        train_f_list.append(tr_features_data[train_indices, :])
        train_t_list.append(tr_targets_data[train_indices, :])

        val_f_list.append(tr_features_data[val_indices, :])
        val_t_list.append(tr_targets_data[val_indices, :])

        len_list.append(len(val_indices))


    assert len(set(len_list)) == 1, 'Should have equal sized val sets even if bootstrapping'

    # Concatenate the bootstraps together to give one matrix of size
    # [num_bootstraps * train/val_size * num_features/targets]

    def get_joined(data_list, re_sc):
        if re_sc is not None:
            rescaled = [re_sc(i) for i in data_list]
        else:
            rescaled = copy.deepcopy(data_list)
        expanded = np.concatenate([np.expand_dims(i, axis=0) for i in data_list], axis=0)
        exp_resc = np.concatenate([np.expand_dims(i, axis=0) for i in rescaled], axis=0)
        return expanded, exp_resc

    train_f_data, train_f_resc = get_joined(train_f_list, sc_d['rev_feature_scaler'])
    train_t_data, train_t_resc = get_joined(train_t_list, sc_d['rev_target_scaler'])
    val_f_data, val_f_resc = get_joined(val_f_list, sc_d['rev_feature_scaler'])
    val_t_data, val_t_resc = get_joined(val_t_list, sc_d['rev_target_scaler'])

    def get_test_joined(data, re_sc):
        if re_sc is not None:
            rescaled = re_sc(data)
        else:
            rescaled = copy.deepcopy(data)
        expanded = np.tile(np.expand_dims(data, axis=0), [num_networks, 1, 1])
        exp_resc = np.tile(np.expand_dims(rescaled, axis=0), [num_networks, 1, 1])
        return expanded, exp_resc

    test_f_data, test_f_resc = get_test_joined(test_features_data, sc_d['rev_feature_scaler'])
    test_t_data, test_t_resc = get_test_joined(test_targets_data, sc_d['rev_target_scaler'])

    tv_f_data, tv_f_resc = get_test_joined(tr_features_data, sc_d['rev_feature_scaler'])
    tv_t_data, tv_t_resc = get_test_joined(tr_targets_data, sc_d['rev_target_scaler'])

    if var_red_net:
        all_train_indices = np.concatenate(train_i_list, axis=0)
        train_iter = bootstrap_batch_iterator(train_f_data, train_t_data, batch_size=batch_size,
                                              indices=all_train_indices)

    else:
        # Create the iterators
        train_iter = bootstrap_batch_iterator(train_f_data, train_t_data, batch_size=batch_size)

    test_iter = bootstrap_batch_iterator(test_f_data, test_t_data, batch_size=batch_size, dates=test_dates)
    tv_iter = bootstrap_batch_iterator(tv_f_data, tv_t_data, batch_size=batch_size, dates=train_dates)

    val_iter = bootstrap_batch_iterator(val_f_data, val_t_data, batch_size=batch_size)

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter, 'tv': tv_iter}
    data_dict = {'train_features': train_f_resc, 'train_targets': train_t_resc,
                 'val_features': val_f_resc, 'val_targets': val_t_resc,
                 'test_features': test_f_resc, 'test_targets': test_t_resc,
                 'len_list': len_list}

    return iter_dict, data_dict


class bootstrap_batch_iterator(object):
    """
    Iterator class which returns next batch of data when self.next_batch() called
    features_data: shape [num_bootstrap * data_size * features_dim]
    targets_data: [num_bootstrap * data_size * targets_dim]
    dates: for time series data - np array of the dates/times associated with the data points - for plotting later etc.
    val_len: list of length num_bootstraps holding length of val data for each bootstrap draw
    batch_size: batch_size of data to return
    """

    def __init__(self, features_data, targets_data, dates=None, val_len=None, indices=None, batch_size=50):

        self.num_data_points = targets_data.shape[1]
        self.num_bootstraps = targets_data.shape[0]
        self.targets_data = targets_data
        self.features_data = features_data
        self.dates = dates
        self.batch_size = batch_size
        self.val_len = val_len
        self.indices = indices

        if self.val_len is not None:
            self.add_ones_data()

        self.counter = 0
        self.num_batches = int(math.ceil(float(self.num_data_points / float(batch_size))))

        self.targets_dim = targets_data.shape[2]
        self.features_dim = features_data.shape[2]

    def next_batch(self):
        """
        Returns next batch of features, targets and possibly val_indices when called
        """
        features_batch = self.new_batch(self.counter, self.features_data)
        targets_batch = self.new_batch(self.counter, self.targets_data)

        if self.val_len is not None:
            val_i_batch = self.new_i_batch(self.counter, self.ones_mat, val=True)
        if self.indices is not None:
            indices_batch = self.new_i_batch(self.counter, self.indices, val=False)

        self.counter += 1

        if self.counter == self.num_batches:
            self.counter = 0

        if self.val_len is not None and self.indices is not None:
            return features_batch, targets_batch, val_i_batch, indices_batch
        elif self.val_len is not None:
            return features_batch, targets_batch, val_i_batch
        elif self.indices is not None:
            return features_batch, targets_batch, indices_batch
        else:
            return features_batch, targets_batch

    def new_batch(self, counter, data):
        """
        Returns a new batch of data, based on current value of the iterator's counter
        """
        try:
            new_batch = data[:, counter * self.batch_size: (counter * self.batch_size) + self.batch_size, :]
        except:
            new_batch = data[:, counter * self.batch_size:, :]

        return new_batch

    def new_i_batch(self, counter, data, val=True):
        """
        Returns new batch of the val length data or indices
        """
        try:
            new_batch = data[:, counter * self.batch_size: (counter * self.batch_size) + self.batch_size]

        except:
            new_batch = data[:, counter * self.batch_size:]

        if val:
            return np.sum(new_batch, axis=1)
        else:
            return new_batch

    def add_ones_data(self):
        # Creates a mask array, with ones where there are val data points, and 0s where the val data has been
        # padded with zeros
        max_len = max(self.val_len)
        self.ones_mat = np.zeros([self.num_bootstraps, max_len])
        for num, i in enumerate(self.val_len):
            self.ones_mat[num, :i] = 1

    def all_targets(self, resc):
        # Return all the targets - used in the bootstrap NNs for calculating loss
        assert np.array_equal(self.targets_data[0, :, :], self.targets_data[1, :, :]), 'Data not identical across bootstraps'
        all_ts = self.targets_data[0, :, :]
        if resc:
            all_ts = resc(all_ts)
        return all_ts

    def all_features(self, resc):
        # Return all the features
        assert np.array_equal(self.features_data[0, :, :], self.features_data[1, :, :]), 'Data not identical across bootstraps'
        all_fs = self.features_data[0, :, :]
        if resc:
            all_fs = resc(all_fs)
        return all_fs

# RNN Bootstrap Version ##############################################################################################


def bootstrap_rnn_data(tr_features_data, tr_targets_data, test_features_data, test_targets_data, min_batch_size,
                       num_steps, val_size, features_lags, targets_lags, num_networks=100, train_dates=None, test_dates=None, silent=False,
                       method='simple_block', block_len=4, allowed_waste=0.1, step_ahead=1):
    """
    Bootstraps RNN data properly (i.e. maximises the number of states which are passed on correctly).
    Train/Test split already provided. Returns the data in iterators to pass to the RNN

    tr_features_data: shape [size_train_data, num_features], includes both val and train data
    tr_targets_data: shape [size_train_data, num_targets], includes both val and train data
    test_features_data: shape [size_test_data, num_features]
    test_targets_data: shape [size_test_data, num_targets]
    min_batch_size: minimum batch_size for training data (for test batch_size will be one)
    num_steps: number of steps RNN rolled out for during training
    num_networks: the number of networks being trained at once
    train_/test_dates: for time series data - np array with the dates/times associated with the data points - for plotting later etc.
    val_size: number of data points in the val data set
    silent: whether or not to print batch_sizes
    method: 'simple_block' - random val start index is chosen for each network, and the val data is a single
                             block from this point onwards
            'fixed_at_end' - the val data is fixed at the end of the train data, so is identical for each network
            'random_choice' - val data is randomly mixed amongst the train data, not in blocks
    block_len: the length of the blocks used if doing block bootstrapping
    allowed_waste: the maximum proportion of the train data we can get rid of to ensure a set train batch size
                   without raising an exception

    Returns:
    A dictionary with iterators holding datasets
    """
    data_checks(tr_features_data, tr_targets_data, dates=train_dates)
    data_checks(test_features_data, test_targets_data, dates=test_dates)

    assert method in rnn_meths, 'Haven\'t implemented this method with RNN yet'

    block_methods = ['simple_block', 'overlapping_block']

    # Run this before adding the lags, otherwise lags will be in the features array
    if method in block_methods:
        four_d = True
        # Get the train and val iterators from the block bootstrap function.
        iter_dict = block_bootstrap(tr_features_data, tr_targets_data, test_features_data,
                                    test_targets_data, method, num_networks,
                                    val_size, features_lags=features_lags, targets_lags=targets_lags,
                                    step_ahead=step_ahead, block_len=block_len, num_steps=num_steps,
                                    min_batch_size=min_batch_size, silent=silent, allowed_waste=allowed_waste)

    # Add lags if required
    tr_features_data, tr_targets_data, test_features_data, \
        test_targets_data, l = boot_lags(tr_features_data, tr_targets_data,
                                         test_features_data, test_targets_data,
                                         features_lags=features_lags,
                                         targets_lags=targets_lags,
                                         step_ahead=step_ahead)

    # Chop off the start of the train dates if have used data for lags
    if train_dates is not None:
        train_dates = train_dates[l:]

    # Sort data into batches
    ts = ['tr_val', 'test', 'train']
    d_d = {'test': (test_features_data, test_targets_data),
           'tr_val': (tr_features_data, tr_targets_data),
           'train': (tr_features_data, tr_targets_data)}
    b_d = {}

    for t in ts:
        f_batches, t_batches, seq = prepare_rnn_data(d_d[t][0], d_d[t][1], data_set=t,
                                                     num_steps=num_steps, min_batch_size=min_batch_size,
                                                     silent=silent, allowed_waste=allowed_waste)

        b_d[t] = {'f_batches': f_batches, 't_batches': t_batches, 'seq_lengths': seq}

    train_val_size = tr_targets_data.shape[0]

    if method not in block_methods:
        four_d = False
        iter_dict = {}
        # Make the masks for each network's draw
        mask_list = []
        val_mask_list = []

        def counts(indices):
            ix = list(set(sorted(indices)))
            counts = np.expand_dims(np.array([indices.count(i) for i in ix]), axis=1)
            return ix, counts

        for b in range(num_networks):
            train_mask = np.zeros([train_val_size, 1])
            val_mask = np.zeros([train_val_size, 1])

            train_indices, val_indices = get_val_indices(train_val_size, val_size, method_choice=method)

            tr_ix, tr_counts = counts(train_indices)
            v_ix, v_counts = counts(val_indices)

            train_mask[tr_ix] = tr_counts
            val_mask[v_ix] = v_counts

            mask_batches, _ = sort_batches(train_mask, min_batch_size=min_batch_size, num_steps=num_steps,
                                           dtype='train', t_or_f=None, silent=True, allowed_waste=allowed_waste)
            val_mask_batches, _ = sort_batches(val_mask, min_batch_size=min_batch_size, num_steps=num_steps,
                                               dtype='val', t_or_f=None, silent=True, allowed_waste=allowed_waste)

            mask_list.append(mask_batches)
            val_mask_list.append(val_mask_batches)

        def join_list(l):
            l = list(zip(*l))
            new_l = []
            for m in l:
                new_l.append(np.concatenate([np.expand_dims(i, 0) for i in m]))
            return new_l

        all_mask_list = join_list(mask_list)
        all_val_list = join_list(val_mask_list)

        # Iterator for training
        iter_dict['train'] = rnn_batch_iterator(b_d['train']['f_batches'], b_d['train']['t_batches'],
                                                b_d['train']['seq_lengths'], masks=all_mask_list)

        # Iterator for evaluation of val loss during training - same dataset, different masks
        iter_dict['val'] = rnn_batch_iterator(b_d['tr_val']['f_batches'], b_d['tr_val']['t_batches'],
                                              b_d['tr_val']['seq_lengths'], masks=all_val_list)

    # Expand the batches to shape [num_networks, b_length, num_steps, features/targets_dim]
    if method in block_methods:
        for k in ['tr_val', 'test']:
            for b in ['f_batches', 't_batches']:
                b_d[k][b] = expand_batches(b_d[k][b], num_networks=num_networks)

    # Masks are all ones, so we don't mask out any of the data points
    tr_val_masks = [np.ones([num_networks, 1, num_steps, 1]) for _ in b_d['tr_val']['t_batches']]
    test_masks = [np.ones([num_networks, 1, num_steps, 1]) for _ in b_d['test']['t_batches']]

    # Iterator for the train and validation data together (in correct order)
    train_size = train_val_size - val_size
    iter_dict['tr_val'] = rnn_batch_iterator(b_d['tr_val']['f_batches'], b_d['tr_val']['t_batches'],
                                             b_d['tr_val']['seq_lengths'], dates=train_dates,
                                             masks=tr_val_masks, train_len=train_size, four_d=four_d)

    # Iterator for test data
    iter_dict['test'] = rnn_batch_iterator(b_d['test']['f_batches'], b_d['test']['t_batches'],
                                           b_d['test']['seq_lengths'], dates=test_dates,
                                           masks=test_masks, four_d=four_d)

    return iter_dict


def expand_batches(batch_list, num_networks):
    # Expands RNN batches by tiling from 3d to 4d, giving final shape:
    # [num_networks, b_length, num_steps, features/targets_dim]
    exp = [np.expand_dims(i, axis=0) for i in batch_list]
    return [np.tile(i, [num_networks, 1, 1, 1]) for i in exp]


def block_bootstrap(features, targets, test_features, test_targets,
                    method, num_bootstraps, v_size,
                    block_len, features_lags, targets_lags, step_ahead=1,
                    num_steps=5, min_batch_size=3, silent=True, allowed_waste=1.0):
    """
    Carries out block bootstrapping for time series
    features: np array of shape [num_obs, num_features]
    targets: np array of shape [num_obs, num_targets]
    test_features/targets: np arrays of shape [num_test_obs, num_features/targets] - only
                           used to pass into the lag generating function
    method: one of ['simple_block', 'overlapping_ block'] - the type of bootstrapping
    num_bootstraps: how many bootstrap we're carrying out at once
    v_size: the size of the val data set required
    block_len: the length of the bootstrap blocks
    features_lags: dict of lags for features data where keys are indices and values
                   are tuples of lags for that column e.g. {1: (1,3)}
    targets_lags: same as features_lags but for the targets data
    step_ahead: the number of steps ahead we're forecasting
    num_steps: how many steps we're rolling out training for
    min_batch_size: the batch size
    silent: boolean, whether or not to print the batch sizes created
    allowed_waste: maximum percentage of train data we can discard to get evenly sized batches
    Returns
    iter_dict: dictionary holding iterators for the train and validation data
    """

    ns = ['train', 'val']
    dd = {}
    data = {}
    iter_dict = {}
    for n in ns:
        dd[n] = {'f_draws': [], 't_draws': []}
        data[n] = {}

    for b in range(num_bootstraps):

        # Get the val data as a continuous block
        tr_indices, v_indices = get_val_indices(len(targets), v_size, method_choice='simple_block')
        if features is not None:
            val_features_data = features[v_indices, :]
            t_feats = features[tr_indices, :]

        val_targets_data = targets[v_indices, :]
        t_targs = targets[tr_indices, :]

        ts_len = len(t_targs)

        # Split the train data into blocks
        if method == 'simple_block':
            num_blocks = math.ceil(ts_len / block_len)
            if features is not None:
                f_blocks = [t_feats[i * block_len:i * block_len + block_len] for i in range(num_blocks)]
            t_blocks = [t_targs[i * block_len:i * block_len + block_len] for i in range(num_blocks)]

        if method == 'overlapping_block':
            num_blocks = ts_len - block_len + 1
            if features is not None:
                f_blocks = [t_feats[i:i + block_len] for i in range(num_blocks)]
            t_blocks = [t_targs[i:i + block_len] for i in range(num_blocks)]

        # Randomly sample the blocks with replacement to create num_bootstraps new train ts
        indices = []
        len_sample = 0

        while len_sample < ts_len:
            ch = random.choice(np.arange(num_blocks))
            len_sample += len(t_blocks[ch])
            indices.append(ch)

        if features is not None:
            train_features_data = np.concatenate([f_blocks[i] for i in indices], axis=0)[:ts_len]
        train_targets_data = np.concatenate([t_blocks[i] for i in indices], axis=0)[:ts_len]

        # Add the lags to the new time series - first need to join train and val data
        len_train = len(train_targets_data)
        if features is not None:
            all_f = np.concatenate([train_features_data, val_features_data], axis=0)
        else:
            all_f = None

        all_t = np.concatenate([train_targets_data, val_targets_data], axis=0)

        all_f, all_t, _, _, _ = boot_lags(all_f, all_t, test_features, test_targets,
                                          features_lags=features_lags, targets_lags=targets_lags,
                                          step_ahead=step_ahead)

        # Split back into train and val sets
        data['train']['features'] = all_f[:len_train, :]
        data['train']['targets'] = all_t[:len_train, :]
        data['val']['features'] = all_f[len_train:, :]
        data['val']['targets'] = all_t[len_train:, :]

        for n in ns:
            f_batches, t_batches, seq_lengths = prepare_rnn_data(data[n]['features'],
                                                                 data[n]['targets'],
                                                                 data_set=n,
                                                                 num_steps=num_steps,
                                                                 min_batch_size=min_batch_size,
                                                                 silent=silent, allowed_waste=allowed_waste)

            # Lists of batches, now each batch shape [1, batch_length, num_steps, num_features/targets]
            f_batches = [np.expand_dims(f, axis=0) for f in f_batches]
            t_batches = [np.expand_dims(t, axis=0) for t in t_batches]

            dd[n]['f_draws'].append(f_batches)
            dd[n]['t_draws'].append(t_batches)

            # All seq lengths will be the same so only need to store once
            if b == 0:
                dd[n]['seq_lengths'] = seq_lengths

    # Join all the bootstrap batches together - so we now have len(num_batches) list of batches,
    # each of which have shape [num_bootstraps, batch_length, num_steps, num_features/targets]
    for n in ns:
        feats = list(zip(*dd[n]['f_draws']))
        targs = list(zip(*dd[n]['t_draws']))
        feats = [np.concatenate(f, axis=0) for f in feats]
        targs = [np.concatenate(t, axis=0) for t in targs]

        # Add masks of all ones, as we don't need to mask out any data
        masks = [np.ones([num_bootstraps, f.shape[1], num_steps, 1]) for f in feats]
        iter_dict[n] = rnn_batch_iterator(feats, targs, dd[n]['seq_lengths'], masks=masks, four_d=True)

    return iter_dict


def wild_boot(preds, data_dict, batch_size, num_networks, feat_scaler, targ_scaler,
              train_ratio, val_ratio, test_ratio, all_data):
    """
    Perform wild bootstrap
    Args:
    preds: preds dict from the previously optimized model, holding train_preds, val_preds etc.
    data_dict: dictionary holding the data used for the previously optimized model
    batch_size: the size of the batches for the network
    num_networks: how many networks will be trained - function will produce this many bootstrap
                  draws
    feat/targ_scaler: scaler functions for the features and targets respectively
    train/val/test_ratio: ratios for the datasets
    all_data: if True, then we bootstrap from the combination of the train and val sets first,
              and then split into new train/val. If false, keep the val set the same, and bootstrap
              only the train data
    Return
    iter_dict: dictionary holding the datasets in iterators ready for the NN
    scaler_dict: dictionary holding the scaler functions
    data_dict: dictionary holding the data draws
    """

    # Data split
    d_d = basic_split(data_dict['features'], data_dict['targets'], train_ratio=train_ratio, val_ratio=val_ratio,
                      test_ratio=test_ratio, method='fixed_at_end', include_tv=True)

    scaled_d, scaler_dict = scale_data_dict(d_d, feat_scaler, targ_scaler)

    # Train and val data used for the original model
    if all_data:
        old_p = np.concatenate([preds['train_preds'], preds['val_preds']], axis=0)
        old_t = np.concatenate([d_d['train_targets'], d_d['val_targets']], axis=0)
        old_f = np.concatenate([d_d['train_features'], d_d['val_features']], axis=0)
        old_t_size = len(preds['train_preds'])
    # Or if just bootstrapping the train data and keeping val fixed
    else:
        old_p = preds['train_preds']
        old_t = d_d['train_targets']
        old_f = d_d['train_features']

    tr_size = len(old_t)

    # Get the residuals and center them
    old_res = old_t - old_p
    old_res = old_res - np.average(old_res)

    # Create the new datasets
    new_features = []
    new_targets = []
    for n in range(num_networks):
        # Random indices denoting data points to draw for this set
        inds = list(np.random.choice(tr_size, size=tr_size, replace=True))
        # Generate the lambda values with mean 0 and var 1
        lambdas = np.random.normal(loc=0, scale=1, size=[tr_size, 1])
        # Get the res, features and preds corresponding to the indices
        new_res = np.concatenate([np.expand_dims(old_res[d, :], axis=0) for d in inds], axis=0)
        new_f = np.concatenate([np.expand_dims(old_f[d, :], axis=0) for d in inds], axis=0)
        new_p = np.concatenate([np.expand_dims(old_p[d, :], axis=0) for d in inds], axis=0)

        new_res = new_res * lambdas
        new_t = new_p + new_res

        # Scale the new features and targets
        new_f = scaler_dict['feature_scaler'](new_f)
        new_t = scaler_dict['target_scaler'](new_t)

        new_features.append(new_f)
        new_targets.append(new_t)

    # Add the original val and test data into new data dict
    new_d = {}
    keys = ['test', 'tv']
    if not all_data:
        keys += ['val']
    for key in keys:
        for dtype in ['features', 'targets']:
            k = key + '_' + dtype
            old_d = scaled_d[k]
            new_d[k] = np.tile(np.expand_dims(old_d, axis=0), [num_networks, 1, 1])

    # Add the new train data
    n_f = np.concatenate([np.expand_dims(i, axis=0) for i in new_features], axis=0)
    n_t = np.concatenate([np.expand_dims(i, axis=0) for i in new_targets], axis=0)

    if all_data:
        new_d['train_features'] = n_f[:, :old_t_size, :]
        new_d['train_targets'] = n_t[:, :old_t_size, :]
        new_d['val_features'] = n_f[:, old_t_size:, :]
        new_d['val_targets'] = n_t[:, old_t_size:, :]
    else:
        new_d['train_features'] = n_f
        new_d['train_targets'] = n_t

    # Put the data in iterators
    i_dict = {}
    for key in ['train', 'val', 'test', 'tv']:
        it = bootstrap_batch_iterator(new_d[key + '_features'], new_d[key + '_targets'], batch_size=batch_size)
        i_dict[key] = it

    return i_dict, scaler_dict, new_d
