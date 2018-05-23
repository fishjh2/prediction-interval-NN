import numpy as np


def std_scale(data, var_network=False):
    shape = data_check(data)
    # Scale data by subtracting mean and dividing by std
    mean = np.expand_dims(np.average(data, axis=0), 0)
    std = np.expand_dims(np.std(data, axis=0), 0)

    def scale(ds):
        data_check(ds, shape)
        return np.divide((ds - mean), std)

    def rev_scale(d):
        data_check(d, shape)
        return d * std + mean

    # Add a reverse scaler for the variance predictions
    if var_network:
        def rev_var_scale(d):
            data_check(d, shape)
            return d * (std**2)
        return scale, rev_scale, rev_var_scale
    else:
        return scale, rev_scale


def zero_one_scale(data):
    shape = data_check(data)
    # Scale all features between 0 and 1
    min_vals = np.expand_dims(np.amin(data, axis=0), 0)
    max_vals = np.expand_dims(np.amax(data, axis=0), 0)

    def scale(ds):
        data_check(ds, shape)
        assert ds.shape[1] == shape
        return np.true_divide((ds - min_vals), (max_vals - min_vals))

    def rev_scale(d):
        data_check(d, shape)
        return (d * (max_vals - min_vals)) + min_vals

    return scale, rev_scale


def range_scale(data, bottom=-1, top=1):
    """
    Scales data between bottom and top
    """
    shape = data_check(data)
    # Scale all features between 0 and 1
    min_vals = np.expand_dims(np.amin(data, axis=0), 0)
    max_vals = np.expand_dims(np.amax(data, axis=0), 0)
    r = top - bottom

    def scale(ds):
        data_check(ds, shape)
        s1 = np.true_divide((ds - min_vals), (max_vals - min_vals))
        return s1 * r + bottom

    def rev_scale(d):
        data_check(d, shape)
        s1 = np.true_divide((d - bottom), r)
        return (s1 * (max_vals - min_vals)) + min_vals

    return scale, rev_scale


def data_check(data, shape=None):
    # Checks data format
    assert len(data.shape) == 2, 'Data should have two dimensions'
    assert type(data) == np.ndarray, 'Data should be a np array'

    if shape is not None:
        assert shape == data.shape[1], 'Data being scaled does not have same sized 2nd dimension as original data'

    return data.shape[1]


def scaler_check(scale, rev_scale, data):
    # Check scalers work
    assert np.array_equal(np.round(rev_scale(scale(data)), 10),
                          np.round(data, 10)), 'Scaler or reverse scaler incorrect'


scaler_dict = {'std_scale': std_scale,
               'zero_one_scale': zero_one_scale,
               'minus_one_scale': range_scale,
               'none': None
               }

