import numpy as np
import matplotlib.pyplot as plt


def get_stats(p, percentiles):
    # Get the standard deviation and percentile estimates for each point in the feature grid
    stdevs = np.expand_dims(np.std(p, axis=1), axis=1)
    bottom = np.expand_dims(np.percentile(p, q=percentiles[0], axis=1), axis=1)
    top = np.expand_dims(np.percentile(p, q=percentiles[1], axis=1), axis=1)
    return stdevs, bottom, top


def get_intervals(preds, confidence_percentiles=(5.0, 95.0),
                  prediction_percentiles=(5.0, 95.0)):
    """
    Calculate associated prediction and confidence intervals after training network
    preds: dict, the output of network.predict()
    confidence_percentiles: the percentiles which represent the top and bottom of the confidence
                            interval
    prediction_percentiles: the top and bottom of the predicition interval. These quantiles
                            must have been predicted by the network
    """

    intervals = {'prediction': {}, 'confidence': {}}

    ps = np.concatenate(preds['main'], axis=1)
    _, boot_bottom, boot_top = get_stats(ps, percentiles=confidence_percentiles)
    intervals['confidence'] = {'top': boot_top, 'bottom': boot_bottom}

    pred_ints = []
    for p in prediction_percentiles:
        quant = np.round(p / 100, 2)
        ps = np.concatenate(preds[quant], axis=1)
        pred_ints.append(np.expand_dims(np.percentile(ps, q=p, axis=1), axis=1))

    intervals['prediction']['bottom'] = pred_ints[0]
    intervals['prediction']['top'] = pred_ints[1]

    return intervals


def sin_data(num_draws=500, heter=0.1, std=1):
    """
    Simulate data from function based on sin curve
    """
    def curve(x):
        return 1.5 * np.sin(3 * x) + x

    def stdev(x):
        return float(x * heter)

    def noise(x):
        sc = stdev(x) * std
        return np.random.normal(loc=0.0, scale=sc)

    def function_gen(x):
        return curve(x) + noise(x)

    xs = []
    ys = []

    for _ in range(num_draws):
        x = np.random.uniform(low=2.0, high=6.0)
        xs.append(x)
        ys.append(function_gen(x))

    features = np.concatenate([np.expand_dims(f, axis=0) for f in xs], axis=0)
    targets = np.concatenate([np.expand_dims(f, axis=0) for f in ys], axis=0)

    features = np.expand_dims(features, axis=1)
    targets = np.expand_dims(targets, axis=1)

    return features, targets, curve


def plot_eg(features, targets, feature_grid, preds, intervals):
    """
    Example plot of the generated intervals
    """
    fig = plt.figure(figsize=[12, 8])
    ax = plt.axes()

    ax.plot(feature_grid, preds['ensemble']['main'], zorder=20, label='Average Prediction', color='#002d59')

    if features is not None:
        ax.plot(features, targets, 'x', color='#0066c8', zorder=10, label='Datapoints')
    else:
        ax.plot(feature_grid, targets, 'x', color='#0066c8', zorder=10, label='Datapoints')

    ax.fill_between(np.squeeze(feature_grid), np.squeeze(intervals['prediction']['bottom']),
                    np.squeeze(intervals['prediction']['top']), alpha=0.3, label='Prediction Interval')

    ax.fill_between(np.squeeze(feature_grid), np.squeeze(intervals['confidence']['top']),
                    np.squeeze(intervals['confidence']['bottom']), alpha=0.3, color='blue', label='Confidence Interval')
    ax.legend()

    plt.show()


def ar_1(num_draws=200, std=0.2, coeff=0.9):
    # Simulate data from AR1 process
    targets = [np.random.uniform(low=-1.0, high=1.0)]

    while len(targets) < num_draws:
        next_val = targets[-1] * coeff + np.random.normal(loc=0, scale=std)
        targets.append(next_val)

    targets = np.concatenate([np.reshape(t, [1, 1]) for t in targets], axis=0)

    return targets