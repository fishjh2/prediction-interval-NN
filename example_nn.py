import tensorflow as tf
import numpy as np
from functions.bootstrap_nn import bootstrap_NN
from functions.nn_data_sort import bootstrap_batch_sorter
from functions.utils import *

# Generate 200 draws from some simulated data based on a sin curve
features, targets, curve = sin_data(num_draws=200)

# A grid of points to predict once we've finished training
feature_grid = np.expand_dims(np.linspace(num=100, start=2, stop=6), axis=1)

# The number of bootstrap draws (or the number of networks we'll train at once)
num_networks = 100

# Bootstrap method (see description of available methods at top of nn_data_sort.py)
method = 'train_bootstrap'

# Prepare the data
iter_dict, sc = bootstrap_batch_sorter(features_data=features, targets_data=targets, net_type='NN',
                                       min_batch_size=50, dates=None, num_networks=num_networks,
                                       silent=True, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                       method=method, allowed_waste=0.1,
                                       feat_scaler='std_scale', targ_scaler='std_scale',
                                       features_lags=None, targets_lags=None, step_ahead=1)

tf.reset_default_graph()

with tf.Session() as sess:
    # Initialize the networks
    network = bootstrap_NN(sess, iter_dict, num_networks, num_hidden_nodes=[50, 50],
                           activation_fn='sigmoid', loss_fn='mse', learning_rate=0.05,
                           opt_choice='GradientDescent', average_drop=0.1, quantile=False,
                           feature_scaler=sc['feature_scaler'], target_scaler=sc['target_scaler'],
                           rev_target_scaler=sc['rev_target_scaler'], regularization=None,
                           reg_param=0.0001, data_method=method,
                           w_init=('truncated_normal', 0.5), b_init=('constant', 0.1),
                           f_w_init=('truncated_normal', 0.5), f_b_init=('constant', 0.0),
                           features_lags=None, targets_lags=None, multi_quantile=[0.05, 0.95],
                           quantile_learning_rate=0.1, quantile_hidden_nodes=[],
                           quantile_input=True, model_name='NN', checkpoint_dir='checkpoint')

    # Train the networks
    network.train(num_train_steps=20000, viz_every=2000, quantile_train_steps=5000,
                  quantile_viz_every=500)

    # Predict for the feature grid we generated above
    preds = network.predict(features=feature_grid)


# Calculate 90% confidence and prediction intervals
intervals = get_intervals(preds, confidence_percentiles=(5.0, 95.0),
                          prediction_percentiles=(5.0, 95.0))


# Example plot
plot_eg(features, targets, feature_grid, preds, intervals)