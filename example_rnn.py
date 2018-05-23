import tensorflow as tf

from functions.bootstrap_nn import bootstrap_RNN
from functions.nn_data_sort import bootstrap_batch_sorter
from functions.utils import *

# Simulate data from AR1 process
targets = ar_1(num_draws=200)

# These arguments need to be identical when passed to two functions below
num_networks = 200
num_steps = 5
method = 'train_bootstrap'
features_lags = None
targets_lags = {0: (1, 2)}

iter_dict, sc = bootstrap_batch_sorter(features_data=None, targets_data=targets, net_type='RNN',
                                       min_batch_size=2, dates=None, num_steps=num_steps,
                                       num_networks=num_networks, silent=True, train_ratio=0.7,
                                       val_ratio=0.15, test_ratio=0.15, method=method, allowed_waste=0.1,
                                       feat_scaler='std_scale', targ_scaler='std_scale',
                                       features_lags=features_lags, targets_lags=targets_lags,
                                       step_ahead=1)

tf.reset_default_graph()

with tf.Session() as sess:
    # Initialize the network
    network = bootstrap_RNN(sess, iter_dict, num_networks=num_networks, num_steps=num_steps,
                            num_layers=1, hidden_size=10, cell_type='basic', activation_fn='sigmoid',
                            loss_fn='mse', learning_rate=0.01, opt_choice='GradientDescent',
                            pass_state=False, average_drop=0.1, quantile=False,
                            feature_scaler=sc['feature_scaler'], target_scaler=sc['target_scaler'],
                            rev_target_scaler=sc['rev_target_scaler'], regularization=None,
                            reg_param=0.0001, w_init=('truncated_normal', 0.1), b_init=('constant', 0.1),
                            f_w_init=('truncated_normal', 0.1), f_b_init=('constant', 0.0), data_method=method,
                            features_lags=features_lags, targets_lags=targets_lags, multi_quantile=[0.05, 0.95],
                            quantile_learning_rate=0.1, quantile_hidden_nodes=[], quantile_input=True,
                            model_name='rnn', checkpoint_dir='checkpoint_rnn')

    # Train the network
    network.train(num_train_steps=10000, viz_every=1000, quantile_train_steps=2000,
                  quantile_viz_every=400)

    # Predict on new data - just the test dataset in this case as an example
    test_targets = network.test_iter.all_targets(resc=network.rev_target_scaler)
    preds = network.predict(features=None, targets=test_targets)


# We'll plot the predictions for the out-of-sample 'test' dataset
grid = np.arange(len(preds['main'][0]))

# Calculate 90% confidence and prediction intervals
intervals = get_intervals(preds, confidence_percentiles=(5.0, 95.0),
                          prediction_percentiles=(5.0, 95.0))

# Plot prediction and confidence intervals
plot_eg(None, test_targets, grid, preds, intervals)
