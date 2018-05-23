import tensorflow as tf

from functions.bootstrap_nn import bootstrap_NN
from functions.nn_data_sort import bootstrap_batch_sorter
from functions.utils import *

# Simulate data from AR1 process
targets = ar_1(num_draws=200)


# The four arguments below need to be identical for bootstrap_batch_sorter and
# bootstrap_NN
features_lags = None
targets_lags = {0: (1, 2)}
num_networks = 20
method = 'train_bootstrap'

iter_dict, sc = bootstrap_batch_sorter(features_data=None, targets_data=targets, net_type='NN',
                                       min_batch_size=50, dates=None, num_networks=num_networks,
                                       silent=True, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                       method=method, allowed_waste=0.1,
                                       feat_scaler='std_scale', targ_scaler='std_scale',
                                       features_lags=features_lags, targets_lags=targets_lags,
                                       step_ahead=1)

tf.reset_default_graph()
with tf.device('/cpu:0'):

    with tf.Session() as sess:
        network = bootstrap_NN(sess, iter_dict, num_networks, num_hidden_nodes=[20],
                               activation_fn='sigmoid', loss_fn='mse', learning_rate=0.05,
                               opt_choice='GradientDescent', average_drop=0.1, quantile=False,
                               feature_scaler=sc['feature_scaler'], target_scaler=sc['target_scaler'],
                               rev_target_scaler=sc['rev_target_scaler'], regularization=None,
                               reg_param=0.0001, data_method=method,
                               w_init=('truncated_normal', 0.5), b_init=('constant', 0.1),
                               f_w_init=('truncated_normal', 0.5), f_b_init=('constant', 0.0),
                               features_lags=features_lags, targets_lags=targets_lags, multi_quantile=[0.05, 0.95],
                               quantile_learning_rate=0.1, quantile_hidden_nodes=[],
                               quantile_input=True, model_name='NN', checkpoint_dir='checkpoint')

        network.train(num_train_steps=5000, viz_every=500, quantile_train_steps=2000,
                      quantile_viz_every=500)


# We'll plot the predictions for the out-of-sample 'test' dataset
preds = network.TEST_PREDS
grid = np.arange(len(preds['main'][0]))

# Calculate 90% confidence and prediction intervals
intervals = get_intervals(preds, confidence_percentiles=(5.0, 95.0),
                          prediction_percentiles=(5.0, 95.0))

# Plot prediction and confidence intervals
test_targets = network.test_iter.all_targets(resc=network.rev_target_scaler)
plot_eg(None, test_targets, grid, preds, intervals)