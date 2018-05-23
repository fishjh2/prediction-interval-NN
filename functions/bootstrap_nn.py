import tensorflow as tf
import numpy as np

from functions.neural_networks import network_base
from functions.losses import tf_quantile_loss, np_quantile_loss, tf_loss_lookup, np_loss_lookup
from functions.nn_layers import multi_nn, bootstrap_linear_layer

# Bootstrap NN ###################################################################################################


class bootstrap_NN(network_base):
    def __init__(self, sess, batch_iterators, num_networks, num_hidden_nodes=[20, 20],
                 activation_fn='sigmoid', loss_fn='mse', learning_rate=0.1,
                 opt_choice='GradientDescent', average_drop=0.1, quantile=False,
                 feature_scaler=None, target_scaler=None, rev_target_scaler=None,
                 regularization=None, reg_param=0.0001, data_method='fixed_at_end',
                 w_init=('truncated_normal', 0.1), b_init=('constant', 0.1),
                 f_w_init=('truncated_normal', 0.1), f_b_init=('constant', 0.0),
                 features_lags=None, targets_lags=None, multi_quantile=None,
                 quantile_learning_rate=0.1, quantile_hidden_nodes=[],
                 quantile_input=True, model_name='NN', checkpoint_dir='checkpoint_bootstrap'):
        """
        NN for training multiple networks at the same time
        Args
        sess: a Tensorflow session
        batch_iterators: a dict of data iterators which return target and feature batches when next_batch() called
        num_networks: the number of networks being trained at the same time
        num_hidden_nodes: the size of the hidden layers. List, with each value the size of a layer
                          e.g [20, 30] corresponds to one layer of size 20 followed by a layer of size 30
        activation_fn: activation function used in the NN. E.g sigmoid, tanh, relu etc.
        loss_fn: the loss function used. One of ['mse', 'mae']
        learning_rate: the learning rate for training the main network
        opt_choice: tf optimizer to use for training. E.g. 'Adam', 'GradientDescent' etc.
        average_drop: the ratio of networks to drop with the worst val losses when calculating the
                      ensemble prediction
        quantile: if not False, this is the quantile we want the networks main output to predict
                  e.g. 0.95 for the 95th quantile
        feature_scaler/target_scaler: a np scaler function for the features/targets data -
                                      used during prediction with new data
        rev_target_scaler: a np reverse scaling function for the targets (used if targets have been scaled)
        regularization: one of ['L1', 'L2', 'elastic'] - the regularization to apply to model weights
        reg_param: the regularization weighting term in the cost function. Tuple of two weights if reg is 'elastic'
        data_method: method used for generating data. See possible types at top of nn_data_sort.py
        w_init, b_init, f_w_init, f_b_init: initializers for the weights, biases, final layer weights and biases
                    consecutively. Should be tuples, length dependent on type of initializer. One of:
                    [('constant', value), ('random_normal', stddev), ('xavier'), ('truncated_normal', stddev)]
        features/targets_lags: dicts holding lags to add to features/targets respectively. Keys
                               are the indices of the columns, and values are tuples,
                               listing the lags to add for that column. E.g. {0: (1,2)} would add
                               the first and second lags of the variable in column 0 as additional features
        multi_quantile: either None, or list of the quantiles we wish the network to predict on top
                        of the main output. E.g. [0.05, 0.95] will predict 5th and 95th quantiles, giving
                        a 90% prediction interval
        quantile_learning_rate: learning rate for optimizing the weights for the quantile outputs
        quantile_hidden_nodes: list, denoting sizes of hidden layers of quantile output networks. If
                               empty, quantile outputs are calculated using a single output layer
        quantile_input: bool, if True, inputs to quantile output networks is the features concatenated
                        with the final hidden layer of the main network. If False, input to quantile
                        networks is just the final hidden layer of the main network.
        model_name: name for the model - used in the saving of checkpoint files
        checkpoint_dir: directory in which to save the checkpoint files
        """
        self.net_type = 'NN'
        self.sess = sess

        # Iterators holding the datasets
        self.train_iter = batch_iterators['train']
        self.val_iter = batch_iterators['val']
        self.test_iter = batch_iterators['test']
        self.train_val_iter = batch_iterators['tv']

        self.num_hidden_nodes = num_hidden_nodes
        self.activation_fn = getattr(tf.nn, activation_fn)
        self.learning_rate = learning_rate
        self.opt_choice = opt_choice

        self.num_networks = num_networks

        self.quantile = quantile
        self.multi_quantile = multi_quantile
        self.quantile_learning_rate = quantile_learning_rate
        self.quantile_hidden_nodes = quantile_hidden_nodes
        self.quantile_input = quantile_input

        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

        self.average_drop = average_drop

        # Add the loss functions - both tensorflow and numpy versions
        self.loss_fns = {}
        self.np_loss_fns = {}
        if self.quantile:
            self.loss_fns['main'] = tf_quantile_loss(self.quantile)
            self.np_loss_fns['main'] = np_quantile_loss(self.quantile)
        else:
            self.loss_fns['main'] = tf_loss_lookup[loss_fn]
            self.np_loss_fns['main'] = np_loss_lookup[loss_fn]

        if self.multi_quantile is not None:
            for q in self.multi_quantile:
                self.loss_fns[q] = tf_quantile_loss(q)
                self.np_loss_fns[q] = np_quantile_loss(q)

        # Add the scalers
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.rev_target_scaler = rev_target_scaler

        # Regularization
        self.regularization = regularization
        self.reg_param = reg_param

        # Weight initialization
        self.add_initializers(w_init, b_init, f_w_init, f_b_init)

        # Check whether the train set is the same for each separate network
        self.train_fixed(data_method)

        # Calculate the dimensions of the features and targets
        self.targets_dim = self.train_iter.targets_dim
        self.features_dim = self.train_iter.features_dim

        # Lags
        self.features_lags = features_lags
        self.targets_lags = targets_lags

        # List of the outputs of the network
        self.keys = ['main']
        if self.multi_quantile:
            self.keys += self.multi_quantile

        # Lists to store the steps at which the network achieves lowest loss on val set
        self.best_step, self.best_val_losses = {}, {}
        for k in self.keys:
            self.best_step[k] = [0] * self.num_networks
            self.best_val_losses[k] = [float('inf')] * self.num_networks

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=None)

    def build_model(self):
        """
        Build the network
        """
        self.features_pl = tf.placeholder(tf.float32, [self.num_networks, None, self.features_dim], 'features_pl')
        self.targets_pl = tf.placeholder(tf.float32, [self.num_networks, None, self.targets_dim], 'targets_pl')

        # Hidden layers
        output = multi_nn(self.features_pl, self.features_dim, self.num_networks,
                          self.num_hidden_nodes, self.w_init, self.b_init,
                          self.activation_fn)

        # Output layer
        output_layer = bootstrap_linear_layer(output, self.num_hidden_nodes[-1], self.targets_dim,
                                              activation_fn=None, num_networks=self.num_networks,
                                              w_init=self.f_w_init, b_init=self.f_b_init,
                                              name='final')

        self.output = {'main': output_layer}

        # Calculate the main loss used for training
        self.loss = self.loss_fns['main'](self.targets_pl, self.output['main'])

        # Add regularization
        if self.regularization:
            self.loss += self.add_regularizer(multi_network=True)

        # Optimizer
        self.optimizer = self.bootstrap_optimizer(self.loss, key='main', lr=self.learning_rate)

        # Quantile outputs
        if self.multi_quantile is not None:
            self.add_quantile_outputs(output, self.num_hidden_nodes[-1], self.features_pl, self.targets_pl)

        # Trainable vars
        self.trainable_vars = tf.trainable_variables()

    def run_train_steps(self, out_keys, optimizer, v_every, num_steps):
        """
        Optimize the parts of the network denoted by out_keys
        """
        for step in range(num_steps):

            f_batch, t_batch = self.train_iter.next_batch()

            _ = self.sess.run(optimizer, feed_dict={self.features_pl: f_batch,
                                                    self.targets_pl: t_batch})

            if step % v_every == 0:

                _, TRAIN_LOSSES = self.run_data_set(self.train_iter)
                _, VAL_LOSSES = self.run_data_set(self.val_iter)

                print("Step: {}".format(step))
                self.print_loss(losses=[TRAIN_LOSSES, VAL_LOSSES], names=['Train', 'Val'], keys=out_keys)

                # Update the best val losses if any of the networks beat their best val loss so far
                save_update = False

                for k in out_keys:
                    for b in range(self.num_networks):
                        if VAL_LOSSES[k][b] < self.best_val_losses[k][b]:
                            self.best_val_losses[k][b] = VAL_LOSSES[k][b]
                            self.best_step[k][b] = step
                            save_update = True

                # Save if any of the separate bootstrapped networks have improved on their best val loss
                if save_update:
                    T_VARS = self.sess.run(self.trainable_vars)
                    for k in out_keys:
                        self.update_t_vars(k, T_VARS, step=step)

    def train(self, viz_every=500, num_train_steps=5000, quantile_viz_every=500,
              quantile_train_steps=5000):
        """
        Train the network
        viz_every: number of steps after which to print current loss, and evaluate val loss for each network to see if
                   it has improved. Model weights save if val loss is lower than previous best
        num_train_steps: the number of steps of training to complete
        quantile_viz_every/quantile_train_steps: same as above, but for the quantile outputs of the network
        """
        self.sess.run(tf.global_variables_initializer())

        # Train main network
        print('Training main network...')
        self.run_train_steps(['main'], self.optimizer, viz_every, num_train_steps)
        self.restore_best_vars()

        # Train quantile outputs
        if self.multi_quantile is not None:
            print('Training quantile outputs...')
            self.run_train_steps(self.multi_quantile, self.q_opts, quantile_viz_every, quantile_train_steps)
            self.restore_best_vars()

        # Evaluate final network predictions and losses for each dataset
        self.TEST_PREDS, self.TEST_LOSSES = self.run_data_set(self.test_iter)
        self.TRAIN_VAL_PREDS, self.TRAIN_VAL_LOSSES = self.run_data_set(self.train_val_iter)

        if self.train_set == 'fixed':
            self.TRAIN_PREDS, self.TRAIN_LOSSES = self.run_data_set(self.train_iter)
            self.VAL_PREDS, self.VAL_LOSSES = self.run_data_set(self.val_iter)

        # Print final losses
        print('Average Losses')
        if hasattr(self, 'VAL_LOSSES'):
            self.print_loss(losses=[self.TRAIN_LOSSES, self.VAL_LOSSES, self.TEST_LOSSES],
                            names=['Train', 'Val', 'Test'], keys=self.keys)
        else:
            self.print_loss(losses=[self.TRAIN_VAL_LOSSES, self.TEST_LOSSES], names=['Train_Val', 'Test'],
                            keys=self.keys)

        # Calculate the ensemble prediction, and associated loss
        self.average_preds()

    def run_data_set(self, iterator):
        """
        Runs predictions and loss ops for the whole data set stored in "iterator"
        """
        # Store starting value of iterator to return to
        counter_start = iterator.counter
        # Make sure we start from the first batch
        iterator.counter = 0

        targets = []
        preds_dict = {}
        for k in self.output.keys():
            preds_dict[k] = []

        for step in range(iterator.num_batches):

            f_batch, t_batch = iterator.next_batch()

            PREDS = self.sess.run(self.output, feed_dict={self.features_pl: f_batch,
                                                          self.targets_pl: t_batch})

            for k in PREDS.keys():
                preds_dict[k].append(PREDS[k])

            targets.append(t_batch)

        all_preds = {}
        average_losses = {}

        # Join targets into shape [num_networks, dataset_size, targets_dim]
        targets = np.concatenate(targets, axis=1)

        for k in preds_dict.keys():
            # Join predictions into shape [num_networks, dataset_size, targets_size]
            preds_dict[k] = np.concatenate(preds_dict[k], axis=1)
            all_preds[k] = []
            average_losses[k] = []
            # Split the predictions of the individual networks
            for b in range(self.num_networks):
                p = preds_dict[k][b]
                ts = targets[b]
                if self.rev_target_scaler:
                    p = self.rev_target_scaler(p)
                    ts = self.rev_target_scaler(ts)
                all_preds[k].append(p)
                average_losses[k].append(self.np_loss_fns[k](ts, p))

        # Return iterator counter to starting value
        iterator.counter = counter_start

        return all_preds, average_losses


# Bootstrap RNN ######################################################################################################


class boot_BasicRNNCell(tf.contrib.rnn.RNNCell):
    """The most basic RNN cell -  bootstrap version"""

    def __init__(self, num_units, input_size, num_networks, name, activation_fn,
                 w_init, b_init):
        self._num_units = num_units
        self._input_size = input_size
        self._num_networks = num_networks
        self._name = name
        self._activation_fn = activation_fn
        self._w_init = w_init
        self._b_init = b_init

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        inputs = tf.reshape(inputs, [self._num_networks, -1, self._input_size])
        state = tf.reshape(state, [self._num_networks, -1, self._num_units])

        input_weights = tf.get_variable(name='input_weights_' + self._name,
                                        shape=[self._num_networks, self._input_size, self._num_units],
                                        initializer=self._w_init)

        state_weights = tf.get_variable(name='state_weights_' + self._name,
                                        shape=[self._num_networks, self._num_units, self._num_units],
                                        initializer=self._w_init)
        bias = tf.get_variable(name='bias_' + self._name, shape=[self._num_networks, 1, self._num_units],
                               initializer=self._b_init)

        inner = tf.matmul(inputs, input_weights) + tf.matmul(state, state_weights) + bias

        # output will be dimensions [num_networks, batch_size, num_units]
        output = self._activation_fn(inner)

        # change output to 2D: [num_networks * batch_size, num_units]
        output = tf.reshape(output, [-1, self._num_units])

        return output, output


class boot_GRUCell(tf.contrib.rnn.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size, num_networks, name, activation_fn,
                 w_init):
        self._num_units = num_units
        self._input_size = input_size
        self._activation_fn = activation_fn
        self._num_networks = num_networks
        self._name = name
        self._w_init = w_init
        # For GRU cell start with bias of 1.0 to not reset and not update
        self._b_init = tf.constant_initializer(1.0)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with num_units cells - bootstrapped version."""

        # Reshape inputs to [num_networks, batch_size, input_size]
        inputs = tf.reshape(inputs, [self._num_networks, -1, self._input_size])
        # Reshape state to [num_networks, batch_size, num_units]
        state = tf.reshape(state, [self._num_networks, -1, self._num_units])

        # Update gate
        up_input_weights = tf.get_variable(name='update_input_weights_' + self._name,
                                           shape=[self._num_networks, self._input_size, self._num_units],
                                           initializer=self._w_init)
        up_state_weights = tf.get_variable(name='update_state_weights_' + self._name,
                                           shape=[self._num_networks, self._num_units, self._num_units],
                                           initializer=self._w_init)
        up_bias = tf.get_variable(name='update_bias_' + self._name, shape=[self._num_networks, 1, self._num_units],
                                  initializer=self._b_init)

        update_gate = tf.sigmoid(tf.matmul(inputs, up_input_weights) + tf.matmul(state, up_state_weights) + up_bias)

        # Reset gate
        reset_input_weights = tf.get_variable(name='reset_input_weights_' + self._name,
                                              shape=[self._num_networks, self._input_size, self._num_units],
                                              initializer=self._w_init)
        reset_state_weights = tf.get_variable(name='reset_state_weights_' + self._name,
                                              shape=[self._num_networks, self._num_units, self._num_units],
                                              initializer=self._w_init)
        reset_bias = tf.get_variable(name='reset_bias_' + self._name, shape=[self._num_networks, 1, self._num_units],
                                     initializer=self._b_init)

        reset_gate = tf.sigmoid(
            tf.matmul(inputs, reset_input_weights) + tf.matmul(state, reset_state_weights) + reset_bias)

        # Output/hidden state update
        h_input_weights = tf.get_variable(name='h_input_weights_' + self._name,
                                          shape=[self._num_networks, self._input_size, self._num_units],
                                          initializer=self._w_init)
        h_state_weights = tf.get_variable(name='h_state_weights_' + self._name,
                                          shape=[self._num_networks, self._num_units, self._num_units],
                                          initializer=self._w_init)
        h_bias = tf.get_variable(name='h_bias_' + self._name, shape=[self._num_networks, 1, self._num_units],
                                 initializer=self._b_init)

        h = self._activation_fn(
            tf.matmul(inputs, h_input_weights) + tf.matmul(state * reset_gate, h_state_weights) + h_bias)

        output = update_gate * state + (1 - update_gate) * h

        # change output to 2D: [num_networks * batch_size, num_units]
        output = tf.reshape(output, [-1, self._num_units])

        return output, output


class bootstrap_RNN(network_base):
    def __init__(self, sess, batch_iterators, num_networks, num_steps=5, num_layers=1, hidden_size=10,
                 cell_type='basic', activation_fn='tanh', loss_fn='mse', learning_rate=0.001,
                 opt_choice='GradientDescent', pass_state=False, average_drop=0.1, quantile=False,
                 feature_scaler=None, target_scaler=None, rev_target_scaler=None,
                 regularization=None, reg_param=0.0001, w_init=('truncated_normal', 0.1),
                 b_init=('constant', 0.1), f_w_init=('truncated_normal', 0.1), f_b_init=('constant', 0.0),
                 data_method='fixed_at_end', features_lags=None, targets_lags=None,
                 multi_quantile=None, quantile_learning_rate=0.1, quantile_hidden_nodes=[],
                 quantile_input=True, model_name='rnn', checkpoint_dir='checkpoint_rnn'):
        """
        RNN for training multiple networks (possibly with data composed of bootstrap draws) at the same time
        Args
        sess: a Tensorflow session
        batch_iterators: a dict of data iterators which return target and feature batches when next_batch() called
        num_steps: the number of steps the RNN is rolled out during training
        num_networks: how many networks we're training at once
        num_layers: the number of layers of the RNN
        hidden_size: the hidden size of each RNN cell
        cell_type: one of ['basic', 'gru'] - the RNN cell type
        activation_fn: activation function used in the RNN cells. One of ['sigmoid', 'tanh', 'relu']
        loss_fn: the loss function used. One of ['mse', 'mae']
        learning_rate: the learning rate for training
        opt_choice: tensorflow optimizer to use for training
        pass_state: whether to pass the state from the last training batch in the epoch onto the first training batch
                    in the next epoch
        average_drop: the ratio of networks to drop with the worst val losses when calculating the average prediction
                      of all networks at the end
        quantile: if not False, this is the quantile we want the networks to predict e.g. 0.95 for the 95th quantile
        feature_scaler: a np scaler function for the features data - used during prediction of new data
        target_scaler: a np scaler function for the targets data - used during prediction of new data
        rev_target_scaler: a np reverse scaling function for the targets (used if targets have been scaled)
        regularization: one of ['L1', 'L2', 'elastic'] - the regularization to apply to model weights
        reg_param: the regularization weighting term in the cost function. Tuple of two weights if reg is 'elast
        w_init, b_init, f_w_init, f_b_init: initializers for the weights, biases, final layer weights and biases
                    consecutively. Should be tuples, length dependent on type of initializer. One of:
                    [('constant', value), ('random_normal', stddev), ('xavier'), ('truncated_normal', stddev)]
        data_method: the method which has been used for drawing the data (e.g. 'basic_bootstrap', 'simple_block' etc)
        features/targets_lags: dicts holding lags to add to features/targets respectively. Keys
                               are the indices of the columns, and values are tuples,
                               listing the lags to add for that column. E.g. {0: (1,2)} would add
                               the first and second lags of the variable in column 0 as additional features
        multi_quantile: either None, or list of the quantiles we wish the network to predict on top
                of the main output. E.g. [0.05, 0.95] will predict 5th and 95th quantiles, giving
                a 90% prediction interval
        quantile_learning_rate: learning rate for optimizing the weights for the quantile outputs
        quantile_hidden_nodes: list, denoting sizes of hidden layers of quantile output networks. If
                               empty, quantile outputs are calculated using a single output layer
        quantile_input: bool, if True, inputs to quantile output networks is the features concatenated
                        with the final hidden layer of the main network. If False, input to quantile
                        networks is just the final hidden layer of the main network.
        model_name: name for the model - used in the saving of checkpoint files
        checkpoint_dir: directory in which to save the checkpoint files
        """
        self.net_type = 'RNN'
        self.sess = sess

        # Data iterators - hold the data sets, and return the next batch when next_batch() called
        self.train_iter = batch_iterators['train']  # Training data (train and val data in here - val masked out during
        # training)
        self.test_iter = batch_iterators['test']  # Test data - batch length of 1, and no masking
        self.train_val_iter = batch_iterators['tr_val']  # Train and val data for predictions at end - batch size of
        # 1 and no masking
        self.val_iter = batch_iterators['val']  # Val iterator - batch_length of 1, with train data masked out

        # Whether the data is 4d or 3d
        self.four_d = self.train_iter.four_d

        self.num_steps = num_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.activation_fn = getattr(tf.nn, activation_fn)
        self.learning_rate = learning_rate
        self.opt_choice = opt_choice
        self.pass_state = pass_state

        self.num_networks = num_networks

        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir

        self.average_drop = average_drop
        self.quantile = quantile
        self.multi_quantile = multi_quantile
        self.quantile_learning_rate = quantile_learning_rate
        self.quantile_hidden_nodes = quantile_hidden_nodes
        self.quantile_input = quantile_input

        # Add the scalers
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.rev_target_scaler = rev_target_scaler

        self.regularization = regularization
        self.reg_param = reg_param

        self.add_initializers(w_init, b_init, f_w_init, f_b_init)

        # Check whether the train set is the same for each separate network
        self.train_fixed(data_method)

        # Dimensions of the target and features variables
        self.targets_dim = self.train_iter.targets_dim
        self.features_dim = self.train_iter.features_dim

        # Add the lags
        self.features_lags = features_lags
        self.targets_lags = targets_lags

        # Add the loss functions - both tensorflow and numpy versions
        self.loss_fns = {}
        self.np_loss_fns = {}
        if self.quantile:
            self.loss_fns['main'] = tf_quantile_loss(self.quantile)
            self.np_loss_fns['main'] = np_quantile_loss(self.quantile)
        else:
            self.loss_fns['main'] = tf_loss_lookup[loss_fn]
            self.np_loss_fns['main'] = np_loss_lookup[loss_fn]

        if self.multi_quantile is not None:
            for q in self.multi_quantile:
                self.loss_fns[q] = tf_quantile_loss(q)
                self.np_loss_fns[q] = np_quantile_loss(q)

        # List of the outputs of the network
        self.keys = ['main']
        if self.multi_quantile:
            self.keys += self.multi_quantile

        # Lists to store the steps at which the network achieves lowest loss on val set
        self.best_step, self.best_val_losses = {}, {}
        for k in self.keys:
            self.best_step[k] = [0] * self.num_networks
            self.best_val_losses[k] = [float('inf')] * self.num_networks

        self.build_model()

        self.saver = tf.train.Saver()

    def build_model(self):

        # Placeholders
        if self.four_d:
            self.features_pl = tf.placeholder(tf.float32, [self.num_networks, None, self.num_steps, self.features_dim],
                                              'features_pl')
            self.targets_pl = tf.placeholder(tf.float32, [self.num_networks, None, self.num_steps, self.targets_dim],
                                             'targets_pl')
        else:
            self.features_pl = tf.placeholder(tf.float32, [None, self.num_steps, self.features_dim], 'features_pl')
            self.targets_pl = tf.placeholder(tf.float32, [None, self.num_steps, self.targets_dim], 'targets_pl')

        self.mask_pl = tf.placeholder(tf.float32, [self.num_networks, None, self.num_steps, 1], 'mask_pl')
        self.seq_len_pl = tf.placeholder(tf.int32, shape=[None], name='seq_len_pl')
        self.b_size_pl = tf.placeholder(tf.int32, shape=(), name='b_size_pl')

        # Tile features and targets - one for each network trained concurrently
        if not self.four_d:
            features_tiled = tf.tile(tf.expand_dims(self.features_pl, axis=0), [self.num_networks, 1, 1, 1])
            targets_tiled = tf.tile(tf.expand_dims(self.targets_pl, axis=0), [self.num_networks, 1, 1, 1])
        else:
            features_tiled = self.features_pl
            targets_tiled = self.targets_pl

        # Reshape targets and mask to match the output shape of the RNN
        targets = tf.concat(tf.unstack(targets_tiled, num=self.num_steps, axis=2), axis=1)
        self.targets = targets[:, :self.b_size_pl, :]
        mask = tf.concat(tf.unstack(self.mask_pl, num=self.num_steps, axis=2), axis=1)
        self.mask = mask[:, :self.b_size_pl, :]

        # Unstack features into num_step length list of [num_boot, batch_size, features_dim] tensors to feed to RNN
        self.features_mat = tf.unstack(features_tiled, num=self.num_steps, axis=2)
        self.features = [tf.reshape(t, [-1, self.features_dim]) for t in self.features_mat]

        # Create the cells and state placeholders for each layer
        cells = []
        self.initial_state_list = []
        for l in range(self.num_layers):
            if l == 0:
                in_size = self.features_dim
            else:
                in_size = self.hidden_size

            if self.cell_type == 'gru':
                cells.append(boot_GRUCell(self.hidden_size, input_size=in_size, num_networks=self.num_networks,
                                          name=str(l), activation_fn=self.activation_fn, w_init=self.w_init))
            elif self.cell_type == 'basic':
                cells.append(boot_BasicRNNCell(self.hidden_size, input_size=in_size, num_networks=self.num_networks,
                                               name=str(l), activation_fn=self.activation_fn, w_init=self.w_init,
                                               b_init=self.b_init))

            self.initial_state_list.append(
                tf.placeholder(tf.float32, [None, self.hidden_size], name='init_state_pl_' + str(l)))

        # Set up the RNN
        multi_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.cell_outputs, self.final_state = tf.contrib.rnn.static_rnn(multi_cell, self.features,
                                                                        initial_state=tuple(self.initial_state_list),
                                                                        sequence_length=self.seq_len_pl)

        # Reshape outputs to a length num_steps list, of [num_networks, batch_size, hidden_size] tensors
        outputs = [tf.reshape(c, [self.num_networks, -1, self.hidden_size]) for c in self.cell_outputs]

        # Join the elements of the list, giving shape of [num_networks, batch_size * num_steps, hidden_size]
        joined_outputs = tf.concat(outputs, axis=1)[:, :self.b_size_pl, :]

        # Output fully connected layer
        output_weights = tf.get_variable(name="output_weights",
                                         shape=[self.num_networks, self.hidden_size, self.targets_dim],
                                         initializer=self.f_w_init)
        output_bias = tf.get_variable(name="output_bias", shape=[self.num_networks, 1, self.targets_dim],
                                      initializer=self.f_b_init)
        output = tf.matmul(joined_outputs, output_weights) + output_bias

        self.output = {'main': output}

        # Add networks for quantile outputs
        if self.multi_quantile is not None:
            features = tf.concat(self.features_mat, axis=1)[:, :self.b_size_pl, :]
            self.add_quantile_outputs(joined_outputs, self.hidden_size, features, self.targets)

        # Calculate the loss, masking out val data during training
        self.loss = self.loss_fns['main'](self.targets, self.output['main'], mask=self.mask)

        # Add regularization
        if self.regularization:
            self.loss += self.add_regularizer(multi_network=True)

        # Optimizer
        self.optimizer = self.bootstrap_optimizer(self.loss, key='main', lr=self.learning_rate)

        # Trainable vars
        self.trainable_vars = tf.trainable_variables()

    def run_train_steps(self, out_keys, optimizer, v_every, num_steps):

        for step in range(num_steps):

            # Fill numpy version of state for next batch
            if step == 0:
                state = self.fill_zero_state(self.train_iter)

            if self.train_iter.counter == 0:
                if self.pass_state:
                    state = self.update_train_state(state)
                else:
                    state = self.fill_zero_state(self.train_iter)

            # Get next batch
            f_batch, t_batch, seq_len, mask = self.train_iter.next_batch()
            b_size = int(np.sum(seq_len))

            # Fill the feed_dict
            feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch,
                         self.seq_len_pl: seq_len, self.b_size_pl: b_size, self.mask_pl: mask}

            for l in range(self.num_layers):
                feed_dict[self.initial_state_list[l]] = state[l]

            # Run a training step
            ops = {"opt": optimizer, "final_state": self.final_state}
            if step == 0:
                ops = {"final_state": self.final_state}
            returns = self.sess.run(ops, feed_dict=feed_dict)

            # Pass final state of previous batch onto next batch
            state = returns["final_state"]

            if step % v_every == 0:
                # Check progress
                _, TRAIN_LOSSES, _ = self.run_data_set(self.train_iter)
                _, VAL_LOSSES, _ = self.run_data_set(self.val_iter)

                print("Step: {}".format(step))
                self.print_loss(losses=[TRAIN_LOSSES, VAL_LOSSES], names=['Train', 'Val'], keys=out_keys)

                # Update the best val losses if any of the networks beat their best val loss so far
                save_update = False

                for k in out_keys:
                    for b in range(self.num_networks):
                        if VAL_LOSSES[k][b] < self.best_val_losses[k][b]:
                            self.best_val_losses[k][b] = VAL_LOSSES[k][b]
                            self.best_step[k][b] = step
                            save_update = True

                # Save if any of the separate bootstrapped networks have improved on their best val loss
                if save_update:
                    T_VARS = self.sess.run(self.trainable_vars)
                    for k in out_keys:
                        self.update_t_vars(k, T_VARS, step=step)

    def train(self, viz_every=500, num_train_steps=5000, quantile_viz_every=500,
              quantile_train_steps=5000):
        """
        Train the network and calculate the final predictions/loss for all datasets
        Args
        viz_every: the number of epochs at which we print the train and val losses, and evaluate the val loss for
                   every individual network to check if they've beaten their best val loss. Saves a checkpoint if so.
        num_train_steps: the number of steps to train the model for
        quantile_viz_every/quantile_train_steps: same as above, but for the quantile outputs of the network
        """

        self.sess.run(tf.global_variables_initializer())

        # Train main network
        print('Training main network...')
        self.run_train_steps(['main'], self.optimizer, viz_every, num_train_steps)
        self.restore_best_vars()

        # Train quantile outputs
        if self.multi_quantile is not None:
            print('Training quantile outputs...')
            self.run_train_steps(self.multi_quantile, self.q_opts, quantile_viz_every, quantile_train_steps)
            self.restore_best_vars()

        # Run final predictions and losses
        self.TRAIN_VAL_PREDS, self.TRAIN_VAL_LOSSES, tr_state = self.run_data_set(self.train_val_iter)
        self.TEST_PREDS, self.TEST_LOSSES, _ = self.run_data_set(self.test_iter, previous_state=tr_state)

        # Add individual train and val set predictions if the data is the same for every network
        self.add_train_val_preds()

        # Print final losses
        print('Average Losses')
        if hasattr(self, 'VAL_LOSSES'):
            self.print_loss(losses=[self.TRAIN_LOSSES, self.VAL_LOSSES, self.TEST_LOSSES],
                            names=['Train', 'Val', 'Test'], keys=self.keys)
        else:
            self.print_loss(losses=[self.TRAIN_VAL_LOSSES, self.TEST_LOSSES], names=['Train_Val', 'Test'],
                            keys=self.keys)

        # Calculate the ensemble prediction, and associated loss
        self.average_preds()

    def run_data_set(self, iterator, previous_state=None):
        """
        Runs a full data set, returning the predictions and average loss for the whole dataset, for each
        bootstrap network individually
        Args
        iterator: a data iterator which returns the next batches of data when iterator.next_batch() called
        previous_state: the hidden state of the RNN cell from the previous data point
        Returns
        all_preds: a dictionary, where the keys are the names of the outputs, and the values are
                   lists of len num_networks, with the predictions of each separate network for this data set
        average_losses: as above, but holding the average loss for the dataset for each output and
                        each separate network
        state: the final RNN cell hidden state at the end of this data set
        """
        # Store starting value of iterator to return to
        counter_start = iterator.counter
        # Start from the first batch
        iterator.counter = 0

        # Lists for storing preds, targets and masks for each network
        preds_dict = {}
        masks = []
        targets = []
        for k in self.keys:
            preds_dict[k] = []

        # Numpy version of state
        if not previous_state:
            state = self.fill_zero_state(iterator)
        else:
            state = previous_state

        for step in range(iterator.num_batches):
            f_batch, t_batch, seq_len, mask = iterator.next_batch()
            b_size = int(np.sum(seq_len))

            # Fill the feed_dict
            feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch,
                         self.seq_len_pl: seq_len, self.b_size_pl: b_size, self.mask_pl: mask}

            for l in range(self.num_layers):
                feed_dict[self.initial_state_list[l]] = state[l]

            # Define which ops to run and run the session
            ops = {"final_state": self.final_state, "preds": self.output, "mask": self.mask, "targets": self.targets}
            returns = self.sess.run(ops, feed_dict=feed_dict)

            # Pass final state onto next batch
            state = returns["final_state"]

            # Store predictions, targets and masks for the batch
            for k in self.keys:
                preds_dict[k].append(returns['preds'][k])
            targets.append(returns['targets'])
            masks.append(returns['mask'])

        all_preds = {}
        average_losses = {}

        # Join targets and masks into shape [num_networks, dataset_size, targets_size]
        targets = np.concatenate(targets, axis=1)
        masks = np.concatenate(masks, axis=1)

        for k in self.keys:
            # Join predictions into shape [num_networks, dataset_size, targets_size]
            preds_dict[k] = np.concatenate(preds_dict[k], axis=1)
            all_preds[k] = []
            average_losses[k] = []

            # Split the predictions of the individual networks
            for b in range(self.num_networks):
                p = preds_dict[k][b]
                t = targets[b]
                m = masks[b]
                if self.rev_target_scaler:
                    p = self.rev_target_scaler(p)
                    t = self.rev_target_scaler(t)
                all_preds[k].append(p)
                average_losses[k].append(self.np_loss_fns[k](t, p, mask=m))

        # Return iterator counter to starting value
        iterator.counter = counter_start

        return all_preds, average_losses, state


    def fill_zero_state(self, iter_):
        """
        Returns state filled with zeros for start of training/eval
        Args
        iter_: data iterator
        Returns
        state: a list of the states for each year filled with zeros, used to initiate the RNN cell states
        """
        state = []
        for l in range(self.num_layers):
            state.append(np.zeros([self.num_networks * iter_.batch_size, self.hidden_size]))
        return state

    def update_train_state(self, prev_state):
        """
        Takes the state from last training batch in epoch and shifts it down by one row, so
        that correct state is passed to the first training batch in the next epoch
        Args
        prev_state: the state from the final batch of the previous epoch
        Returns
        new_state: the RNN cell hidden state ready for the next batch of training data
        """
        old_state = list(prev_state)
        new_state = []
        for l in range(self.num_layers):
            s = np.reshape(old_state[l], [self.num_networks, -1, self.hidden_size])
            new_s = np.concatenate([np.zeros([self.num_networks, 1, self.hidden_size],
                                             dtype=np.float64), s], axis=1)[:, :-1, :]
            new_state.append(np.reshape(new_s, [-1, self.hidden_size]))
        return tuple(new_state)

    def add_train_val_preds(self):
        """
        Adds separate train and validation predictions, if the train/val data is the same for every network
        """
        if self.train_set == 'fixed':
            tr_len = self.train_val_iter.train_len
            ts = self.train_val_iter.all_targets(resc=self.rev_target_scaler)
            tr_ts = ts[:tr_len, :]
            v_ts = ts[tr_len:, :]
            self.TRAIN_PREDS, self.VAL_PREDS, self.TRAIN_LOSSES, self.VAL_LOSSES = {}, {}, {}, {}
            for k in self.keys:
                self.TRAIN_PREDS[k] = []
                self.VAL_PREDS[k] = []
                self.TRAIN_LOSSES[k] = []
                self.VAL_LOSSES[k] = []
                for t in range(self.num_networks):
                    t_ps = self.TRAIN_VAL_PREDS[k][t][:tr_len, :]
                    v_ps = self.TRAIN_VAL_PREDS[k][t][tr_len:, :]
                    self.TRAIN_PREDS[k].append(t_ps)
                    self.VAL_PREDS[k].append(v_ps)
                    self.TRAIN_LOSSES[k].append(self.np_loss_fns[k](tr_ts, t_ps))
                    self.VAL_LOSSES[k].append(self.np_loss_fns[k](v_ts, v_ps))