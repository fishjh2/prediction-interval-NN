import tensorflow as tf
import numpy as np
import os

from functions.losses import tf_quantile_loss, tf_mse, np_mse, np_quantile_loss
from functions.nn_data_sort import prepare_rnn_data, rnn_batch_iterator, add_lags, expand_batches, bootstrap_batch_iterator
from functions.nn_layers import multi_nn, bootstrap_linear_layer
tf_l = tf.contrib.layers


class network_base(object):
    """
    Base class for all neural networks, with methods which will be used by them all
    """

    def add_loss(self, targets, preds, average=True):
        """
        Returns the correct loss for the network, including adding regularization if required
        Args
        targets: a tensor holding the targets
        preds: a tensor holding the networks predictions
        Returns
        loss: a scalar tensor holding the loss
        """
        if self.quantile:
            loss = tf_quantile_loss(targets, preds, self.quantile, average=average)
        else:
            loss = tf_mse(targets, preds, average=average)

        # Add regularization
        if self.regularization:
            loss += self.add_regularizer()

        return loss

    def add_optimizer(self, lr):
        """
        Adds the optimizer to the network
        """
        o = getattr(tf.train, self.opt_choice + 'Optimizer')

        if self.opt_choice == 'Momentum':
            opt = o(lr, 0.9)
        else:
            opt = o(lr)

        return opt

    def bootstrap_optimizer(self, loss, lr, key='main'):

        opt = self.add_optimizer(lr=lr)

        if key == 'main':
            variables = [i for i in tf.trainable_variables() if 'quantile_' not in i.name]
        else:
            variables = [i for i in tf.trainable_variables() if 'quantile_' + str(key) in i.name]

        grads_and_vars = opt.compute_gradients(loss, var_list=variables)

        # Scale the gradients by the number of networks
        new_grads_and_vars = [(g * self.num_networks, v) for g, v in grads_and_vars]
        optimizer = opt.apply_gradients(new_grads_and_vars)

        return optimizer

    def add_initializers(self, w_init, b_init, f_w_init, f_b_init):
        """
        Adds the intializers to the network object
        Args
        w_init, b_init, f_w_init, f_b_init: intializers for weights, biases, final layer weights, final layer
                                            biases consecutively
        """
        names = ['w_init', 'b_init', 'f_w_init', 'f_b_init']

        for num, i in enumerate([w_init, b_init, f_w_init, f_b_init]):

            if i[0] == 'truncated_normal':
                setattr(self, names[num], tf.truncated_normal_initializer(stddev=i[1]))
            elif i[0] == 'random_normal':
                setattr(self, names[num], tf.random_normal_initializer(stddev=i[1]))
            elif i[0] == 'constant':
                setattr(self, names[num], tf.constant_initializer(i[1]))
            elif i[0] == 'xavier':
                setattr(self, names[num], tf.contrib.layers.xavier_initializer(uniform=i[1]))
            else:
                raise Exception('Unrecognised initializer type')

    def add_regularizer(self, multi_network=False):
        """
        Adds the regularizer to the network class
        Args
        multi_network: whether multiple networks are being trained at the same time
        """
        assert hasattr(self, 'num_networks') == multi_network, 'multi_network should be True if \
                                                                    training multiple networks at once'

        if self.regularization == 'L1':
            regularizer = tf.contrib.layers.l1_regularizer(self.reg_param)
        elif self.regularization == 'L2':
            regularizer = tf.contrib.layers.l2_regularizer(self.reg_param)
        elif self.regularization == 'elastic':
            assert type(self.reg_param) is tuple, 'reg_param should be a tuple if using \'elastic\' regularization'
            regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=self.reg_param[0], scale_l2=self.reg_param[1])

        else:
            raise Exception('Unrecognised regularization type')

        weights_list = [v for v in tf.trainable_variables() if 'quantile_' not in v.name]
        weights_list = [v for v in weights_list if 'bias' not in v.name]
        check_list = [v for v in weights_list if 'weights' not in v.name and 'kernel' not in v.name]

        assert len(check_list) == 0, 'Unrecognised variable type {}- need to check if should\
                                      be regulated'.format(check_list[0])

        reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list)

        if multi_network:
            reg_loss = tf.divide(reg_loss, self.num_networks)

        return reg_loss

    def add_quantile_outputs(self, output, final_layer_size, features, targets):

        self.q_opts = []

        if self.quantile_input:
            output = tf.concat([output, features], axis=2)
            q_in_size = self.features_dim + final_layer_size
        else:
            q_in_size = final_layer_size

        for q in self.multi_quantile:

            with tf.variable_scope('quantile_' + str(q)):

                if len(self.quantile_hidden_nodes) > 0:
                    # Hidden layers
                    hid_out = multi_nn(output, q_in_size, self.num_networks,
                                       self.quantile_hidden_nodes, self.w_init, self.b_init,
                                       self.activation_fn)
                    o_size = self.quantile_hidden_nodes[-1]

                else:
                    hid_out = output
                    o_size = q_in_size

                # Output layer and losses
                o = bootstrap_linear_layer(hid_out, o_size, self.targets_dim, activation_fn=None,
                                           num_networks=self.num_networks, w_init=self.f_w_init,
                                           b_init=self.f_b_init, name='final_' + str(q))

                if q < 0.5:
                    out = self.output['main'] - o
                else:
                    out = self.output['main'] + o

                self.output[q] = out

                loss = self.loss_fns[q](targets, out)

                # Optimizer
                opt = self.bootstrap_optimizer(loss, key=q, lr=self.quantile_learning_rate)
                self.q_opts.append(opt)

    def predict(self, features=None, targets=None, pred_batch_size=50):
        """
        Returns the predictions of the network for "features". Scales features first if required
        Args
        features: np array of shape [n, num_features]
        Returns
        PREDS: a list of length num_networks, holding the predictions of each individual network
        """
        # Scale the features
        if self.feature_scaler and features is not None:
            features = self.feature_scaler(features)
        if self.target_scaler and targets is not None:
            targets = self.target_scaler(targets)

        # Add the lags
        if self.features_lags is not None or self.targets_lags is not None:
            features, targets, lags = add_lags(features_data=features, targets_data=targets,
                                               features_lags=self.features_lags,
                                               targets_lags=self.targets_lags, return_max=True)

        # Restore the best model
        self.restore_best_vars(save=False)

        if self.net_type == 'RNN':
            # Create a data iterator for the features
            f, t, s_l = prepare_rnn_data(features, targets, data_set='test',
                                         num_steps=self.num_steps, silent=True)

            # Tile data into shape [num_networks, b_length, num_steps, target/features_dim]
            if self.four_d:
                f = expand_batches(f, num_networks=self.num_networks)
                t = expand_batches(t, num_networks=self.num_networks)

            # Make masks of all ones (so no data masked out)
            masks = [np.ones([self.num_networks, 1, self.num_steps, 1]) for _ in t]
            iterator = rnn_batch_iterator(f, t, s_l, masks=masks, four_d=self.four_d)

        elif self.net_type == 'NN':
            if targets is None:
                targets = np.ones(shape=[self.num_networks, features.shape[0], 1])
            else:
                targets = np.tile(np.expand_dims(targets, axis=0), [self.num_networks, 1, 1])
            features = np.tile(np.expand_dims(features, axis=0), [self.num_networks, 1, 1])
            iterator = bootstrap_batch_iterator(features, targets, batch_size=pred_batch_size)

        # Get the predictions of the network
        rts = self.run_data_set(iterator)
        preds = rts[0]

        # Add nans to the start of the preds if we lost targets due to adding lags
        if self.features_lags is not None or self.targets_lags is not None:
            nans = np.empty([lags, 1])
            nans[:] = np.nan
            for p in preds.keys():
                for n in range(self.num_networks):
                    preds[p][n] = np.concatenate([nans, preds[p][n]], axis=0)

        # Add ensemble predictions
        preds['ensemble'] = {}
        for k in self.keys:
            preds['ensemble'][k] = get_avs(targets=None, preds=preds[k], b_i=self.best_indices[k], loss_fn=None)

        return preds

    def record_losses(self, tr, v, te, step):
        """
        Records the losses at each step
        Args
        tr, v, te: train, val and test losses respectively
        step: the step at which we have evaluated
        """
        if not hasattr(self, 'loss_record'):
            self.loss_record = {'train': [], 'val': [], 'test': [], 'steps': []}
        inputs = (tr, v, te, step)
        for num, key in enumerate(['train', 'val', 'test', 'steps']):
            self.loss_record[key].append(inputs[num])

    def add_rescaled_losses(self):
        """
        Adds the (rescaled if necessary) losses once training is complete
        """
        def get_losses(iterator, preds):
            targets = iterator.all_targets(resc=self.rev_target_scaler)
            losses = []
            if hasattr(self, 'var_network') and self.var_network:
                # Just do the MSE loss (not the max likelihood loss)
                losses.append(np_mse(targets, preds[0]))
            else:
                if self.quantile:
                    loss = np_quantile_loss(targets, preds, quantile=self.quantile)
                else:
                    loss = np_mse(targets, preds)
                losses.append(loss)

            return np.average(losses)

        # Use the train_inf iterator if the network is an RNN
        if hasattr(self, 'cell_type'):
            train_iter = self.train_inf_iter
        else:
            train_iter = self.train_iter

        self.TRAIN_LOSS = get_losses(train_iter, self.TRAIN_PREDS)
        self.VAL_LOSS = get_losses(self.val_iter, self.VAL_PREDS)
        self.TEST_LOSS = get_losses(self.test_iter, self.TEST_PREDS)

        if hasattr(self, 'train_val_iter'):
            self.TRAIN_VAL_LOSS = get_losses(self.train_val_iter, self.TRAIN_VAL_PREDS)

    def save(self, key=None):
        """
        Save a model checkpoint, with 'key' appended to the checkpoint name if required
        """
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if key:
            self.saver.save(self.sess, self.checkpoint_dir + '/' + self.model_name + '-' + str(key))
        else:
            self.saver.save(self.sess, self.checkpoint_dir + '/' + self.model_name)

    # Methods for the bootstrap NNs below this point
    def train_fixed(self, data_method):
        """
        Checks if the train and val sets are the same for every network for this data drawing method
        Args
        data_method: the method used to draw the data for each separate network.
                     One of ['simple_block', 'fixed_at_end', 'random_choice', 'basic_bootstrap', 'fixed_val_bootstrap']
        """
        # Deal with case where data method is tuple (second value is block length)
        if type(data_method) == tuple:
            data_method = data_method[0]

        if data_method in ['fixed_at_end']:
            self.train_set = 'fixed'
        elif data_method in ['basic_bootstrap', 'simple_block', 'overlapping_block', 'random_choice', 'fixed_val_bootstrap', 'wild_bootstrap',
                             'wild_bootstrap_all', 'all_bootstrap', 'train_bootstrap']:
            self.train_set = 'variable'
        else:
            print(data_method)
            raise Exception('Unrecognised method')

    def update_t_vars(self, key, current_vars, step):
        """
        Updates the numpy version of the variables if any of the individual networks beat their best val loss
        Args
        key: which output of the network to update (either 'main' or a float denoting quantile)
        current_vars: the numpy variable values from the current train step
        step: how many steps of training have been carried out so far
        """
        # Initialize variables
        if not hasattr(self, 'best_vars'):
            self.best_vars = {}
            for v in self.trainable_vars:
                shape = v.shape.as_list()
                self.best_vars[v.name] = np.zeros(shape=shape)
            self.var_names = [v.name for v in self.trainable_vars]

        # Return 1 for variables which should be updated and 0 otherwise
        if key == 'main':
            upd_vars = [1 if 'quantile_' not in v else 0 for v in self.var_names]
        else:
            upd_vars = [1 if 'quantile_' + str(key) in v else 0 for v in self.var_names]

        # If loss was lowest for this bootstrap with these variables, update our new best variables
        for b in range(self.num_networks):
            if self.best_step[key][b] == step:
                # Update the variables
                for ix, v in enumerate(self.var_names):
                    if upd_vars[ix] == 1:
                        self.best_vars[v][b] = current_vars[ix][b]

    def restore_best_vars(self, save=True):
        """
        Restores the variable values from the epoch in which each network achieved its lowest val loss
        Args
        save: whether or not to save a checkpoint file after the variables have been restored
        """
        # Assign the new vars which we have created to the variables in the original graph
        all_assign_ops = [v.assign(self.best_vars[v.name]) for num, v in enumerate(self.trainable_vars)]
        self.sess.run(all_assign_ops)

        if save:
            # Save the new combined best variables and delete the null old checkpoints
            self.save(key='best')

    def print_loss(self, losses, names, keys):
        """
        Helper function for printing the losses from the different outputs of the network
        Args:
        losses: list of dictionaries, each holding the losses from one of the datasets
        names: list of names of the datasets ('Train', 'Val' etc.)
        keys: list of the outputs to print the losses for ('Main', '0.05' etc)
        """
        for k in keys:
            if type(k) == float:
                o = str(k) + ' quantile'
            else:
                o = 'Main'
            print(o, end=' --- ')
            for ix, l in enumerate(losses):
                if ix + 1 == len(losses):
                    end = '\n'
                else:
                    end = ', '
                print(names[ix], 'Loss: {:.3f}'.format(np.average(l[k])), end=end)

    def all_outputs_avs(self, targets, predictions, best_indices):
        """
        Calculate the ensemble predictions for all the different outputs of the network
        targets: np array of target values
        predictions: dictionary, holding predictions for each of the outputs of the network
        best_indices: dictionary, holding the indices of the networks which achieve lowest val loss
                      for each of the outputs of the network
        """
        preds = {}
        losses = {}

        for k in self.keys:
            preds[k], losses[k] = get_avs(targets, predictions[k], best_indices[k], self.np_loss_fns[k])

        return preds, losses

    def average_preds(self):
        """
        Adds the average predictions of all the networks as an attribute to self, and calculates the loss associated
        with this average prediction. Drops percentage self.average_drop of the worst networks before doing this
        """
        if not hasattr(self, 'TRAIN_VAL_PREDS'):
            raise Exception('Need to train the network first')

        self.best_indices = get_best_indices(self.average_drop, self.TRAIN_VAL_LOSSES)

        tr_v_targets = self.train_val_iter.all_targets(resc=self.rev_target_scaler)
        te_targets = self.test_iter.all_targets(resc=self.rev_target_scaler)

        self.TRAIN_VAL_PREDS['ensemble'], self.TRAIN_VAL_LOSSES['ensemble'] = self.all_outputs_avs(tr_v_targets, self.TRAIN_VAL_PREDS,
                                                                               self.best_indices)

        self.TEST_PREDS['ensemble'], self.TEST_LOSSES['ensemble'] = self.all_outputs_avs(te_targets, self.TEST_PREDS,
                                                                     self.best_indices)

        # Check if train/val data is the same for every network - no point in comparing train losses for each
        # network if the data is different for every network
        if self.train_set == 'fixed':
            if self.net_type == 'RNN':
                # For RNN, need to get the train and val targets first
                all_targets = self.train_val_iter.all_targets(resc=self.rev_target_scaler)
                tr_targets = all_targets[:self.train_val_iter.train_len, :]
                val_targets = all_targets[self.train_val_iter.train_len:, :]

                self.TRAIN_PREDS['ensemble'], self.TRAIN_LOSSES['ensemble'] = self.all_outputs_avs(tr_targets, self.TRAIN_PREDS,
                                                                                self.best_indices)
                self.VAL_PREDS['ensemble'], self.VAL_LOSSES['ensemble'] = self.all_outputs_avs(val_targets, self.VAL_PREDS,
                                                                            self.best_indices)
            else:
                # NN
                sc = self.rev_target_scaler
                self.TRAIN_PREDS['ensemble'], self.TRAIN_LOSSES['ensemble'] = self.all_outputs_avs(self.train_iter.all_targets(resc=sc),
                                                                               self.TRAIN_PREDS, self.best_indices)
                self.VAL_PREDS['ensemble'], self.VAL_LOSSES['ensemble'] = self.all_outputs_avs(self.val_iter.all_targets(resc=sc),
                                                                           self.VAL_PREDS, self.best_indices)

        print('\nEnsemble Losses')
        self.print_loss(losses=[self.TEST_LOSSES['ensemble']], names=['Test'], keys=self.keys)
        # print("Average Test Loss (rescaled): {:.4f}".format(self.av_test_loss['main']))


def get_best_indices(av_drop, choice_losses):
    """
    Returns the indices of the networks with the best losses in the choice_losses list (generally best val loss)
    Args
    av_drop: ratio of the number of networks with the worst loss which we're dropping
    choice_losses: a dict where keys are the different outputs of the network and values are lists
                   of scalar losses - we pick the best networks out of these
    Returns
    best_indices: dict of the indices of the best networks for each of the different outputs
    """
    # Get the number of networks which have been trained
    num_nets = len(choice_losses['main'])

    # Drop those networks with the worst val loss
    num_to_drop = int(num_nets * av_drop)

    best_indices = {}

    for k in choice_losses.keys():
        best_indices[k] = list(np.array(choice_losses[k]).argsort()[:num_nets - num_to_drop])

    return best_indices


def get_avs(targets, preds, b_i, loss_fn):
    """
    Gets new predictions as the average of the predictions of all the best networks at each point
    Args
    targets: np array of targets
    preds: list of predictions for each separate network for this data set
    b_i: list of the best indices (i.e. normally the top 90% or so of networks by best val loss)
    loss_fn: the loss function to use to calculate the loss for the ensembles predictions
    Returns
    av_preds: the new predictions - an ensemble of the best network's predictions
    av_loss: the loss of our new ensemble predictions
    """
    av_preds = np.average([preds[i] for i in b_i], axis=0)

    if loss_fn is not None:
        av_loss = loss_fn(targets, av_preds)
        return av_preds, av_loss
    else:
        return av_preds

