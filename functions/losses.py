import numpy as np
import tensorflow as tf


def np_mse(targets, predictions, mask=None):
    assert targets.shape == predictions.shape, 'Targets and predictions arrays not the same shape'
    residuals = targets - predictions
    residuals_squared = residuals ** 2
    if mask is not None:
        assert mask.shape == residuals_squared.shape
        residuals_squared = mask * residuals_squared
        loss = np.sum(residuals_squared) / np.sum(mask)
    else:
        loss = np.average(residuals_squared)
    return loss


def np_mae(targets, predictions, mask=None):
    assert targets.shape == predictions.shape, 'Targets and predictions arrays not the same shape'
    residuals = np.abs(targets - predictions)
    if mask is not None:
        assert mask.shape == residuals.shape
        residuals = mask * residuals
        loss = np.sum(residuals) / np.sum(mask)
    else:
        loss = np.average(residuals)

    return loss


def np_ql(targets, predictions):
    assert targets.shape == predictions.shape, 'Targets and predictions arrays not the same shape'
    ratio = np.true_divide(targets, predictions)
    ql_ind = ratio + np.log(predictions)
    ql_inner = np.average(ql_ind)
    return ql_inner


class np_quantile_loss(object):
    def __init__(self, quantile):
        self.quantile = quantile

    def __call__(self, targets, predictions, mask=None):
        """
        Calculates the quantile loss.
        Args
        targets: tf constant holding the targets
        predictions: tf constant holding the predictions of the network
        quantile: the quantile at which to evaluate the loss
        Returns
        final_loss: the average loss for this dataset
        """
        assert targets.shape == predictions.shape, 'Targets and predictions should be the same size'

        # Indices of predictions which are under or over the targets
        i_under = targets > predictions
        i_over = predictions >= targets
        indices_under = i_under.astype(np.float32)
        indices_over = i_over.astype(np.float32)

        # Absolute loss
        abs_loss = np.abs(targets - predictions)

        # Weight the losses by the quantiles
        u_losses = indices_under * abs_loss * self.quantile
        o_losses = indices_over * abs_loss * (1 - self.quantile)

        # Mask out over/under values
        u_losses_red = u_losses * i_under
        o_losses_red = o_losses * i_over

        j_losses = u_losses_red + o_losses_red

        if mask is not None:
            assert mask.shape == j_losses.shape
            j_losses = mask * j_losses
            final_loss = np.sum(j_losses) / np.sum(mask)
        else:
            final_loss = np.average(j_losses)

        return final_loss


def all_losses(targets, preds):
    save_dict = {'mse': np_mse(targets, preds),
                 'ql': np_ql(targets, preds),
                 'mae': np_mae(targets, preds)}
    return save_dict


# TF Losses ########################################################################################################


def tf_check_shapes(tensor1, tensor2):
    """
    Checks the shape of the two tensors is equal - needs to be called within tf.control_dependencies
    Args
    tensor1/2: the two tensors whose shapes we're checking are identical
    Returns
    assertion: the tf assertion
    """
    sh_1 = tf.shape(tensor1)
    sh_2 = tf.shape(tensor2)

    return [tf.assert_equal(sh_1, sh_2)]


def tf_mse(targets, predictions, mask=None, average=True):
    """
    TF mean squared error function
    Args
    targets: tensor of targets
    predictions: tensor of predictions
    mask: a tensor of 1s and 0s, used for masking out the validation data from the loss for bootstrap networks
    average: if False, return all the losses separately as opposed to averaging, as will be getting rid of some
             of them in the bootstrap_NN
    Returns
    loss: the average mean squared error
    """
    with tf.control_dependencies(tf_check_shapes(targets, predictions)):
        difference = targets - predictions

    if mask is not None:
        assert average is True, 'Shouldn\'t apply both mask weighting and weighting in NN model'
        with tf.control_dependencies(tf_check_shapes(difference, mask)):
            difference = mask * difference
        loss = tf.reduce_sum(tf.pow(difference, 2)) / tf.reduce_sum(mask)
    else:
        loss = tf.pow(difference, 2)
        if average:
            loss = tf.reduce_mean(loss)

    return loss


def tf_mae(targets, predictions, mask=None, average=True):
    """
    TF mean absolute error function
    Args
    targets: tensor of targets
    predictions: tensor of predictions
    mask: a tensor of 1s and 0s, used for masking out the validation data from the loss for bootstrap networks
    average: if False, return all the losses separately as opposed to averaging, as will be getting rid of some
             of them in the bootstrap_NN
    Returns
    loss: the average mean absolute error
    """
    with tf.control_dependencies(tf_check_shapes(targets, predictions)):
        difference = targets - predictions

    if mask is not None:
        assert average is True, 'Shouldn\'t apply both mask weighting and weighting in NN model'
        with tf.control_dependencies(tf_check_shapes(difference, mask)):
            difference = mask * difference
        loss = tf.reduce_sum(tf.abs(difference)) / tf.reduce_sum(mask)
    else:
        loss = tf.abs(difference)
        if average:
            loss = tf.reduce_mean(loss)

    return loss


class tf_quantile_loss(object):
    """
    Calculates the quantile loss. If huber is true, applies a transformation to ensure
    differentiability around the origin.
    Args
    targets: tf constant holding the targets
    predictions: tf constant holding the predictions of the network
    quantile: the quantile at which to evaluate the loss
    huber: boolean which indicates whether or not to apply a huber transformation
    epsilon: the level below which residuals need to fall for the huber transformation to be applied
    mask: a mask indicating which values are train set, and which are val. Masks out the val points from the loss
    average: if False, return all the losses separately as opposed to averaging, as will be getting rid of some
             of them in the bootstrap_NN
    Returns
    final_loss: the average loss for this dataset
    """
    def __init__(self, quantile, huber=True, epsilon=0.00001):

        self.quantile = quantile
        self.huber = huber
        self.epsilon = epsilon

    def __call__(self, targets, predictions, mask=None, average=True):

        with tf.control_dependencies(tf_check_shapes(targets, predictions)):

            # Indices of predictions which are under or over the targets
            i_under = targets > predictions

        i_over = predictions >= targets
        indices_under = tf.to_float(i_under)
        indices_over = tf.to_float(i_over)

        # Absolute loss
        abs_loss = tf.abs(targets - predictions)

        if self.huber:
            # Get indices of losses which are below the epsilon threshold
            under_eps = tf.to_float(abs_loss <= self.epsilon)
            over_eps = tf.to_float(abs_loss > self.epsilon)
            # Apply the huber update
            abs_loss = abs_loss - (under_eps * abs_loss) + (under_eps * (tf.pow(abs_loss, 2) / (2.0 * self.epsilon)))
            abs_loss = abs_loss - (over_eps * abs_loss) + (over_eps * (abs_loss - self.epsilon / 2.0))

        # Weight the losses by the quantiles
        weight_tensor = (indices_under * self.quantile) + (indices_over * (1 - self.quantile))
        f_losses = abs_loss * weight_tensor

        if mask is not None:
            # Mask out the val data
            with tf.control_dependencies(tf_check_shapes(f_losses, mask)):
                f_losses = mask * f_losses
            # Weight the average by the sum of the mask
            final_loss = tf.reduce_sum(f_losses) / tf.reduce_sum(mask)
        else:
            if average:
                final_loss = tf.reduce_mean(f_losses)
            else:
                # Return the losses separately as will be getting rid of some of them in the bootstrap_NN
                final_loss = f_losses

        return final_loss

tf_loss_lookup = {
    'mse': tf_mse,
    'mae': tf_mae
}

np_loss_lookup = {
    'mse': np_mse,
    'mae': np_mae
}