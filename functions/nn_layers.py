import tensorflow as tf


def bootstrap_linear_layer(in_tensor, in_size, out_size, activation_fn, num_networks, w_init,
                           b_init, name):
    """
    Linear network layer for training multiple networks at once
    Args
    in_tensor: tensor of shape [num_networks, prev_layer_size, layer_size]
    in_size: size of the previous layer
    out_size: size of this layer
    activation_fn: activation function for the layer
    num_networks: the number of networks we're training at once
    w_init/b_init: initiation parameters for the weights and biases respectively
    name: string, scope under which layer created
    Returns
    layer_inner: output of the layer, tensor of shape [num_networks, batch_size, out_size]
    """

    weights = tf.get_variable('weights_layer_' + name, [num_networks, in_size, out_size],
                              dtype=tf.float32, initializer=w_init)
    bias = tf.get_variable('bias_layer_' + name, [num_networks, 1, out_size],
                           dtype=tf.float32, initializer=b_init)

    layer_inner = tf.matmul(in_tensor, weights) + bias

    if activation_fn is not None:
        return activation_fn(layer_inner)
    else:
        return layer_inner


def multi_nn(in_tensor, in_size, num_networks, num_hidden_nodes, w_init, b_init,
             activation_fn):
    """
    GPU efficient implementation of multiple separate neural nets
    Args
    in_tensor: the input to the network - size [num_nets, batch_size, input_size]
    in_size: the 2nd dimension of the input tensor (normally num_features)
    num_networks: the number of separate networks to create
    num_hidden_nodes: list, holding the sizes of each of the hidden layers
    w_init/b_init: weight and bias initialization parameters for the hidden layers
    activation_fn: the activation function for the hidden layers
    Returns
    layer: tensor of shape [num_networks, batch_size, output_size] -- output of the last hidden layer
    """

    num_layers = len(num_hidden_nodes)

    # Hidden layers
    layer_list = []

    for layer in range(num_layers):

        if layer == 0:
            input_size = in_size
            input_tensor = in_tensor
        else:
            input_size = num_hidden_nodes[layer - 1]
            input_tensor = layer_list[layer - 1]

        layer_out = bootstrap_linear_layer(input_tensor, input_size, num_hidden_nodes[layer], activation_fn,
                                           num_networks, w_init=w_init, b_init=b_init, name=str(layer))

        layer_list.append(layer_out)

    return layer_out

