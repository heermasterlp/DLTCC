import tensorflow as tf


def net(X, width, height):

    # Net parameters
    n_input = width * height
    n_output = width * height

    n_hidden_1 = 3000
    n_hidden_2 = 2000

    # Store layers weights & biases
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    # Hidden 1 layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden 2 layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with SIGMODE activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    out_layer = tf.nn.dropout(out_layer, 0.8)

    out_layer = tf.nn.sigmoid(out_layer)

    return out_layer