import tensorflow as tf


def net(X, width, height):

    # Net parameters
    n_input = width * height
    n_output = width * height

    n_hidden_1 = 3000
    n_hidden_2 = 2000
    n_hidden_3 = 1500
    n_hidden_4 = 2000
    # n_hidden_5 = 1500
    # n_hidden_6 = 2000

    # Store layers weights & biases
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.05)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.05)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.05)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.05)),
        # 'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5], stddev=0.05)),
        # 'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6], stddev=0.05)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_output], stddev=0.05))
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.0, shape=[n_hidden_1])),
        'b2': tf.Variable(tf.constant(0.0, shape=[n_hidden_2])),
        'b3': tf.Variable(tf.constant(0.0, shape=[n_hidden_3])),
        'b4': tf.Variable(tf.constant(0.0, shape=[n_hidden_4])),
        # 'b5': tf.Variable(tf.constant(0.0, shape=[n_hidden_5])),
        # 'b6': tf.Variable(tf.constant(0.0, shape=[n_hidden_6])),
        'out': tf.Variable(tf.constant(0.0, shape=[n_output]))
    }

    # Hidden 1 layer with RELU activation
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden 2 layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # layer_5 = tf.nn.relu(layer_5)
    #
    # layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    # layer_6 = tf.nn.relu(layer_6)

    # Output layer with SIGMODE activation
    out_layer = tf.add(tf.matmul(layer_4, weights['out']), biases['out'])

    out_layer = tf.nn.dropout(out_layer, 0.8)

    out_layer = tf.nn.sigmoid(out_layer)

    return out_layer
