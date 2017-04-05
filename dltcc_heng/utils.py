import tensorflow as tf


def _net(inputs, phase_train, img_width, img_height):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    channels = [1, 8, 8, 8,
                8, 16, 16, 16,
                16, 32, 32, 32, 32, 32, 32, 32,
                32, 64, 64, 64, 64, 64, 64, 64,
                64, 128, 128, 128, 128, 128, 128, 128]

    net = {}

    current = tf.reshape(inputs, [-1, img_width, img_height, 1], name="x_reshape")

    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            current = _conv_layer(inputs=current, inputs_channels=channels[i], filter_size=3,
                                  output_channels=channels[i+1], phase_train=phase_train)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _max_pool_layer(current)
        net[name] = current

    assert len(net) == len(layers)
    return net


# Convolutional layer
def _conv_layer(inputs, input_channels, filter_size, output_channels, phase_train=True):

    shape = [filter_size, filter_size, input_channels, output_channels]

    weights = _weights(shape=shape)
    biases = _biases(length=output_channels)

    layer = tf.nn.conv2d(input=inputs, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    layer += biases

    # batch normalization
    layer = _batch_norm(layer, phase_train=phase_train)
    return layer


def _max_pool_layer(inputs):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")


def _avg_pool_layer(inputs):
    return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding="SAME")


def _weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))


def _biases(length):
    return tf.Variable(tf.constant(0.01, shape=[length]))


def _normalize_func(x, minVal, maxVal, newMinVal=0, newMaxVal=1):
    result = (x-minVal)*newMaxVal/(maxVal-minVal) + newMinVal
    return result


def _batch_norm(input, phase_train, scope="batch_normal"):
    with tf.name_scope(scope):
        out_filters = input.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[out_filters]), name="beta", trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_filters]), name="gamma", trainable=True)
        batch_mean, batch_var = tf.nn.moments(input, [0, 1, 2], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normaled = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
        return normaled