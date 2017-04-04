import tensorflow as tf


def normalize_func(x, minVal, maxVal, newMinVal=0, newMaxVal=1):
    result = (x-minVal)*newMaxVal/(maxVal-minVal) + newMinVal
    return result


def conv2d_block(X, shape, strides, padding, scope='conv2d'):
    with tf.name_scope(scope):
        if not strides:
            strides = [1, 1, 1, 1]
        outfilters = shape[-1]
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[outfilters]), name='biases')
        cnn_layer = tf.nn.conv2d(X, W, strides, padding) + b
        return cnn_layer


def max_pool(X, scope='max_pool'):
    with tf.name_scope(scope):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], padding='SAME')


def batch_norm(X, phase_train, scope='batch_normal'):
    with tf.name_scope(scope):
        outFilters = X.get_shape()[-1]
        beta = tf.Variable(tf.constant(0.0, shape=[outFilters]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[outFilters]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity([batch_var])
        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normaled = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
        return normaled


def Leak_relu(X, alpha):
    return tf.maximum(X, alpha*X)


def block(X, shape, phase_train, strides=None, padding='SAME', scope='_block'):
    with tf.name_scope(scope):
        cnn = conv2d_block(X, shape, strides, padding)
        cnn_bn = batch_norm(cnn, phase_train)
        # if FLAGS.alpha < 0:
        #     _relu = tf.nn.relu(cnn_bn)
        # else:
        #     _relu = Leak_relu(cnn_bn, FLAGS.alpha)
        return tf.nn.relu(cnn_bn)


def construct_network(X, size, infilters, outfilters, laywers, phase_train,
                      strides=None, scope='construct_network'):
    with tf.name_scope(scope):
        # The first laywer:
        conv1 = block(X, [size, size, infilters, outfilters], phase_train, strides,
                      scope="conv1_%dx%d" % (outfilters, outfilters))
        cur_conv = conv1
        for i in range(laywers - 1):
            the_next_laywer = block(cur_conv, [size, size, outfilters, outfilters], phase_train, strides,
                                    scope='conv%d_%dx%d' % (i + 2, outfilters, outfilters))
            cur_conv = the_next_laywer
        return cur_conv


def total_variation_loss(x, side):
    """
        Total variation loss for regularization of image smoothness
        """
    loss = tf.nn.l2_loss(x[:, 1:, :, :] - x[:, :side - 1, :, :]) / side + \
           tf.nn.l2_loss(x[:, :, 1:, :] - x[:, :, :side - 1, :]) / side
    return loss