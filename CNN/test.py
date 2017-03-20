import sys
sys.path.append("/Users/liupeng/Documents/python/DeepLearning2TCC/DataSet")
import tensorflow as tf
import numpy as np
import input_data

"""
    Load data set of training, testing and validation.
"""

# data set file
train_data_dir = "../DataSetFiles/TrainSet/Kai_train_50_50_200_npy.npy"
train_target_dir = "../DataSetFiles/TrainSet/Qigong_train_50_50_200_npy.npy"

test_data_dir = "../DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"
test_target_dir = "../DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}


# train progress
if __name__ == '__main__':
    def train():
        # data set
        dataset = input_data.read_data_sets(train_dir, validation_size=VALIDATION_SIZE)

        print("train data size: %d, train target size: %d" % (len(dataset.train.data), len(dataset.train.target)))
        print("validation data size: %d, validation target size: %d" % (len(dataset.validation.data), len(dataset.validation.target)))
        print("test data size: %d, test target size: %d" % (len(dataset.test.data), len(dataset.test.target)))

        # Create a multilayer model

        sess = tf.Session()

        # Input placeholders
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, 2500], name="x-input")
            y_ = tf.placeholder(tf.float32, [None, 2500], name="y-input")
        # reshape the input
        with tf.name_scope("input_shape"):
            image_shaped_input = tf.reshape(x, [-1, 50, 50, 1])
            tf.summary.image("input", image_shaped_input, 2500)

        # Conv-1
        conv1 = nn_layer(x, 2500, 500, "layer1")

        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar("drop_keep_probability", keep_prob)
            dropped = tf.nn.dropout(conv1, keep_prob)

        # Conv-2
        y = nn_layer(dropped, 500, 2500, "layer2")

        with tf.name_scope("cross_entropy"):
            diff = tf.abs(y_ - y)

            with tf.name_scope("total"):
                cross_entropy = tf.reduce_mean(diff)

        tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_prediction"):
                correct_prediction = tf.abs(y - y_)
            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar("accuracy", accuracy)

        # Merge all the summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("./train_summary", sess.graph)
        test_writer = tf.summary.FileWriter("./test_summary")

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Begin to train
        print("Begin to train models")
        for i in range(1000):
            if i % 10 == 0:
                print("train step:", i)
            summary, _ = sess.run([merged, train_step], feed_dict={x: dataset.train.data, y_: dataset.train.target, keep_prob: 1.0})
            train_writer.add_summary(summary, i)
        print("Train models end!")

        print("Begin to test models")
        for i in range(1000):
            if i % 10 == 0:
                print("test step:", i)
            summary, acc = sess.run([merged, accuracy], feed_dict={x: dataset.test.data, y_: dataset.test.target, keep_prob: 1.0})
            print("Accuracy at step %s : %s", (i, acc))
            test_writer.add_summary(summary, i)
        print("Test models end!")

        train_writer.close()
        test_writer.close()






# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def conv2d_block(X, shape, strides, padding, scope="conv2d"):
    with tf.name_scope(scope):
        if not strides:
            strides = [1, 1, 1, 1]
        out_filters = shape[-1]
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.01,), name="weights")
        b = tf.Variable(tf.constant(0.1, shape=[out_filters]), name="biases")
        cnn_result = tf.nn.conv2d(X, W, strides, padding) + b
        return cnn_result


def max_pooling(X, scope="max_pool"):
    with tf.name_scope(scope):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")






def main(_):
    train()

if __name__ == "__main__":
    tf.app.run(main=main)

