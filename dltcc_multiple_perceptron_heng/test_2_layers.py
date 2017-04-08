import tensorflow as tf
import os
import datetime
import os
import time
import input_data

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 10

# Networking parameters
image_width = 200
image_height = 40

n_input = image_width * image_height
n_output = image_width * image_height

n_hidden_1 = 3000
n_hidden_2 = 2000

# 200x40 data set
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_heng_200_40_30_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_heng_200_40_30_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_heng_200_40_11_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_heng_200_40_11_test.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

model_path = "../../checkpoints/models_200_40_mac_4_8"
checkpoint_path = "../../checkpoints/checkpoints_200_40_mac"


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


def multilayer_perceptron(x, weights, biases):

    # Hidden 1 layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden 2 layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with SIGMODE activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    out_layer = tf.nn.sigmoid(out_layer)

    return out_layer


def test():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data sets are none!")
        return

    x = tf.placeholder("float", [None, n_input])
    y_true = data_set.test.target
    y_pred = multilayer_perceptron(x, weights, biases)

    # Loss
    with tf.device("cpu:0"):
        accuracy_op = tf.reduce_mean(tf.abs(y_true - y_pred))

    print("Build models end!")

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Reload the well-trained models
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("The checkpoint models found!")
        else:
            print("The checkpoint models not found!")

        # prediction shape: [batch_size, width * height]
        accuracy = sess.run(accuracy_op, feed_dict={x: data_set.test.data})

        # Calculate the accuracy
        # avg_acc = 0
        # for ac  in accuracy:
        #     avg_acc += ac
        # avg_acc /= image_width*image_height

        print('avg accuracy:{}'.format(accuracy))


if __name__ == '__main__':
    test()

