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


def train():
    # Data set
    data_set = input_data.read_data_sets(train_dir, validation_size=2)

    if data_set is None:
        print("data sets are none!")
        return
    else:
        print("data set train data shape:{}".format(data_set.train.data.shape))
        print("data set train target shape:{}".format(data_set.train.target.shape))
        print("data set test data shape:{}".format(data_set.test.data.shape))
        print("data set test target shape:{}".format(data_set.test.target.shape))


    print("Start build models")
    # place variable
    x = tf.placeholder("float", [None, n_input])
    y_true = tf.placeholder("float", [None, n_output])

    # Models
    y_pred = multilayer_perceptron(x, weights, biases)

    # Loss
    with tf.device("cpu:0"):
        cost_op = tf.reduce_mean(tf.abs(y_true - y_pred))
        optimizer_op = tf.train.RMSPropOptimizer(0.1).minimize(cost_op)

    print("Build models end!")

    # initialize variable
    init_op = tf.global_variables_initializer()

    # save the models and checkpoints.
    # the formatting: (models) models-date.ckpt, (checkpoint) checkpoint-date-step.ckpt
    saver = tf.train.Saver()

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    now = datetime.datetime.now()
    today = "{}-{}-{}".format(now.year, now.month, now.day)

    # Train models

    with tf.Session() as sess:
        start_time = time.time()

        sess.run(init_op)

        # Train the models
        for epoch in range(training_epochs):
            x_batch = data_set.train.data
            y_batch = data_set.train.target

            _, cost = sess.run([optimizer_op, cost_op], feed_dict={x: x_batch,
                                                                       y_true: y_batch})

            if epoch % display_step == 0:
                    print("Epoch {0} : {1}".format(epoch, cost))

        duration = time.time() - start_time

        # Save the trained models.
        saver.save(sess, os.path.join(model_path, "models-{}".format(today)))
        print("Training end!{}".format(duration))


if __name__ == '__main__':
    train()

