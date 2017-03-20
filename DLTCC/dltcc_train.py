from __future__ import absolute_import

import tensorflow as tf

from dltcc_models import Dltcc

import input_data

IMAGE_WIDTH = 250
IMAGE_HEIGHT = 250

BATCH_SIZE = 80

# data set file
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_250_250_400_train.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_250_250_400_train.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_250_250_40_test.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Qigong_250_250_40_test.npy"

# Checkpoint dir
Checkpoint_dir = "../../models/"

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

MAX_EPOCH_STEPS = 10

# Train the models
def train():

    data_set = input_data.read_data_sets(train_dir, validation_size=40)
    if data_set is None:
        print('data_set is None')
        return

    print('train data:', len(data_set.train.data))
    print('train target:', len(data_set.train.target))
    print('test data:', len(data_set.test.data))
    print('test target:', len(data_set.test.target))

    # model saver
    # saver = tf.train.Saver()


    with tf.name_scope("Build"):
        # Train the models
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT])
            y_true = tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT])

            # model object
            dltcc_model = Dltcc()

            dltcc_model.build(x)

            # with tf.device('/cpu:0'):
            with tf.device('/gpu:0'):
                # loss operation
                loss_op = tf.reduce_mean(tf.abs(dltcc_model.pred - y_true))

                optimizer_op = tf.train.RMSPropOptimizer(0.1).minimize(loss_op)

                # accuracy operation
                accuracy_op = 1 - tf.reduce_mean(tf.abs(dltcc_model.pred - y_true))

            # initialize the parameters
            sess.run(tf.global_variables_initializer())

            # The accuracy of before training
            acc = sess.run(accuracy_op, feed_dict={x: data_set.test.data,
                                                   y_true: data_set.test.target})
            print("Before accuracy:", acc)

            # Train models
            for step in range(MAX_EPOCH_STEPS):

                x_batch, y_batch = data_set.next_batch(BATCH_SIZE)

                sess.run(optimizer_op, feed_dict={x: x_batch, y_true: y_batch})
                if step % 100 == 0:
                    acc = sess.run(accuracy_op, feed_dict={x: data_set.test.data,
                                                           y_true: data_set.test.target})
                    print("Step {0} : {1} ".format(step, acc))

            # The accuracy of after training
            acc = sess.run(accuracy_op, feed_dict={x: data_set.test.data,
                                                   y_true: data_set.test.target})
            print("After accuracy:", acc)

            # save the trained model
            # saver.save(sess, Checkpoint_dir + 'model.ckpt')
            print("Save models finished!")


# Test the models.
def test():
    data_set = input_data.read_data_sets(train_dir, validation_size=40)
    if data_set is None:
        print('data_set is None')
        return
    print('train data:', len(data_set.train.data))
    print('train target:', len(data_set.train.target))
    print('test data:', len(data_set.test.data))
    print('test target:', len(data_set.test.target))

    # saver of models
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, Checkpoint_dir + 'model.ckpt')


if __name__ == '__main__':
    print('Test the models')
    train()