import sys
sys.path.append("/Users/liupeng/Documents/dl2tcc/DeepLearning2TCC/CNN")
sys.path.append("/Users/liupeng/Documents/dl2tcc/DeepLearning2TCC/utils")
sys.path.append("/Users/liupeng/Documents/dl2tcc/DeepLearning2TCC/DataSet")

import tensorflow as tf

import vgg_based

import input_data
# import cnn_simple
import utils

# data set file
train_data_dir = "../../DataSet/DataSetFiles/TrainSet/Kai_train_50_50_200_npy.npy"
train_target_dir = "../../DataSet/DataSetFiles/TrainSet/Qigong_train_50_50_200_npy.npy"

test_data_dir = "../../DataSet/DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"
test_target_dir = "../../DataSet/DataSetFiles/TestSet/Kai_test_50_50_40_npy.npy"

# validation size
VALIDATION_SIZE = 50

# train data set files
train_dir = {"train": {"data": train_data_dir, "target": train_target_dir},
             "test": {"data": test_data_dir, "target": test_target_dir}}

with tf.Session() as sess:

    data_set = input_data.read_data_sets(train_dir, validation_size=10)

    images = tf.placeholder(tf.float32, [None, 2500])
    true_out = tf.placeholder(tf.float32, [None, 2500])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg_based.Vgg19(train_dir)
    print("train data size:", len(data_set.train.data), data_set.train.data.shape)
    print("train target size:", len(data_set.train.target), data_set.train.target.shape)
    print("test data size:", len(data_set.test.data), data_set.test.data.shape)
    print("test target size:", len(data_set.test.target), data_set.test.target.shape)
    # exit()
    print(data_set.train.data[10])

    vgg.build(images, train_mode)

    sess.run(tf.global_variables_initializer())

    # simple 1-step training
    loss_op = tf.reduce_mean(tf.abs(vgg.prob - true_out))
    optimizer_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)

    # accuracy
    accuracy_op = tf.reduce_mean(tf.abs(vgg.prob - true_out))

    # Before train
    acc = sess.run(accuracy_op, feed_dict={images: data_set.test.data,
                                        true_out: data_set.test.target,
                                        train_mode: False})
    print("Before accuracy:", acc)
    # exit()

    for step in range(50):

        x_batch, y_batch = vgg.data_dict.train.next_batch(100)
        sess.run(optimizer_op, feed_dict={images: x_batch,
                                        true_out: y_batch,
                                        train_mode: True})
        if step % 5 == 0:
            acc = sess.run(accuracy_op, feed_dict={images: data_set.test.data,
                                        true_out: data_set.test.target,
                                        train_mode: False})
            print("Step %d : %d" % (step, acc))


    # After train
    acc = sess.run(accuracy_op, feed_dict={images: data_set.test.data,
                                        true_out: data_set.test.target,
                                        train_mode: False})
    print("After accuracy:", acc)
    # vgg.save_npy(sess, "./test-save.npy")


