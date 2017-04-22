from __future__ import absolute_import

import tensorflow as tf
from ops import *


class DltccHeng(object):
    def __init__(self, ngf=64, image_width=200, image_height=40):
        self.ngf = ngf
        self.image_width = image_width
        self.image_height = image_height

    # Build the models
    def build(self, inputs):
        if inputs is None:
            print("Input should not none!")
        self.x_reshape = tf.reshape(inputs, [-1, self.image_width, self.image_height, 1], name="x_reshape")

        with tf.variable_scope("conv_layers"):
            # conv_1: [batch, 200, 4, 1] => [batch, 100, 20, ngf]
            self.conv1_1 = conv2d(self.x_reshape, self.ngf*2, stride=1, name="conv1_1")
            self.conv1_2 = conv2d(self.conv1_1, self.ngf*2, stride=1, name="conv1_2")
            self.max_pool1 = maxpool2d(self.conv1_2, k=2)

            # [batch, 100, 20, ngf*2] => [batch, 50, 10, ngf*4]
            self.conv2_1 = conv2d(self.max_pool1, self.ngf * 4, stride=1, name="conv2_1")
            self.conv2_2 = conv2d(self.conv2_1, self.ngf * 4, stride=1, name="conv2_2")
            self.conv2_3 = conv2d(self.conv2_2, self.ngf * 4, stride=1, name="conv2_3")
            self.conv2_4 = conv2d(self.conv2_3, self.ngf * 4, stride=1, name="conv2_4")
            self.max_pool2 = maxpool2d(self.conv2_4, k=2)

            # [batch, 50, 10, ngf*4] => [batch, 25, 5, ngf*8]
            self.conv3_1 = conv2d(self.max_pool2, self.ngf * 8, stride=1, name="conv3_1")
            self.conv3_2 = conv2d(self.conv3_1, self.ngf * 8, stride=1, name="conv3_2")
            self.conv3_3 = conv2d(self.conv3_2, self.ngf * 8, stride=1, name="conv3_3")
            self.conv3_4 = conv2d(self.conv3_3, self.ngf * 8, stride=1, name="conv3_4")
            self.max_pool3 = maxpool2d(self.conv3_4, k=2)

        with tf.variable_scope("deconv_layers"):
            # [batch, 25, 5, ngf*4] => [batch, 50, 10, ngf*2]
            self.deconv3 = deconv2d(self.max_pool3, self.ngf * 2, name="deconv3")
            self.de_batchnorm3 = batchnorm(self.deconv3, name="de_batchnorm3")
            self.de_rectified3 = tf.nn.relu(self.de_batchnorm3, name="de_rectified3")

            # [batch, 50, 10, ngf*2] => [batch, 100, 20, ngf]
            self.deconv2 = deconv2d(self.de_rectified3, self.ngf, name="deconv2")
            self.de_batchnorm2 = batchnorm(self.deconv2, name="de_batchnorm2")
            self.de_rectified2 = tf.nn.relu(self.de_batchnorm2, name="de_rectified2")

            # [batch, 100, 20, ngf] => [batch, 200, 40, 1]
            self.deconv1 = deconv2d(self.de_rectified2, out_channels=1, name="deconv1")
            self.de_batchnorm1 = batchnorm(self.deconv1, name="de_batchnorm1")
            self.out = tf.nn.sigmoid(self.de_batchnorm1, name="out")

            # FC layers
            # self.layer_flat, self.num_flat_features = flatten_layer(self.deconv_out)
            # # FC layer 1
            # self.layer_fc1 = new_fc_layer(self.layer_flat, num_inputs=self.num_flat_features,
            #                               num_outputs=1000)
            #
            # # self.layer_fc2 = new_fc_layer(self.layer_fc1, num_inputs=1000, num_outputs=1000)
            # # self.layer_fc3 = new_fc_layer(self.layer_fc2, num_inputs=1000, num_outputs=1000)
            #
            # self.layer_fc2 = new_fc_layer(self.layer_fc1, num_inputs=1000,
            #                               num_outputs=self.image_width*self.image_height)
            # # out
            # self.out = tf.nn.sigmoid(self.layer_fc2)

