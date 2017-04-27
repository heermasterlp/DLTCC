from __future__ import absolute_import

import tensorflow as tf
from ops import *


class DltccHeng(object):
    def __init__(self, ngf=64, image_width=256, image_height=256):
        self.ngf = ngf
        self.image_width = image_width
        self.image_height = image_height

    # Build the models
    def build(self, inputs):
        if inputs is None:
            print("Input should not none!")
        self.x_reshape = tf.reshape(inputs, [-1, self.image_width, self.image_height, 1], name="x_reshape")

        with tf.variable_scope("conv_layers"):
            # conv_1: [batch, 256, 256, 1] => [batch, 256, 256, ngf]
            self.conv1_1 = conv2d(self.x_reshape, self.ngf, stride=1, name="conv1_1", use_batchnorm=False)
            self.max_pool1 = maxpool2d(self.conv1_1, k=2)

            # [batch, 128, 128, ngf] => [batch, 64, 64, ngf]
            self.conv2_1 = conv2d(self.max_pool1, self.ngf, stride=1, name="conv2_1")
            # self.conv2_2 = conv2d(self.conv2_1, self.ngf, stride=1, name="conv2_2")
            self.max_pool2 = maxpool2d(self.conv2_1, k=2)

            # [batch, 64, 64, ngf] => [batch, 32, 32, ngf*2]
            self.conv3_1 = conv2d(self.max_pool2, self.ngf * 2, stride=1, name="conv3_1")
            # self.conv3_2 = conv2d(self.conv3_1, self.ngf * 2, stride=1, name="conv3_2")
            self.max_pool3 = maxpool2d(self.conv3_1, k=2)

            # [batch, 32, 32, ngf*2] => [batch, 16, 16, ngf*4]
            self.conv4_1 = conv2d(self.max_pool3, self.ngf * 4, stride=1, name="conv4_1")
            # self.conv4_2 = conv2d(self.conv4_1, self.ngf * 4, stride=1, name="conv4_2")
            self.max_pool4 = maxpool2d(self.conv4_1, k=2)

            # [batch, 16, 16, ngf*4] => [batch, 8, 8, ngf *4]
            self.conv5_1 = conv2d(self.max_pool4, self.ngf * 4, stride=1, name="conv5_1")
            # self.conv5_2 = conv2d(self.conv5_1, self.ngf * 4, stride=1, name="conv5_2")
            self.max_pool5 = maxpool2d(self.conv5_1, k=2)

            # [batch, 8, 8, ngf*4] => [batch, 4, 4, ngf*4]
            self.conv6_1 = conv2d(self.max_pool5, self.ngf * 4, stride=1, name="conv6_1")
            # self.conv6_2 = conv2d(self.conv6_1, self.ngf * 4, stride=1, name="conv6_2")
            self.max_pool6 = maxpool2d(self.conv6_1, k=2)
            # [batch, 4, 4, ngf*8] => [batch, 2, 2, ngf*8]
            self.conv7_1 = conv2d(self.max_pool6, self.ngf * 8, stride=1, name="conv7_1")
            # self.conv7_2 = conv2d(self.conv7_1, self.ngf * 4, stride=1, name="conv7_2")
            self.max_pool7 = maxpool2d(self.conv7_1, k=2)
            # [batch, 2, 2, ngf*8] => [batch, 1, 1, ngf*8]
            self.conv8_1 = conv2d(self.max_pool7, self.ngf * 8, stride=1, name="conv8_1")
            # self.conv8_2 = conv2d(self.conv8_1, self.ngf * 4, stride=1, name="conv8_2")
            self.max_pool8 = maxpool2d(self.conv8_1, k=2)

        with tf.variable_scope("deconv_layers"):
            # [batch, 1, 1, ngf*8] => [batch, 2, 2, ngf*8]
            self.deconv7_1 = deconv2d(self.max_pool8, self.ngf * 8, name="deconv7_1")
            self.de_batchnorm7_1 = batchnorm(self.deconv7_1, name="de_batchnorm7_1")
            self.de_rectified7_1 = tf.nn.relu(self.de_batchnorm7_1, name="de_rectified7_1")

            # [batch, 2, 2, ngf*8] => [batch, 4, 4, ngf*4]
            self.deconv6_1 = deconv2d(self.de_rectified7_1, self.ngf * 4, name="deconv6_1")
            self.de_batchnorm6_1 = batchnorm(self.deconv6_1, name="de_batchnorm6_1")
            self.de_rectified6_1 = tf.nn.relu(self.de_batchnorm6_1, name="de_rectified6_1")

            # [batch, 4, 4, ngf*4] => [batch, 8, 8, ngf*4]
            self.deconv5_1 = deconv2d(self.de_rectified6_1, self.ngf * 4, name="deconv5_1")
            self.de_batchnorm5_1 = batchnorm(self.deconv5_1, name="de_batchnorm5_1")
            self.de_rectified5_1 = tf.nn.relu(self.de_batchnorm5_1, name="de_rectified5_1")

            # [batch, 8, 8, ngf*4] => [batch, 16, 16, ngf*2]
            self.deconv4_1 = deconv2d(self.de_rectified5_1, self.ngf * 2, name="deconv4_1")
            self.de_batchnorm4_1 = batchnorm(self.deconv4_1, name="de_batchnorm4_1")
            self.de_rectified4_1 = tf.nn.relu(self.de_batchnorm4_1, name="de_rectified4_1")

            # [batch, 16, 16, ngf*2] => [batch, 32, 32, ngf*2]
            self.deconv3_1 = deconv2d(self.de_rectified4_1, self.ngf * 2, name="deconv3_1")
            self.de_batchnorm3_1 = batchnorm(self.deconv3_1, name="de_batchnorm3_1")
            self.de_rectified3_1 = tf.nn.relu(self.de_batchnorm3_1, name="de_rectified3_1")

            # [batch, 32, 32, ngf*2] => [batch, 64, 64, ngf]
            self.deconv2_1 = deconv2d(self.de_rectified3_1, self.ngf, name="deconv2_1")
            self.de_batchnorm2_1 = batchnorm(self.deconv2_1, name="de_batchnorm2_1")
            self.de_rectified2_1 = tf.nn.relu(self.de_batchnorm2_1, name="de_rectified2_1")

            # [batch, 64, 64, ngf] => [batch, 64, 64, ngf]
            self.deconv1_1 = deconv2d(self.de_rectified2_1, self.ngf, name="deconv1_1")
            self.de_batchnorm1_1 = batchnorm(self.deconv1_1, name="de_batchnorm1_1")
            self.de_rectified1_1 = tf.nn.relu(self.de_batchnorm1_1, name="de_rectified1_1")

            # [batch, 128, 128, ngf] => [batch, 256, 256, 1]
            self.deconv0_1 = deconv2d(self.de_rectified1_1, out_channels=1, name="deconv0_1")
            self.de_batchnorm0_1 = batchnorm(self.deconv0_1, name="de_batchnorm0_1")
            self.out = tf.nn.sigmoid(self.de_batchnorm0_1, name="out")


