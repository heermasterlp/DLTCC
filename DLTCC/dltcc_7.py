from __future__ import absolute_import
import tensorflow as tf
import numpy as np

SIZE = 100
IMAGE_WIDTH = SIZE
IMAGE_HEIGHT = SIZE

class Dltcc7(object):
    def __init__(self):
        pass

    # Build the models
    def build(self, inputs):
        if inputs is None:
            print("Input should not none!")

        self.x_reshape = tf.reshape(inputs, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1], name="x_reshape")

        # Conv 1
        with tf.name_scope("conv1"):
            self.conv1 = conv_layer(input=self.x_reshape, input_channels=1, filter_size=3, output_channels=16, use_pooling=True)

        with tf.name_scope("conv2"):
            self.conv2 = conv_layer(input=self.conv1, input_channels=16, filter_size=3, output_channels=16, use_pooling=True)

            # Flatten layer
        with tf.name_scope("flatten1"):
            self.layer_flat, self.num_flat_features = flatten_layer(self.conv2)

        with tf.name_scope("fc_layer"):
            self.layer_fc1 = new_fc_layer(input=self.layer_flat, num_inputs=self.num_flat_features,
                                     num_outputs=IMAGE_WIDTH * IMAGE_HEIGHT * 4, use_sigmoid=True)

            self.layer_fc2 = new_fc_layer(input=self.layer_fc1, num_inputs=IMAGE_WIDTH * IMAGE_HEIGHT * 4,
                                     num_outputs=IMAGE_WIDTH * IMAGE_HEIGHT, use_sigmoid=True)

            # Predict
        with tf.name_scope("probability"):
            # layer_dropped = tf.nn.dropout(layer_fc2, keep_prob=1.0)
            self.y_prob = tf.sigmoid(self.layer_fc2)

            print("Build models end!")
