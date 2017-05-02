import tensorflow as tf
import math


weight_dim = [1000, 1000, 1000, 1000, 1000, 1000]
bias_dim = weight_dim


class DltccHeng(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.hidden_layers = []
        self.output = None

    def build_model(self, images):
        input_dim = self.width * self.height

        output_dim = input_dim

        h1_layer = hidden_layer(images, input_dim, weight_dim[0])
        self.hidden_layers.append(h1_layer)

        for index in range(len(weight_dim) - 1):
            layer = hidden_layer(self.hidden_layers[-1], weight_dim[index], weight_dim[index + 1])
            self.hidden_layers.append(layer)

        self.output = hidden_layer(self.hidden_layers[-1], weight_dim[-1], output_dim)

        self.output = tf.nn.sigmoid(self.output)


# Hidden layers
def hidden_layer(images, input_dim, output_dim):
    weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1.0),
                              name='weights')
    biases = tf.Variable(tf.zeros([output_dim]), name='biases')

    layer = tf.add(tf.matmul(images, weights), biases)

    return layer
