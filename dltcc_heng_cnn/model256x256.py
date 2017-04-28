from __future__ import absolute_import

import tensorflow as tf
from ops import *


class DltccHeng(object):
    def __init__(self, batch_size=64, generator_dim=64, input_width=256, output_width=256, input_filters=3,
                 output_filters=3, is_training=True, reuse=False):
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.input_width = input_width
        self.output_width = output_width
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.is_training = is_training
        self.reuse = reuse

    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="g_e{}_conv".format(layer))
                enc = batch_norm(conv, is_training, scope="g_e{}_bn".format(layer))
                encode_layers["e{}".format(layer)] = enc
                return enc

            # [256, 256] => [128, 128]
            e1 = conv2d(images, self.generator_dim, scope="g_e1_conv")
            encode_layers["e1"] = e1
            # [128, 128]    =>  [64, 64]
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            # [64, 64] => [32, 32]
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            # [32, 32] => [16, 16]
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            # [16, 16] => [8, 8]
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            # [8, 8] => [4, 4]
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            # [4, 4] => [2, 2]
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            # [2, 2] => [1, 1]
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            print("s:: {}".format(s))

            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64),\
                                              int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width, output_width, output_filters],
                               scope="g_d{}_deconv".format(layer))

                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    dec = batch_norm(dec, is_training, scope="g_d{}_bn".format(layer))
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            # [1, 1] => [2, 2]
            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["e7"],
                              dropout=True)
            # [2, 2] => [4, 4]
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["e6"], dropout=True)
            # [4, 4] => [8, 8]
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["e5"], dropout=True)
            # [8, 8] => [16, 16]
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["e4"])
            # [16, 16] => [32, 32]
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["e3"])
            # [32, 32] = > [64, 64]
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["e2"])
            # [64, 64] => [128, 128]
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["e1"])
            # [128, 128] => [256, 256]
            d8 = decode_layer(d7, s, 1, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.sigmoid(d8)

            return output

    def generator(self, images, is_training, reuse=False):
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)
        output = self.decoder(e8, enc_layers, is_training=is_training, reuse=reuse)

        return output, e8

    # Build the models
    def build_model(self, images):
        if images is None:
            print("Input should not none!")

        self.x_reshape = tf.reshape(images, [self.batch_size, self.input_width, self.input_width, 1],
                                    name="x_reshape")

        self.output, self.e8 = self.generator(self.x_reshape, is_training=self.is_training, reuse=self.reuse)



