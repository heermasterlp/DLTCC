from __future__ import absolute_import
from DLTCC import dltcc_models
from ImageDisplay import ImageDisplay

import tensorflow as tf

checkpoint_dir = ""

IMG_WIDTH = 250
IMG_HEIGHT = 250

def inference(images):

    if images is None:
        print("The images is none!")
        return

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, [None, IMG_WIDTH * IMG_HEIGHT])
        dltcc = Dltcc()
        dltcc.build(x)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Checkpoint found")
            else:
                print("No checkpoint found")

            # Run the model to predictions
            predictions = sess.run(dltcc.pred, feed_dict={x: images})

            show_image(predictions)




