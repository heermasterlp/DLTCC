import tensorflow as tf


class wgan_multilayer_perceptron_net(object):
    def __init__(self, X, Z, X_dim):

        self.X = X
        self.Z = Z
        self.X_dim = X_dim

        self.z_dim = X_dim  # width * height
        self.d_h1_dim = 1000
        self.d_h2_dim = 1000
        self.d_h3_dim = 1000
        self.d_h4_dim = 1000

        self.g_h1_dim = 1000
        self.g_h2_dim = 1000
        self.g_h3_dim = 1000
        self.g_h4_dim = 1000

    def build(self):
        """ Discriminator Net model """
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])

        self.D_W1 = tf.Variable(xavier_init([self.X_dim, self.d_h1_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.d_h1_dim]))

        self.D_W2 = tf.Variable(xavier_init([self.d_h1_dim, self.d_h2_dim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.d_h2_dim]))

        self.D_W3 = tf.Variable(xavier_init([self.d_h2_dim, self.d_h3_dim]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[self.d_h3_dim]))

        self.D_W4 = tf.Variable(xavier_init([self.d_h3_dim, self.d_h4_dim]))
        self.D_b4 = tf.Variable(tf.zeros(shape=[self.d_h4_dim]))

        self.D_W5 = tf.Variable(xavier_init([self.d_h4_dim, 1]))
        self.D_b5 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_W4, self.D_W5,
                   self.D_b1, self.D_b2, self.D_b3, self.D_b4, self.D_b5]

        """ Generator Net model """
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self.G_W1 = tf.Variable(xavier_init([self.z_dim, self.g_h1_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.g_h1_dim]))

        self.G_W2 = tf.Variable(xavier_init([self.g_h1_dim, self.g_h2_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.g_h2_dim]))

        self.G_W3 = tf.Variable(xavier_init([self.g_h2_dim, self.g_h3_dim]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.g_h3_dim]))

        self.G_W4 = tf.Variable(xavier_init([self.g_h3_dim, self.g_h4_dim]))
        self.G_b4 = tf.Variable(tf.zeros(shape=[self.g_h4_dim]))

        self.G_W5 = tf.Variable(xavier_init([self.g_h4_dim, self.X_dim]))
        self.G_b5 = tf.Variable(tf.zeros(shape=[self.X_dim]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_W4, self.G_W5,
                        self.G_b1, self.G_b2, self.G_b3, self.G_b4, self.G_b5]

        # Build net
        self.G_sample = self.generator(self.Z)
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample)

        self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake)
        self.G_loss = -tf.reduce_mean(self.D_fake)

        # clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

    def discriminator(self, x):
        self.D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        self.D_h2 = tf.nn.relu(tf.matmul(self.D_h1, self.D_W2) + self.D_b2)
        self.D_h3 = tf.nn.relu(tf.matmul(self.D_h2, self.D_W3) + self.D_b3)
        self.D_h4 = tf.nn.relu(tf.matmul(self.D_h3, self.D_W4) + self.D_b4)
        out = tf.matmul(self.D_h4, self.D_W5) + self.D_b5
        return out

    def generator(self, z):
        self.G_h1 = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_b1)
        self.G_h2 = tf.nn.relu(tf.matmul(self.G_h1, self.G_W2) + self.G_b2)
        self.G_h3 = tf.nn.relu(tf.matmul(self.G_h2, self.G_W3) + self.G_b3)
        self.G_h4 = tf.nn.relu(tf.matmul(self.G_h3, self.G_W4) + self.G_b4)

        self.G_log_prob = tf.matmul(self.G_h4, self.G_W5) + self.G_b5
        G_prob = tf.nn.sigmoid(self.G_log_prob)
        return G_prob


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)