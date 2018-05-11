import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict

from dcgan_ops import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class BGAN(object):
    def __init__(self,
                 x_dim = [2],
                 z_dim = 2,
                 hidden_size=128,
                 batch_size=64, prior_std=1.0, J=1, M=1,
                 num_classes=1, alpha=0.01, lr=0.0002,
                 optimizer='adam', ml=False):

        self.dataset_size = 10000
        self.batch_size = batch_size
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.optimizer = optimizer.lower()
        self.G_hidden_size = hidden_size
        self.D_hidden_size = 128

        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_mcmc = M
        self.alpha = alpha
        self.lr = lr

        self.ml = ml
        if self.ml:
            assert self.num_gen * self.num_mcmc == 1, "cannot have multiple generators in ml mode"

        self.weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, self.G_hidden_size)),
                                        ("g_h0_lin_b", (self.G_hidden_size,)),
                                        ("g_h1_lin_W", (self.G_hidden_size, self.G_hidden_size)),
                                        ("g_h1_lin_b", (self.G_hidden_size,)),
                                        ("g_lin_W", (self.G_hidden_size, self.x_dim[0])),
                                        ("g_lin_b", (self.x_dim[0],))])

        self.sghmc_noise = {}
        self.noise_std = np.sqrt(2 * self.alpha)
        for name, dim in self.weight_dims.items():
            self.sghmc_noise[name] = tf.contrib.distributions.Normal(loc=0., scale=self.noise_std * tf.ones(
                self.weight_dims[name]))

        self.K = num_classes  # 1 means unsupervised, label == 0 always reserved for fake

        self.build_bgan_graph()

    def _get_optimizer(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

    def build_bgan_graph(self):

        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size] + self.x_dim, name='real_images')

        self.labeled_inputs = tf.placeholder(tf.float32,
                                             [self.batch_size] + self.x_dim, name='real_images_w_labels')

        self.labels = tf.placeholder(tf.float32,
                                     [self.batch_size, self.K + 1], name='real_targets')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.gen_param_list = []
        with tf.variable_scope("generator") as scope:
            for gi in range(self.num_gen):
                for m in range(self.num_mcmc):
                    gen_params = AttributeDict()
                    for name, shape in self.weight_dims.items():
                        gen_params[name] = tf.get_variable("%s_%04d_%04d" % (name, gi, m),
                                                           shape, initializer=tf.random_normal_initializer(stddev=0.1))
                    self.gen_param_list.append(gen_params)

        self.D, self.D_logits = self.discriminator(self.inputs, self.K + 1)
        self.Dsup, self.Dsup_logits = self.discriminator(self.labeled_inputs, self.K + 1, reuse=True)

        if self.K == 1:
            # regular GAN
            constant_labels = np.zeros((self.batch_size, 2))
            constant_labels[:, 1] = 1.0
            self.d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits,
                                                                                      labels=tf.constant(
                                                                                          constant_labels)))
        else:
            self.d_loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Dsup_logits,
                                                                                     labels=self.labels))
            self.d_loss_real = -tf.reduce_mean(tf.log((1.0 - self.D[:, 0]) + 1e-8))

        self.generation = defaultdict(list)
        for gen_params in self.gen_param_list:
            self.generation["g_prior"].append(self.gen_prior(gen_params))
            self.generation["g_noise"].append(self.gen_noise(gen_params))
            self.generation["generators"].append(self.generator(self.z, gen_params))
            self.generation["gen_samplers"].append(self.sampler(self.z, gen_params))
            D_, D_logits_ = self.discriminator(self.generator(self.z, gen_params), self.K + 1, reuse=True)
            self.generation["d_logits"].append(D_logits_)
            self.generation["d_probs"].append(D_)

        d_loss_fakes = []

        constant_labels = np.zeros((self.batch_size, self.K + 1))
        constant_labels[:, 0] = 1.0
        for d_logits_ in self.generation["d_logits"]:
            d_loss_fakes.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_,
                                                                                           labels=tf.constant(
                                                                                               constant_labels))))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]


        self.d_loss_fake = tf.reduce_mean(d_loss_fakes)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_vars = []
        for gi in range(self.num_gen):
            for m in range(self.num_mcmc):
                self.g_vars.append([var for var in t_vars if 'g_' in var.name and "_%04d_%04d" % (gi, m) in var.name])

        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])
        d_opt = self._get_optimizer(self.d_learning_rate)
        self.d_optim = d_opt.minimize(self.d_loss, var_list=self.d_vars)

        d_opt_adam = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=0.5)
        self.d_optim_adam = d_opt_adam.minimize(self.d_loss, var_list=self.d_vars)

        self.g_optims, self.g_optims_adam = [], []
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])
        for gi in range(self.num_gen * self.num_mcmc):
            g_loss = - tf.reduce_mean(tf.log((1.0 - self.generation["d_probs"][gi][:, 0]) + 1e-8))
            g_loss += self.generation["g_noise"][gi]
            self.generation["g_losses"].append(g_loss)
            g_opt = self._get_optimizer(self.g_learning_rate)
            self.g_optims.append(g_opt.minimize(g_loss, var_list=self.g_vars[gi]))
            g_opt_adam = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
            self.g_optims_adam.append(g_opt_adam.minimize(g_loss, var_list=self.g_vars[gi]))

    def discriminator(self, x, K, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(linear(x, self.D_hidden_size, 'd_lin_0'), leak=0)
            h1 = linear(h0, K, 'd_lin_1')
            return tf.nn.softmax(h1), h1

    def sup_discriminator(self, x, K, reuse=False):

        pass

    def generator(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            h0 = lrelu(linear(z, self.G_hidden_size, 'g_h0_lin',
                              matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b),
                       leak=0)
            h1 = lrelu(linear(h0, self.G_hidden_size, 'g_h1_lin',
                              matrix=gen_params.g_h1_lin_W, bias=gen_params.g_h1_lin_b),
                       leak=0)
            self.x_ = linear(h1, self.x_dim[0], 'g_lin',
                             matrix=gen_params.g_lin_W, bias=gen_params.g_lin_b)
            return self.x_

    def sampler(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.generator(z, gen_params)

    def gen_prior(self, gen_params):
        with tf.variable_scope("generator") as scope:
            prior_loss = 0.0
            for var in gen_params.values():
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))

        prior_loss /= self.dataset_size

        return prior_loss

    def gen_noise(self, gen_params):  # for SGHMC
        with tf.variable_scope("generator") as scope:
            noise_loss = 0.0
            for name, var in gen_params.items():
                noise_loss += tf.reduce_sum(var * self.sghmc_noise[name].sample())
        noise_loss /= self.dataset_size
        return noise_loss

    def disc_prior(self):
        with tf.variable_scope("discriminator") as scope:
            prior_loss = 0.0
            for var in self.d_vars:
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))

        prior_loss /= self.dataset_size

        return prior_loss

    def disc_noise(self):  # for SGHMC
        with tf.variable_scope("discriminator") as scope:
            noise_loss = 0.0
            for var in self.d_vars:
                noise_ = tf.contrib.distributions.Normal(loc=0., scale=self.noise_std * tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())
        noise_loss /= self.dataset_size
        return noise_loss
