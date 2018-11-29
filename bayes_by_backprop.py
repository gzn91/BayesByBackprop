import tensorflow as tf
import numpy as np


def batchnflat(x, y, mb_size, flatten=False):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    while True:
        ind = np.random.choice(inds, mb_size, replace=False)
        x_mb = x[ind]
        y_mb = y[ind].reshape(mb_size, )
        if flatten:
            x_mb = np.reshape(x_mb, (mb_size, np.prod(x[0].shape)))
        else:
            x_mb = x_mb[..., None]

        yield x_mb, y_mb


class BayesByBackprop(object):

    def __init__(self, shape=(28, 28, 1), target_dim=10, lr=1e-3, model='cnn'):

        # placeholders
        self.input_ph = tf.placeholder(tf.float32, (None, *shape), 'input-ph')
        self.target_ph = tf.placeholder(tf.int32, (None,), 'target-ph')
        self.M = tf.placeholder(tf.float32, (), 'batch-scaling-ph')

        self.scaling = None  # used to feed self.M
        self.flatten = False  # use image representation

        # stddevs
        stddev1 = 2 * tf.to_float(np.sqrt(1. / mb_size))
        stddev2 = tf.to_float(np.sqrt(1. / mb_size))
        loss_stddev = 5e-2  # for scaling the likelihood loss

        def log_gaussian(x, mean, stddev):
            return (-0.5 * np.log(2 * np.pi) - tf.log(stddev) - tf.square(x - mean) /
                    (2 * tf.square(stddev)))

        def sample_params(shape, scale=-.3):

            mu = tf.Variable(tf.random_normal(shape, 0, 0.05))
            rho = tf.Variable(tf.random_normal(shape, scale, 0.2))

            return mu, tf.nn.softplus(rho)

        def fc_var_layer(x, shape, activate=True):

            # sample means and stddev
            w_mu, w_std = sample_params(shape, scale=-4.)

            # sample noise
            eps = tf.random_normal(shape)

            # sample weights with gaussian reparameterization trick
            w = w_mu + w_std * eps
            b_mu, b_std = sample_params((1, shape[-1]))

            # calculate loss terms
            pi = 0.25  # scaled mixture prior
            log_p = pi * tf.reduce_sum(log_gaussian(w, 0., stddev1))
            log_p += (1 - pi) * tf.reduce_sum(log_gaussian(w, 0., stddev2))
            log_q = tf.reduce_sum(log_gaussian(w, tf.stop_gradient(w_mu), tf.stop_gradient(w_std)))
            neg_kl = log_p - log_q

            h = tf.matmul(x, w) + b_mu

            if activate:
                h = tf.nn.leaky_relu(h)

            return h, neg_kl

        def conv_var_layer(x, shape, kernel_size=[3, 3], activate=True):

            # sample means and stddev
            shape = (*kernel_size, *shape)
            w_mu, w_std = sample_params(shape, scale=-4.)

            # sample noise
            eps = tf.random_normal(shape)

            # sample weights with gaussian reparameterization trick
            w = w_mu + w_std * eps
            b_mu, b_std = sample_params((1, shape[-1]))

            # calculate loss terms
            pi = 0.25  # scaled mixture prior
            log_p = pi * tf.reduce_sum(log_gaussian(w, 0., stddev1))
            log_p += (1 - pi) * tf.reduce_sum(log_gaussian(w, 0., stddev2))
            log_q = tf.reduce_sum(log_gaussian(w, tf.stop_gradient(w_mu), tf.stop_gradient(w_std)))
            neg_kl = log_p - log_q

            h = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME') + b_mu

            if activate:
                h = tf.nn.leaky_relu(h)

            return h, neg_kl

        def cnn_model(state, layer_filters=[32, 32, 32]):

            x = state
            neg_kl_sum = 0
            # hidden layers
            for nfilters in layer_filters:
                shape = (x.get_shape().as_list()[-1], nfilters)
                x, neg_kl = conv_var_layer(x, shape)
                neg_kl_sum += neg_kl
            # linear output layer
            x = tf.layers.flatten(x)
            shape = (x.get_shape().as_list()[-1], target_dim)
            logits, neg_kl = fc_var_layer(x, shape, activate=False)

            neg_kl_sum += neg_kl
            return logits, neg_kl_sum

        def fc_model(state, layer_units=[128, 128, 64]):

            x = state
            neg_kl_sum = 0
            # hidden layers
            for nunits in layer_units:
                shape = (x.get_shape().as_list()[-1], nunits)
                x, neg_kl = fc_var_layer(x, shape)
                neg_kl_sum += neg_kl
            shape = (x.get_shape().as_list()[-1], target_dim)
            # linear output layer
            logits, neg_kl = fc_var_layer(x, shape, activate=False)

            neg_kl_sum += neg_kl
            return logits, neg_kl_sum

        if model == 'cnn':
            self.logits, neg_kl = cnn_model(self.input_ph)
        else:
            self.flatten = True
            self.logits, neg_kl = fc_model(self.input_ph)

        # loss fn
        log_like_loss = tf.reduce_sum(log_gaussian(tf.one_hot(self.target_ph, target_dim), self.logits, loss_stddev))
        self.loss = -(neg_kl / self.M + log_like_loss / mb_size)

        # optimizer
        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params, name='grads')
        grads, _grad_norm = tf.clip_by_global_norm(grads, 5.0)
        grads_and_vars = list(zip(grads, params))
        self.global_step = tf.train.get_or_create_global_step()
        self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads_and_vars, global_step=self.global_step)

        self.sess = tf.get_default_session()

        tf.global_variables_initializer().run()

    def train(self, dataset, mb_size):
        normalize = lambda x: (x - np.mean(x, axis=0)) / (1e-8 + np.std(x, axis=0))

        (x_train, y_train), (x_test, y_test) = dataset()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # scale to zero mean unit variance
        x_train = normalize(x_train)
        x_test = normalize(x_test)

        ntest = x_test.shape[0] // mb_size

        self.scaling = x_train.shape[0] / mb_size

        data = batchnflat(x_train, y_train, mb_size, flatten=self.flatten)
        test_data = batchnflat(x_test, y_test, mb_size, flatten=self.flatten)

        for i in range(100000):
            x, y = next(data)
            loss, _ = self.sess.run([self.loss, self.optimize],
                                    feed_dict={self.M: self.scaling, self.target_ph: y, self.input_ph: x})
            if i % 100 == 0:
                print(f'training loss: {loss:.2f}')
            if i % 1000 == 0:
                self.evaluate(test_data, ntest)

    def evaluate(self, data, n):

        acc = []
        losses = []
        for i in range(n):
            x, y = next(data)
            loss, yhat = self.sess.run([clf.loss, clf.logits],
                                       feed_dict={clf.M: self.scaling, self.target_ph: y, self.input_ph: x})
            acc.append(np.sum(np.argmax(yhat, -1).T == y) / yhat.shape[0])
            losses.append(loss)
        print(f'evaluating at step {self.sess.run(self.global_step) - 1}')
        print(f'test loss: {np.mean(losses):.3f}')
        print(f'test accuracy: {np.mean(acc):.3f}')


if __name__ == '__main__':
    dataset = tf.keras.datasets.mnist

    np.random.seed(1)

    # shape = (784,)  # fc
    shape = (28, 28, 1)  # cnn

    mb_size = 32

    with tf.Session().as_default():
        clf = BayesByBackprop(shape=shape, target_dim=10)
        clf.train(dataset.load_data, mb_size)
