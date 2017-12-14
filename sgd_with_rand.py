from keras.optimizers import Optimizer
import keras.backend as K
import numpy as np
import tensorflow as tf
import copy


def clip_noise(g, numupdates):

    # tf require using a special op to multiply IndexedSliced by scalar
    if K.backend() == 'tensorflow':
        noise_expression = tf.random_normal(g.get_shape(), 0.0, 1.0/(numupdates+1))
        # saving the shape to avoid converting sparse tensor to dense
        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(noise_expression, tf.Tensor):
            g_shape = copy.copy(noise_expression.get_shape())
        elif isinstance(noise_expression, tf.IndexedSlices):
            g_shape = copy.copy(noise_expression.dense_shape)
        g = tf.add(g, noise_expression)

        if isinstance(noise_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(noise_expression, tf.IndexedSlices):
            g._dense_shape = g_shape

    return g


class SGD_with_rand(Optimizer):

    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD_with_rand, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.iter = 0

    # @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        # ----------------------------- ADDED BY NIR ---------------------------
        for g in grads:
            g = clip_noise(g, self.iter)
        self.iter = self.iter + 1
        # -------------------------- END OF ADDED BY NIR -----------------------

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_with_rand, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


