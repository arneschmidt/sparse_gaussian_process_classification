import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def generate_data(samples_per_class):
    number_of_classes = 3
    class_means = [[0, 0],
                   [0, 1],
                   [1, 0]]
    class_covariances = [[[0.1, 0.06], [0.06, 0.2]],
                         [[0.2, 0.06], [0.06, 0.1]],
                         [[0.1, 0], [0, 0.1]]]
    plot_colours = ['g^', 'ro', 'bs']
    x = []
    y = []
    np.random.seed(42)
    plt.figure()
    for i in range(number_of_classes):
        x_samples = np.random.multivariate_normal(class_means[i], class_covariances[i], samples_per_class)
        y_targets = np.full(shape=(samples_per_class), fill_value=i)
        x.append(x_samples)
        y.append(to_categorical(y_targets, num_classes=number_of_classes))
        plt.plot(x_samples[...,0], x_samples[...,1], plot_colours[i])
    #plt.show()

    x = np.array(x).reshape(number_of_classes*samples_per_class, 2)
    y = np.array(y).reshape(number_of_classes*samples_per_class, number_of_classes)

    return x, y

class RBFKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(1.0 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(1.0 * self._length_scale) # 5.
        )


def main():
    samples_per_class = 100
    num_inducing_points = 10
    number_of_classes = 3
    x,y = generate_data(samples_per_class)

    tensor_fn = tfp.distributions.Distribution.sample
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[2]), #, batch_size=config["model"]["batch_size"]),
            tf.keras.layers.Dense(10, activation='relu'),
            tfp.layers.VariationalGaussianProcess(
                num_inducing_points=num_inducing_points,
                kernel_provider=RBFKernelFn(),
                event_shape=[number_of_classes], # output dimensions
                inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                    minval=0.0, maxval=1.0, seed=None
                ),
                jitter=10e-3,
                convert_to_tensor_fn=tensor_fn,
                # unconstrained_observation_noise_variance_initializer=(
                #     tf.constant_initializer(np.array(0.54).astype(np.float32))),
            ),
            #tf.keras.layers.Softmax()
            # tfp.layers.DistributionLambda(
            # make_distribution_fn=lambda t: tfp.distributions.Normal(
            #     loc=t[..., 0], scale=tf.exp(t[..., 1])),
            # convert_to_tensor_fn=lambda s: s.sample(5))

    ])
    loss = lambda y, rv_y: rv_y.variational_loss(
        y, kl_weight=np.array(x.shape[0], x.dtype) / x.shape[0])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
                  loss=loss,
                  metrics=['accuracy'])

    model.fit(x=x, y=y, epochs=10)
    print('stop')
    model.fit(x=x, y=y, epochs=10)
    print('stop')





if __name__ == '__main__':
    main()
