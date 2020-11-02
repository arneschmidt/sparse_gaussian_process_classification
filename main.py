import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tensorflow.keras.utils import to_categorical
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

def generate_data(samples_per_class):
    number_of_classes = 3
    class_means = [[-1, -1],
                   [0, 1],
                   [1, 0]]
    class_means_2 = [[1, 1],
                   [1, -1],
                   [-1.5, 1.5]]
    class_covariances = [[[0.1, 0.06], [0.06, 0.2]],
                         [[0.2, 0.06], [0.06, 0.1]],
                         [[0.1, 0], [0, 0.1]]]
    plot_colours = ['gx', 'rx', 'bx']
    x = []
    y = []
    np.random.seed(42)
    plt.figure()
    for i in range(number_of_classes):
        x_samples = np.random.multivariate_normal(class_means[i], class_covariances[i], int(samples_per_class/2))
        x_samples_2 = np.random.multivariate_normal(class_means_2[i], class_covariances[i], int(samples_per_class/2))
        y_targets = np.full(shape=(samples_per_class), fill_value=i)
        x.append(x_samples)
        x.append(x_samples_2)
        y.append(to_categorical(y_targets, num_classes=number_of_classes))
        plt.plot(x_samples[...,0], x_samples[...,1], plot_colours[i])
        plt.plot(x_samples_2[...,0], x_samples_2[...,1], plot_colours[i])
    # plt.show()

    x = np.array(x).reshape(number_of_classes*samples_per_class, 2)
    y = np.array(y).reshape(number_of_classes*samples_per_class, number_of_classes)

    return x, y

class RBFKernelFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(1),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(1),
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
    num_inducing_points = 8
    num_classes = 3
    epochs_per_step = 5
    num_steps = 20
    mode = 'test'
    save_name = 'test'

    x,y = generate_data(samples_per_class)

    tensor_fn = tfp.distributions.Distribution.sample
    inputs = tf.keras.layers.Input(shape=[2], batch_size=300) #, batch_size=config["model"]["batch_size"]),
    # tf.keras.layers.Dense(10, activation='relu'),
    vgp = tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(),
        event_shape=[num_classes], # output dimensions
        inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
            minval=-2.0, maxval=2.0, seed=None
        ),
        jitter=10e-3,
        convert_to_tensor_fn=tensor_fn,
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(np.array(0.54).astype(np.float32))),
    )
    outputs = vgp(inputs)

    if mode == 'mc':
        outputs_softmax = tf.keras.layers.Softmax()(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs_softmax, name="vgp")
        # model.add_loss(kl_loss)
        loss = 'categorical_crossentropy'
    elif mode == 'vi':
        loss = lambda y, rv_y: rv_y.variational_loss(
            y, kl_weight=np.array(x.shape[0], x.dtype) / x.shape[0])
    else:
        tfd = tfp.distributions
        outputs_categorical = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Categorical(logits=t), convert_to_tensor_fn=tfp.distributions.Distribution.sample)(outputs)
        # out_mean = outputs_categorical.mean()
        model = tf.keras.Model(inputs=inputs, outputs=outputs_categorical, name="vgp")
        def log_probab(observations, f):
            cat = tfd.Independent(tfd.Categorical(logits=f), reinterpreted_batch_ndims=1)

            # independent.Independent(
            #     normal.Normal(loc=fn_vals, scale=scale),
            #     reinterpreted_batch_ndims=1).log_prob(obs)
            return cat.log_prob(observations)
        loss = lambda y, rv_y: outputs.variational_loss(y, kl_weight=np.array(x.shape[0], x.dtype) / x.shape[0],
                                                                    log_likelihood_fn = log_probab)

    model.build(input_shape=[2, 300])
    tfd = tfp.distributions
    # loss = lambda y, rv_y: - rv_y.submodules[2].surrogate_posterior_expected_log_likelihood(y,log_likelihood_fn = log_probab, quadrature_size=20) + rv_y.submodules[2].surrogate_posterior_kl_divergence_prior()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                  loss=loss,
                  # loss='categorical_crossentropy',
                  metrics=['accuracy'])
    for i in range(num_steps):
        show_inducing_locations(x, model, num_inducing_points, num_classes, samples_per_class, epochs=i*epochs_per_step,
                                save_name=save_name)
        model.fit(x=x, y=y, epochs=epochs_per_step)


def show_inducing_locations(x, model, num_inducing_points, num_classes, num_samples_per_class,epochs,
                            save_name='figures', input_dim=2):
    # plot = plot.copy()
    weights = model.layers[1].get_weights()
    inducing_locations = weights[2].reshape(num_inducing_points*num_classes, input_dim)
    inducing_observations = weights[3].reshape(num_inducing_points*num_classes)
    # plot.plot(inducing_locations[...,0], inducing_locations[...,1], 'bs')

    marker_data = ['gx', 'rx', 'bx']
    marker_inducing_points= ['g', 'r', 'b']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(num_classes):
        x_class = x[i*num_samples_per_class:(i+1)*num_samples_per_class]
        ax.plot(x_class[:, 0], x_class[:, 1], marker_data[i])
        plot_inducing_points(ax, i, inducing_locations, inducing_observations, num_inducing_points, marker_inducing_points)
    plot_variance(ax, model)
    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))

    os.makedirs(save_name, exist_ok=True)
    plt.savefig('./'+save_name +'/' + str(epochs) + '.jpg', dpi=300)
    plt.close(fig)

def plot_inducing_points(plot, class_id, inducing_locations, inducing_observations, num_inducing_points, marker_inducing_points):
    inducing_locations_class = inducing_locations[class_id * num_inducing_points:(class_id + 1) * num_inducing_points]
    inducing_observations_class = inducing_observations[class_id * num_inducing_points:(class_id + 1) * num_inducing_points]

    for point in range(num_inducing_points):
        indication = inducing_observations_class[point]
        if indication > 0:
            marker = 'P'
        else:
            marker = 'D'
        markersize = min(abs(indication), 1) * 10
        plot.plot(inducing_locations_class[point, 0], inducing_locations_class[point, 1], color=marker_inducing_points[class_id],
                  markersize=markersize, marker=marker)


def plot_variance(plot, model):
    # make these smaller to increase the resolution
    x = np.arange(-3.0, 3.0, 0.1)
    x_1, x_2 = np.array(meshgrid(x,x))
    grid = np.stack((x_1, x_2))
    grid = grid.T.reshape(-1, 2)
    num_samples = 20
    outs = []
    for i in range(num_samples):
        outs.append(model.predict(grid))
    outs = np.array(outs)
    std = np.mean(np.std(outs, axis=0), axis=-1)

    std_plot = std.reshape(x.shape[0], x.shape[0])
    plot.contourf(x, x, std_plot,cmap= 'Greys', norm=Normalize(), alpha=0.7)
    num_classes = 3
    thresholds = [0.9, 1]
    colours = ['green', 'blue', 'red']
    for class_id in range(num_classes):
        class_pred = np.mean(outs, axis=0)[:,class_id]
        class_pred = class_pred.reshape(x.shape[0], x.shape[0])
        plot.contourf(x, x, class_pred,  colors=[colours[class_id]], levels=thresholds ,alpha=0.2)
    # plt.show()


if __name__ == '__main__':
    main()

    # def log_probab(observations, f):
    #     cat = tfd.Independent(tfd.Categorical(logits=f), reinterpreted_batch_ndims=1)
    #     return cat.log_prob(observations)
    # loss = lambda y, rv_y: rv_y.submodules[2].variational_loss(y, kl_weight=np.array(x.shape[0], x.dtype) / x.shape[0],
    #                                                             log_likelihood_fn = log_probab)

    # def log_probab(observations, f):
    #     cat = tfd.Independent(tfd.Categorical(logits=f), reinterpreted_batch_ndims=1)
    #     return cat.log_prob(observations)

        # independent.Independent(
        #     normal.Normal(loc=fn_vals, scale=scale),
        #     reinterpreted_batch_ndims=1).log_prob(obs)

        # Compute the expected log likelihood using Gauss-Hermite quadrature.
    # vgp = model.layers[1].submodules[5]
    # recon = vgp.surrogate_posterior_expected_log_likelihood(
    #     y,
    #     log_likelihood_fn=log_prob,
    #     quadrature_size=20)
    # elbo = -recon + vgp.surrogate_posterior_kl_divergence_prior()
    # # model.layers[1].submodules[5].variational_loss
    # model.add_loss(elbo)