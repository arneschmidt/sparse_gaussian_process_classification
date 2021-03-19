import os
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from matplotlib.colors import Normalize
from tensorflow.keras.utils import to_categorical
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from model_architecture import create_model, compile_model


def generate_mnist_data(split_of_training_samples=1.0):
    labeled_split_arg = 'train[:' + str(int(split_of_training_samples * 100)) + '%]'
    unlabeled_split_arg = 'train[:' + str(int((1-split_of_training_samples) * 100)) + '%]'
    (ds_train_labeled, ds_train_unlabeled, ds_test), ds_info = tfds.load(
        'mnist',
        split=[labeled_split_arg, unlabeled_split_arg, 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    def preprocess(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        image = tf.cast(image, tf.float32) / 255.
        label = tf.clip_by_value(label, clip_value_min=0, clip_value_max=2)
        label = tf.dtypes.cast(label, tf.int32)

        return image, label

    def prepare(ds):
        ds = ds.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(128)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    ds_train_labeled = prepare(ds_train_labeled)
    ds_train_unlabeled = prepare(ds_train_unlabeled)
    ds_test = prepare(ds_test)

    return ds_train_labeled, ds_train_unlabeled, ds_test, ds_info




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
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    with open('config.yaml') as file:
        config = yaml.full_load(file)

    ds_train_labeled, ds_train_unlabeled, ds_test, ds_info = generate_mnist_data(config['data']['train_split'])
    model = create_model(config=config, num_classes=3, num_training_points=config['data']['train_split'])
    compile_model(config, model)
    vis_epochs = config['visualization']['epochs']
    vis_steps = int(config['model']['epochs'] / vis_epochs)
    for i in range(vis_steps):
        visualize(ds_train_labeled, ds_train_unlabeled, model, epochs=i*vis_epochs,save_name='test')
        model.fit(ds_train_labeled, validation_data=ds_test, epochs=vis_epochs)
    visualize(ds_train_labeled, ds_train_unlabeled, model, epochs=vis_steps * vis_epochs, save_name='test')

def visualize(data_labeled, data_unlabeled, model, epochs, save_name):
    feature_extractor = model.layers[0]
    head = model.layers[1]
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    plot.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    plot_variance(plot, head)
    plot_data_distribution(plot, data_unlabeled, feature_extractor, small_points=True)
    plot_data_distribution(plot, data_labeled, feature_extractor)
    os.makedirs(save_name, exist_ok=True)
    fig.savefig('./'+save_name +'/' + str(epochs) + '.jpg', dpi=300)
    plt.close(fig)


def plot_data_distribution(plot: plt, data, feature_extractor, small_points=False):
    features = feature_extractor.predict(data)
    labels = np.concatenate([y for x, y in data], axis=0)
    if small_points:
        marker_data = ['kx', 'kx', 'kx']
        markersize = 3
        alpha = 0.5
    else:
        marker_data = ['gx', 'bx', 'rx']
        markersize = 3
        alpha = 0.7

    for class_id in range(3):
        x_class = features[labels == class_id]
        plot.plot(x_class[:, 1], x_class[:, 0], marker_data[class_id], ms=markersize, alpha=alpha)


def plot_variance(plot, head):
    # make these smaller to increase the resolution
    x = np.arange(0.0, 1.05, 0.05)
    x_1, x_2 = np.array(meshgrid(x,x))
    grid = np.stack((x_1, x_2))
    grid = grid.T.reshape(-1, 2)
    num_samples = 50
    outs = []
    for i in range(num_samples):
        outs.append(head.predict(grid))
    outs = np.array(outs)
    stds = np.mean(np.std(outs, axis=0), axis=-1)

    # vgp = head.layers[0]
    # outs = tf.nn.softmax(vgp(grid).sample(100))
    # stds = np.mean(np.std(outs, axis=0), axis=-1)

    std_plot = stds.reshape(x.shape[0], x.shape[0])
    plot.contourf(x, x, std_plot, cmap='Greys', vmin=0.0, vmax=0.5, alpha=0.7) #  norm=Normalize(),
    num_classes = 3
    thresholds = [0.9, 1.1]
    colours = ['green', 'blue', 'red']
    for class_id in range(num_classes):
        class_pred = np.mean(outs, axis=0)[:,class_id]
        class_pred = class_pred.reshape(x.shape[0], x.shape[0])
        plot.contourf(x, x, class_pred,  colors=[colours[class_id]], levels=thresholds ,alpha=0.2)

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

def show_inducing_locations(x, model, num_inducing_points, num_classes, num_samples_per_class,epochs,
                            save_name='figures', input_dim=2):
    # plot = plot.copy()
    weights = model.layers[0].get_weights()
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