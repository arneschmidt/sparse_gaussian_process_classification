import os
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tensorflow.keras.utils import to_categorical
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from model_architecture import create_model, compile_model
from data import Data


def main():
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    with open('config.yaml') as file:
        config = yaml.full_load(file)

    data = Data(config['data']['num_classes'])
    ds_test = data.generate_test_data()
    data.randomly_add_labels(config['data']['start_samples'])

    model = create_model(config=config, num_classes=config['data']['num_classes'], num_training_points=100)
    compile_model(config, model)

    vis_epochs = config['visualization']['epochs']
    vis_steps = int(config['model']['epochs'] / vis_epochs)
    for acquisition_step in range(config['data']['acquisition_steps']):
        print('Acquisition step ' + str(acquisition_step) + ' | number of labels: ' + str(len(data.labeled_indices)))
        ds_train_labeled = data.generate_train_data()
        ds_train_unlabeled = data.generate_train_data(indices=data.unlabeled_indices)
        model.variables[13].assign(len(data.labeled_indices))
        for i in range(vis_steps):
            model.fit(ds_train_labeled, validation_data=ds_test, epochs=vis_epochs)
            total_epoch = (acquisition_step*config['model']['epochs']) + (i+1)*vis_epochs
            print('Total epochs finished: ' + str(total_epoch))
            visualize(ds_train_labeled, ds_train_unlabeled, model, config['data']['num_classes'], epochs=total_epoch)

        ids = select_data(ds_train_unlabeled, model, config['data']['acquisition_samples'])
        ds_train_newly_labeled = data.generate_train_data(indices=data.unlabeled_indices[ids])
        visualize(ds_train_labeled, ds_train_unlabeled, model, config['data']['num_classes'],
                  epochs=((acquisition_step+1)*config['model']['epochs']),
                  save_name='acquisition',
                  data_newly_labeled=ds_train_newly_labeled)
        data.add_labels(ids)

def select_data(ds_train_unlabeled, model, number_samples):
    feature_extractor = model.layers[0]
    cnn_out = feature_extractor.predict(ds_train_unlabeled)
    vgp = model.layers[1].layers[0]
    stds = np.array([])
    batch_size = 128
    steps = int(np.ceil(len(cnn_out)/128))
    for step in range(steps):
        start = step*batch_size
        stop = (step+1)*batch_size
        if stop > len(cnn_out):
            stop = len(cnn_out)
        pred = tf.nn.softmax(vgp(cnn_out[start:stop]).sample(10))
        std = np.mean(np.std(pred, axis=0), axis=-1)
        stds = np.concatenate((stds, std))
    # stds = np.mean(np.std(outs, axis=0), axis=-1)
    # outs = []
    # for i in range(20):
    #     outs.append(vgp(cnn_out))
    stds = np.array(stds)
    ids = stds.argsort()[::-1][:number_samples]
    return ids

def visualize(data_labeled, data_unlabeled, model, num_classes, epochs, out_dir='test', save_name='', data_newly_labeled=None):
    feature_extractor = model.layers[0]
    head = model.layers[1]
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    plot.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    plot_variance(plot, head, num_classes)
    plot_data_distribution(plot, data_unlabeled, feature_extractor, num_classes, marker='small')
    plot_data_distribution(plot, data_labeled, feature_extractor, num_classes)
    if data_newly_labeled is not None:
        plot_data_distribution(plot, data_newly_labeled, feature_extractor, num_classes, marker='star')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig('./'+out_dir +'/' + str(epochs) + save_name + '.jpg', dpi=300)
    plt.close(fig)


def plot_data_distribution(plot: plt, data, feature_extractor, num_classes = 3, marker='normal'):
    features = feature_extractor.predict(data)
    labels = np.concatenate([y for x, y in data], axis=0)
    if marker=='small':
        marker_data = 'x'
        marker_colour = ['k','k','k','k','k','k','k','k','k','k']
        markersize = 3
        alpha = 0.3
    elif marker=='star':
        marker_data = '*'
        marker_colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        markersize = 7
        alpha = 1.0
    else:
        marker_data = 'x'
        marker_colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        markersize = 3
        alpha = 0.7

    for class_id in range(num_classes):
        x_class = features[labels == class_id]
        plot.plot(x_class[:, 1], x_class[:, 0], marker_data, color=marker_colour[class_id], ms=markersize, alpha=alpha)


def plot_variance(plot, head, num_classes):
    # make these smaller to increase the resolution
    x = np.arange(0.0, 1.05, 0.05)
    x_1, x_2 = np.array(meshgrid(x,x))
    grid = np.stack((x_1, x_2))
    grid = grid.T.reshape(-1, 2)
    num_samples = 50
    # outs = []
    # for i in range(num_samples):
    #     outs.append(head.predict(grid))
    # outs = np.array(outs)
    # stds = np.mean(np.std(outs, axis=0), axis=-1)

    vgp = head.layers[0]
    outs = tf.nn.softmax(vgp(grid).sample(50))
    stds = np.mean(np.std(outs, axis=0), axis=-1)

    std_plot = stds.reshape(x.shape[0], x.shape[0])
    plot.contourf(x, x, std_plot, cmap='Greys', vmin=0.0, vmax=0.3, alpha=0.7) #  norm=Normalize(),
    thresholds = [0.9, 1.1]
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
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

# def show_inducing_locations(x, model, num_inducing_points, num_classes, num_samples_per_class,epochs,
#                             save_name='figures', input_dim=2):
#     # plot = plot.copy()
#     weights = model.layers[0].get_weights()
#     inducing_locations = weights[2].reshape(num_inducing_points*num_classes, input_dim)
#     inducing_observations = weights[3].reshape(num_inducing_points*num_classes)
#     # plot.plot(inducing_locations[...,0], inducing_locations[...,1], 'bs')
#
#     marker_data = ['gx', 'rx', 'bx']
#     marker_inducing_points= ['g', 'r', 'b']
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     for i in range(num_classes):
#         x_class = x[i*num_samples_per_class:(i+1)*num_samples_per_class]
#         ax.plot(x_class[:, 0], x_class[:, 1], marker_data[i])
#         plot_inducing_points(ax, i, inducing_locations, inducing_observations, num_inducing_points, marker_inducing_points)
#     plot_variance(ax, model)
#     ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
#
#     os.makedirs(save_name, exist_ok=True)
#     plt.savefig('./'+save_name +'/' + str(epochs) + '.jpg', dpi=300)
#     plt.close(fig)
#
# def plot_inducing_points(plot, class_id, inducing_locations, inducing_observations, num_inducing_points, marker_inducing_points):
#     inducing_locations_class = inducing_locations[class_id * num_inducing_points:(class_id + 1) * num_inducing_points]
#     inducing_observations_class = inducing_observations[class_id * num_inducing_points:(class_id + 1) * num_inducing_points]
#
#     for point in range(num_inducing_points):
#         indication = inducing_observations_class[point]
#         if indication > 0:
#             marker = 'P'
#         else:
#             marker = 'D'
#         markersize = min(abs(indication), 1) * 10
#         plot.plot(inducing_locations_class[point, 0], inducing_locations_class[point, 1], color=marker_inducing_points[class_id],
#                   markersize=markersize, marker=marker)
#
# def generate_data(samples_per_class):
#     number_of_classes = 3
#     class_means = [[-1, -1],
#                    [0, 1],
#                    [1, 0]]
#     class_means_2 = [[1, 1],
#                    [1, -1],
#                    [-1.5, 1.5]]
#     class_covariances = [[[0.1, 0.06], [0.06, 0.2]],
#                          [[0.2, 0.06], [0.06, 0.1]],
#                          [[0.1, 0], [0, 0.1]]]
#     plot_colours = ['gx', 'rx', 'bx']
#     x = []
#     y = []
#     np.random.seed(42)
#     plt.figure()
#     for i in range(number_of_classes):
#         x_samples = np.random.multivariate_normal(class_means[i], class_covariances[i], int(samples_per_class/2))
#         x_samples_2 = np.random.multivariate_normal(class_means_2[i], class_covariances[i], int(samples_per_class/2))
#         y_targets = np.full(shape=(samples_per_class), fill_value=i)
#         x.append(x_samples)
#         x.append(x_samples_2)
#         y.append(to_categorical(y_targets, num_classes=number_of_classes))
#         plt.plot(x_samples[...,0], x_samples[...,1], plot_colours[i])
#         plt.plot(x_samples_2[...,0], x_samples_2[...,1], plot_colours[i])
#     # plt.show()
#
#     x = np.array(x).reshape(number_of_classes*samples_per_class, 2)
#     y = np.array(y).reshape(number_of_classes*samples_per_class, number_of_classes)
#
#     return x, y