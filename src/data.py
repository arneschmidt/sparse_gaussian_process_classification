import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

class Data:
    def __init__(self, num_classes):
        self.x_full, self.y_full, self.x_test, self.y_test = self.load_mnist_data(num_classes)
        self.n_data_points = 60000
        self.labeled_indices = np.array([], dtype=int)
        self.unlabeled_indices = np.arange(self.n_data_points)

    def load_mnist_data(self, num_classes=3):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_full = x_train / 255
        y_full = np.clip(y_train, a_min=0, a_max=num_classes-1).astype(int)
        x_test = x_test / 255
        y_test = np.clip(y_test, a_min=0, a_max=num_classes-1).astype(int)
        return x_full, y_full, x_test, y_test

    def generate_train_data(self, batch_size=128, indices=None):
        if indices is None:
            indices = self.labeled_indices

        if len(indices) > 0:
            train_gen = tf.data.Dataset.from_tensor_slices((self.x_full[indices], self.y_full[indices]))
        else:
            raise Exception('No labeled data available.')
        train_gen = self._prepare_data(train_gen, batch_size)
        return train_gen

    def generate_test_data(self, batch_size=128):
        test_gen = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        test_gen = self._prepare_data(test_gen, batch_size)
        return test_gen

    def _prepare_data(self, ds, batch_size):
        ds = ds.cache()
        # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def randomly_add_labels(self, n):
        np.random.seed(42)
        ids = np.random.choice(self.unlabeled_indices.shape[0], n, replace=False)
        newly_labeled = self.unlabeled_indices[ids]
        self.labeled_indices = np.concatenate((self.labeled_indices, newly_labeled), axis=0)
        self.unlabeled_indices = np.delete(self.unlabeled_indices, ids)

    def add_labels(self, ids):
        # ids = np.where(np.isin(label_indices, self.unlabeled_indices))
        newly_labeled = self.unlabeled_indices[ids]
        self.labeled_indices = np.concatenate((self.labeled_indices, newly_labeled), axis=0)
        self.unlabeled_indices = np.delete(self.unlabeled_indices, ids)




