import tensorflow as tf
import pickle
import numpy as np


def load_cifar10(data_path):
    train = []
    valid = []
    for i in range(1, 6):
        with open(f'{data_path}/cifar-10-batches-py/data_batch_{i}', 'rb') as f:
            imgs = pickle.load(f, encoding='bytes')
            imgs = np.transpose(imgs[b'data'].reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
            if i < 5:
                train.append(imgs)
            else:
                valid.append(imgs)
    train = np.concatenate(train)
    valid = np.concatenate(valid)
    return train, valid


def get_dataset(imgs, batch_size=16, mode='train'):
    def _map_fn(x):
        labels = tf.cast(x, tf.int32)
        imgs = tf.cast(x, tf.float32)
        imgs = imgs / 255.
        return imgs, labels
    ds = tf.data.Dataset.from_tensor_slices(imgs)
    if mode == 'train':
        ds = ds.shuffle(1024)
        ds = ds.repeat(-1)
    ds = ds.map(_map_fn)
    return ds.batch(batch_size)

