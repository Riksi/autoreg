import numpy as np
import tensorflow as tf
from typing import Iterable


def generate_images(
                model: tf.keras.models.Model,
                img_dims: Iterable,
                n_images: int=1,
                n_channels: int=3,
                initial_values=None,   
                ) -> tf.Tensor:
    """
    Samples from an autoregressive model to generate images.
    Parameters
    ----------
    model: tf.keras.models.Model
        The model to sample from.
    img_dims: Iterable
        The dimensions of the image to generate.
        Must be of length 2.
    n_images: int
        The number of images to generate.
    n_channels: int
        The number of channels in the image.
    initial_values: tf.Tensor
        The initial values to use for top left pixel of the first channel.
        Must be of shape (n_images,).
        If None, values are sampled uniformly from [0, 1].
    
    Returns
    -------
    img: tf.Tensor
        The generated samples.
        Has shape (n_images, *img_dims, n_channels).
    """
    batch_inds = tf.expand_dims(tf.range(n_images), axis=-1)
    # [0, 0, 0, 0, ...], [1, 0, 0, 0, ...], [2, 0, 0, 0, ...], ...
    init_inds = tf.concat(
        [
            batch_inds,
            tf.zeros((n_images, n_channels), dtype='int32')
        ], axis=-1
    )
    if initial_values is None:
        initial_values = tf.random.uniform(shape=(n_images,),
                                                minval=0,
                                                maxval=1,
                                                dtype='float32')
    img = tf.scatter_nd(indices=init_inds,
                        updates=initial_values,
                        shape=(n_images, *img_dims, n_channels))
    # Letting step = 1 lets us omit the condition
    # if (clr + colm + row) > 0;
    # which is used to avoid updating the top left pixel of the first channel
    step = tf.constant(1)
    total_steps = tf.math.reduce_prod(img_dims) * n_channels
    def cond(step, img):
        return tf.less(step, total_steps)
    def body(step, img):
        row = tf.math.floordiv(step, img.shape[2] * n_channels)
        colm = tf.math.floordiv(tf.math.mod(step, img.shape[2] * n_channels), n_channels)
        clr = tf.math.mod(step, n_channels)
        update_inds = tf.concat(
            [
                batch_inds,
                tf.tile([[row, colm, clr]], [n_images, 1])
            ], axis=-1
        )
        # The prediction only depends on the pixels to the left and above
        # so we can be get a bit of speedup by using only the rows
        # upto and including the current row
        result = model(img[:, :row + 1, :, :])
        result_rgb = tf.random.categorical(
            logits=result[:, row, colm, clr], 
            num_samples=1
        )
        result_rgb = tf.cast(tf.squeeze(result_rgb, axis=-1), tf.float32)
        img = tf.tensor_scatter_nd_update(img,
                                        indices=update_inds,
                                        updates=result_rgb / 255.)
        return step + 1, img
    step, img = tf.while_loop(cond, body, [step, img])
    return img
