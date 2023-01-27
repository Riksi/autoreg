import tensorflow as tf
from masks import get_mask


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters_in, filters, act=None, mode='B'):
        super(MaskedConv2D, self).__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.filters = filters
        self.mode = mode
        if act is not None:
            self.act = tf.keras.layers.Activation(act)

        self.kernel = self.add_weight(name='kernel', shape=(*self.kernel_size, filters_in, self.filters))
        self.bias = self.add_weight(name='bias', shape=(self.filters,))
        mask_kwargs = dict(kernel_size=self.kernel_size,
                           features_in=filters_in,
                           features_out=self.filters,
                           mode=self.mode)
        self.mask = get_mask(**mask_kwargs)

    def call(self, x, **kwargs):
        kernel = self.kernel * self.mask
        out = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(out, self.bias)
        if hasattr(self, 'act'):
            return self.act(out)
        return out


class PixelCNNResBlock(tf.keras.layers.Layer):
    def __init__(self, filters_in, filters):
        super(PixelCNNResBlock, self).__init__()
        self.conv_in = MaskedConv2D(1, filters_in, filters, act='relu')
        self.conv_mid = MaskedConv2D(3, filters, filters, act='relu')
        self.conv_out = MaskedConv2D(1, filters, 2 * filters, act='relu')

    def call(self, x, **kwargs):
        out = self.conv_in(x)
        out = self.conv_mid(out)
        out = self.conv_out(out)
        return x + out