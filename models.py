import tensorflow as tf
import blocks
from blocks import MaskedConv2D


def build_pixel_cnn(hidden_dim, out_dim, n_layers):
    return PixelRNN(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers=n_layers,
        pixel_layer=blocks.PixelCNNResBlock
    )


class PixelRNN(tf.keras.models.Model):
    def __init__(self, hidden_dim, out_dim, n_layers, pixel_layer):
        super(PixelRNN, self).__init__()
        hidden_dim = hidden_dim * 3
        out_dim = out_dim * 3

        self.input_conv = MaskedConv2D(kernel_size=7,
                                       filters_in=3,
                                       filters=2 * hidden_dim,
                                       mode='A')

        self.pixel_model = tf.keras.Sequential(
            [pixel_layer(2 * hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.output_conv1 = MaskedConv2D(kernel_size=1,
                                         filters_in=2 * hidden_dim,
                                         filters=out_dim)
        self.output_conv2 = MaskedConv2D(kernel_size=1,
                                         filters_in=out_dim,
                                         filters=out_dim)

        self.final_conv = MaskedConv2D(kernel_size=1,
                                       filters_in=out_dim,
                                       filters=256 * 3)

    def call(self, x, **kwargs):
        y = self.input_conv(x)
        y = self.pixel_model(y)

        y = self.output_conv1(tf.nn.relu(y))
        y = self.output_conv2(tf.nn.relu(y))
        y = self.final_conv(y)

        y = tf.reshape(y, tf.concat([tf.shape(y)[:-1], [3, 256]], 0))
        return y