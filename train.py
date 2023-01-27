import os
import tensorflow as tf
from inference import generate_images
from typing import Callable, Iterable, Union


class Trainer(object):
    def __init__(self,
                config: dict,   
                 model: tf.keras.models.Model,
                 loss_fn: Union[tf.keras.losses.Loss, Callable],
                 optim: tf.keras.optimizers.Optimizer):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.ckpt = tf.train.Checkpoint(
            transformer=self.model,
            optimizer=self.optim
        )
        self.trn_writer = tf.summary.create_file_writer(os.path.join(config.log_path, 'train'))
        self.test_writer = tf.summary.create_file_writer(os.path.join(config.log_path, 'test'))
        self.img_writer = tf.summary.create_file_writer(os.path.join(config.log_path, 'imgs'))


    def run_model(self, images, labels):
        predictions = self.model(images)
        batch_size = tf.shape(images)[0]
        # Model can be evaluated on all except for the very first element
        # i.e. the R value of the top left corner pixel
        y_true = tf.reshape(labels, [batch_size, -1])[:, 1:]
        y_pred = tf.reshape(predictions, [batch_size, -1, 256])[:, 1:]
        loss = tf.reduce_mean(self.loss_fn(labels=y_true, logits=y_pred))

        return loss, predictions

    @tf.function
    def create_image_samples(self):
        itr = self.optim.iterations
        gen_imgs = generate_images(
            model=self.model,
            img_dims=self.config.img_dims,
        )

        with self.img_writer.as_default():
            tf.summary.image('gen_imgs', tf.cast(gen_imgs * 255, tf.uint8), itr)


    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            loss, predictions = self.run_model(images, labels)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        itr = self.optim.iterations

        bits_per_dim = loss / tf.math.log(2.)
        with self.trn_writer.as_default():
            tf.summary.scalar('loss', bits_per_dim, itr)
            tf.summary.scalar('lr', self.optim.learning_rate, itr)

        self.optim.learning_rate.assign(
            self.optim.learning_rate * 0.999995
        )

        return bits_per_dim, predictions

    def valid_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        loss, predictions = self.run_model(images, labels)

        return loss, predictions

    @tf.function
    def evaluate(self, dataset: tf.data.Dataset):
        loss = tf.TensorArray(tf.float32, size=tf.cast(len(dataset), tf.int32))
        for idx, (images, labels) in enumerate(dataset):
            loss_, _ = self.valid_step(images, labels)
            idx = tf.cast(idx, tf.int32)
            loss = loss.write(idx, loss_)
        loss = loss.stack()
        bits_per_dim = tf.reduce_mean(loss / tf.math.log(2.))
        with self.test_writer.as_default():
            tf.summary.scalar(
                'bits_per_dim',
                bits_per_dim,
                self.optim.iterations
            )
        return bits_per_dim

    def save(self, path):
        self.ckpt.save(path)

    def load(self, path):
        self.ckpt.restore(path)
