import datetime
import yaml
from easydict import EasyDict
import argparse

from inference import generate_images
from train import Trainer
from models import build_pixel_cnn
import numpy as np
from data import load_cifar10, get_dataset

import tensorflow as tf

import os


def main(config):
    if config.debug and config.debug_interactive:
        print('Debug mode is on. Continue? (y/n)')
        if input() != 'y':
            print("Exiting...")
            return

    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(cfg.ckpt_path, exist_ok=True)
    os.makedirs(cfg.log_path, exist_ok=True)

    with open(os.path.join(cfg.save_path, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f)

    print('Model name: {}'.format(model_name))
    print('Saving checkpoints to: {}'.format(cfg.ckpt_path))
    print('Saving logs to: {}'.format(cfg.log_path))
    
    model = build_pixel_cnn(
                     hidden_dim=config.hidden_dim,
                     out_dim=config.out_dim,
                     n_layers=config.n_layers)

    trainer = Trainer(
        config=config,
        model=model,
        loss_fn=tf.nn.sparse_softmax_cross_entropy_with_logits,
        optim=tf.keras.optimizers.Adam(learning_rate=config.lr)
    )
    train, val = load_cifar10(config.data_path)
    train_ds = get_dataset(train, config.batch_size)
    val_ds = get_dataset(val, config.batch_size, mode='test')
    if config.debug:
        # Train can be limited by max_iter 
        # so only need to limit val_ds
        val_ds = val_ds.take(config.debug_size)

    train_loss = []

    for itr, (imgs, labels) in enumerate(train_ds.take(config.max_iter), 1):
        loss, _ = trainer.train_step(imgs, labels)
        train_loss.append(loss.numpy())

        if (itr % config.train_loss_window) == 0:
            print('Iteration {}, Loss: {}'.format(itr, np.mean(train_loss[-100:])))

        if (itr == config.img_gen_init) or (itr % config.img_gen_window) == 0:
            if config.debug:
                print('Generating images...')
            trainer.create_image_samples()

        if (itr % config.valid_window) == 0:
            trainer.save(os.path.join(config.ckpt_path, f'weights_{itr}'))
            val_loss = trainer.evaluate(val_ds)
            print('Validation loss: {}'.format(val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    model_name = 'pixel_cnn_{}'.format(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    cfg.save_path = os.path.join(cfg.save_dir, model_name)
    cfg.ckpt_path = os.path.join(cfg.save_path, 'checkpoints')
    cfg.log_path = os.path.join(cfg.save_path, 'logs')

    main(cfg)


