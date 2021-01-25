import click
from typing import Tuple

import matplotlib.pyplot as plt

import tensorflow as tf

import deeperase


def _show_batch(input_images: tf.Tensor, target_images: tf.Tensor):
    cols = 2
    rows = input_images.shape[0]
    i = 1

    for inp, target in zip(input_images, target_images):
        plt.subplot(rows, cols, i)
        plt.title('Input Image (Noise)')
        plt.imshow(inp)
        plt.axis('off')

        plt.subplot(rows, cols, i + 1)
        plt.title('Target (Clean)')
        plt.imshow(target)
        plt.axis('off')

        i += 2

    plt.show()


def _build_model(input_shape: Tuple[int, int, int], 
                 is_gan: bool) -> tf.keras.Model:

    unet = deeperase.build_unet(input_shape, 
                                backbone='mobilenetv2', 
                                n_channels=3, 
                                activation='sigmoid',
                                name='unet_generator')
    unet_optimizer = tf.optimizers.Adam(learning_rate=2e-4, beta_1=.5)

    if is_gan:
        discriminator = deeperase.build_discriminator(input_shape, 
                                                      name='discriminator')
        discriminator_optimizer = tf.optimizers.Adam(learning_rate=2e-4, 
                                                     beta_1=.5)
    else:
        discriminator = None
        discriminator_optimizer = None

    model = deeperase.DeepErase(generator=unet, discriminator=discriminator)
    model.build((None,) + input_shape)
    model.compile(optimizer=unet_optimizer, 
                  discriminator_optimizer=discriminator_optimizer)
    return model


@click.command()
@click.option('-i', '--input-pattern', required=True)
@click.option('-t', '--target-pattern', required=True)
@click.option('--gan/--no-gan', default=False)

@click.option('--logdir', type=click.Path(file_okay=False), default='logs')

@click.option('--epochs', default=50, type=int)
@click.option('--batch-size', default=64, type=int)
def train(input_pattern: str, target_pattern: str, 
          logdir: str,
          gan: bool, 
          epochs: int,
          batch_size: int):

    image_size = (64, 256)
    train_ds, test_ds = deeperase.data.build_dataset(input_pattern,
                                                     target_pattern,
                                                     image_size=image_size,
                                                     do_crop=True,
                                                     do_hflip=True)

    _show_batch(*next(iter(train_ds.batch(4))))

    model = _build_model(image_size + (3,), gan)
    model.summary()

    model.fit(train_ds.batch(batch_size).take(2), 
              epochs=epochs, 
              callbacks=[
                  deeperase.callbacks.PlotPredictions(test_ds, 4),
                  deeperase.callbacks.TensorBoard(logdir=logdir,
                                                  images_dataset=test_ds),
                  tf.keras.callbacks.ModelCheckpoint(
                        filepath='./models/deeperase')])


if __name__ == '__main__':
    train()
