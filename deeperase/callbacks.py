import io
from typing import Mapping

import matplotlib.pyplot as plt

import tensorflow as tf


################################################################################

class PlotPredictions(tf.keras.callbacks.Callback):
    """
    After every epoch, plots the model predictions using matplotlib.
    Be aware that this callback is blocking, meaning that 
    training won't continue if you do not close the graphics window.

    Parameters
    ----------
    dataset: tf.data.Dataset
        Dataset that returns pairs of images of the same size (input and target)
    num_img: int, default 4
        Number of images to plot after every epoch coming from the given 
        images_dataset.
    """
    def __init__(self, dataset: tf.data.Dataset, num_img: int = 4):
        self.num_img = num_img
        self.dataset_iter = iter(dataset.repeat())

    def on_epoch_end(self, epoch, logs=None):
        _compose_prediction_image(self.model, 
                                  self.dataset_iter, 
                                  self.num_img, epoch)
        plt.show()


class TensorBoard(tf.keras.callbacks.Callback):

    """
    Simplified version of the tf.keras.callbacks.Tensorboard that only:
        - Plots losses after each training batch
        - Writes the predictions to Tensorboard with a nice looking image

    Parameters
    ----------
    images_dataset: tf.data.Dataset
        Dataset that returns pairs of images of the same size (input and target)
    logdir: str, default 'logs'
        Tensorboard logs directory
    num_img: int, default 4
        Number of images to plot after every epoch coming from the given 
        images_dataset.
    """
    def __init__(self, 
                 images_dataset: tf.data.Dataset,
                 logdir: str = 'logs',
                 num_img: int = 4):

        self.writer = tf.summary.create_file_writer(str(logdir),
                                                    flush_millis=5000)
        self.step = 0

        self.dataset_iter = iter(images_dataset.repeat())
        self.num_img = num_img

    def on_train_batch_end(self, 
                           batch: tf.Tensor, 
                           logs: Mapping[str, float] = None):

        with self.writer.as_default():
            for k, v in logs.items():
                tf.summary.scalar(f'train/{k}', v, step=self.step)

        self.step += 1
        self.writer.flush()

    def on_epoch_end(self, epoch: int, logs: Mapping[str, float] = None):
        _compose_prediction_image(self.model, 
                                  self.dataset_iter, 
                                  self.num_img, epoch)

        with self.writer.as_default():
            tf.summary.image("Predictions", _plot_to_image(), step=epoch)


################################################################################

def _compose_prediction_image(model: tf.keras.Model,
                              dataset: tf.data.Dataset,
                              n_samples: int,
                              epoch: int):

    _, ax = plt.subplots(self.num_img, 2, figsize=(12, 12))

    plt.suptitle(f'Epoch {epoch}')
    samples = [next(dataset) for _ in range(n_samples)]

    for i, (img, target) in enumerate(samples):
        prediction = model(tf.expand_dims(img, 0), training=False)[0]
        prediction = (prediction * 255).numpy().astype('uint8')
        target = (target * 255).numpy().astype('uint8')

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Prediction")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")


def _plot_to_image():
    # Converts the current matplotlib context to a tf.Tensor containing the
    # figure as an RGBA image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
