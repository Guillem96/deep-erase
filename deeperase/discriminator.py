from typing import Tuple, Union, Optional

import tensorflow as tf


_KERNEL_INITIALIZER = tf.initializers.RandomNormal(0, .02)


def _conv_bn(x: tf.Tensor, 
             filters: int, 
             activation: tf.keras.layers.Activation, 
             kernel_size: Tuple[int, int] = (3, 3), 
             strides: Tuple[int, int] = (2, 2),
             padding: str = "same",
             kernel_initializer=_KERNEL_INITIALIZER):

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                kernel_initializer=kernel_initializer,
                                padding=padding, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if activation:
        x = activation(x)

    return x


def build_discriminator(
      input_shape: Union[Tuple[int, int, int], tf.TensorShape],
      filters: int = 64, 
      n_hidden: int = 3,
      kernel_initializer=_KERNEL_INITIALIZER,
      name: Optional[str] = None) -> tf.keras.Model:
    """
    Builds a PatchGAN like Discriminator.

    Parameters
    ----------
    input_shape: Union[Tuple[int, int, int], tf.TensorShape]
        Shape of the input tensors (input image and target image) without the 
        batch dimension. Image format should be: Height, Width, # Channels
    filters: int, default 64
        Number of filters for the first convolutional layer. Then, for each 
        layer the filters increase in a exponential manner following this
        equation, $f_l = 2 f_{l - 1}$, where $f_0$ is the given parameter and 
        $l$ is the layer.
    n_hidden: int, default 3
        Number of hidden layers. For example, if n_hidden=3 and filters=64 
        the features of each convolutinal layers will be [64, 128, 256, 512, 1]
        being the last value the channels of the last layer.
    kernel_initializer: tf.initializers.Initializer, 
            default tf.initializers.RandomNormal(0, .02)
    
    Returns
    -------
    tf.keras.Model
        Model implementing the discriminator with a sigmoid in the last 
        activation
    """
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
    tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

    x = tf.keras.layers.concatenate([inp, tar], name='concatenate_inputs')

    x = tf.keras.layers.Conv2D(filters, (4, 4), strides=(2, 2),
                                padding="same",
                                kernel_initializer=kernel_initializer)(x)

    x = tf.keras.layers.LeakyReLU(0.2)(x)

    for l in range(n_hidden):
        filters *= 2
        if l < 2:
            x = _conv_bn(x, 
                         filters=filters,
                         activation=tf.keras.layers.LeakyReLU(0.2),
                         kernel_size=(4, 4), strides=(2, 2),
                         kernel_initializer=kernel_initializer)
        else:
            x = _conv_bn(x, 
                         filters=filters,
                         activation=tf.keras.layers.LeakyReLU(0.2),
                         kernel_size=(4, 4), strides=(1, 1),
                         kernel_initializer=kernel_initializer)

    x = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), 
                               padding="same", 
                               activation='sigmoid')(x)

    return tf.keras.Model(inputs=[inp, tar], outputs=x, name=name)
