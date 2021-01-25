
from typing import Optional, Tuple, Sequence, Union

import tensorflow as tf

from deeperase import backbones

################################################################################

_SUPPORTED_MODELS = '\n'.join(f'- {o}' 
                              for o in backbones.list_supported_models())


################################################################################

class _Conv3x3BnReLU(tf.keras.layers.Layer):

    def __init__(self, 
                 filters: int, 
                 use_batchnorm: bool = True,
                 n_layers: int = 1,
                 name: Optional[str] = None, 
                 **kwargs) -> None:

        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.n_layers = n_layers

        self.conv = [tf.keras.layers.Conv2D(filters, 
                                            kernel_size=3, 
                                            padding='same',
                                            name=f'{name}_{i + 1}')
                     for i in range(self.n_layers)]

        self.relu = tf.keras.layers.ReLU(name=f'{name}_relu')

        if self.use_batchnorm:
            self.bn = [tf.keras.layers.BatchNormalization(
                            name=f'{name}_bn_{i + 1}') 
                       for i in range(self.n_layers)]

        super(_Conv3x3BnReLU, self).__init__(name=name, **kwargs)

    def call(self, 
             x: tf.Tensor, 
             training: Optional[bool] = None) -> tf.Tensor:

        for i in range(self.n_layers):
            x = self.conv[i](x)

            if self.use_batchnorm:
                x = self.bn[i](x)

            x = self.relu(x)

        return x


class _DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, 
                 filters: int, 
                 use_batchnorm: bool = True, 
                 upsample_strategy: str = 'upsample',
                 name: Optional[str] = None, 
                 **kwargs) -> None:

        super(_DecoderBlock, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.upsample_strategy = upsample_strategy

        if self.upsample_strategy == 'conv':
            self.upsample = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                name=f'{name}_upsample_conv')

            self.conv = _Conv3x3BnReLU(filters, 
                                       self.use_batchnorm,
                                       name=f'{name}_conv')

        elif self.upsample_strategy == 'upsample':
            self.upsample = tf.keras.layers.UpSampling2D(
                name=f'{name}_upsample')

            self.conv = _Conv3x3BnReLU(filters, 
                                       self.use_batchnorm, 
                                       n_layers=2,
                                       name=f'{name}_conv')
            # Set to False to avoid using bn layer on `call` method
            self.use_batchnorm = False

        else:
            raise ValueError('`upsample strategy` must be either "upsample" '
                             'or "conv"')

        if self.use_batchnorm:
            self.bn = tf.keras.layers.BatchNormalization(name=f'{name}_bn')
        else:
            self.bn = tf.keras.layers.Layer() # Identity layer

        self.relu = tf.keras.layers.ReLU(name=f'{name}_relu')

    def call(self, 
             x: Tuple[tf.Tensor, tf.Tensor], 
             training: Optional[bool] = None) -> tf.Tensor:

        x, residual = x

        x = self.upsample(x)
        x = self.bn(x)
        x = self.relu(x)

        if residual is not None:
            x = tf.concat([x, residual], axis=-1)

        return self.conv(x)


################################################################################

def build_unet(
        input_shape: Union[Tuple[int, int, int], tf.TensorShape],
        backbone: str,
        n_channels: int,
        activation: str = 'sigmoid',
        decoder_filters: Sequence[int] = (256, 128, 64, 32, 16),
        use_batchnorm: bool = True, 
        upsample_strategy: str = 'upsample',
        freeze_backbone: bool = False,
        weights: Optional[str] = None,
        name: Optional[str] = None) -> tf.keras.Model:
    f"""
    Builds a modified version of the UNet described on 
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597)
    Parameters
    ----------
    input_shape : tuple or tf.TensorShape
        Shape of the input tensor without the batch dimension.
        Image format should be: Height, Width, # Channels
    backbone: str
        Neural Network architecture to use as a backbone. As now we support:
        {_SUPPORTED_MODELS}
    n_channels: int
        Channels of the las convolution.
    activation: str
        Activation of the last convolution
    decoder_filters: Sequence[int], default (256, 128, 64, 32, 16)
        Decoder is composed of five layers. This is the convolutional filters
        used in each decoder layer.
    use_batchnorm: bool, default True
        Whether to use Batch Normalization after convolutions or not.
    upsample_strategy: default 'upsample'
        Method to upsample feature maps in the decoder. It can either be:
         - "upsample": Uses nearest interpolation to upsample feature maps.
         - "conv": Uses Transposed convs to upsample the feature maps.
    freeze_backbone : bool
        Whether or not backbone layers should be trainable or not (frozen).
    """
    if len(decoder_filters) != 5:
            raise ValueError("Decoder has 5 layers."
                             f"You specified {len(decoder_filters)} filters")

    input_tensor = tf.keras.Input(input_shape)

    backbone = backbones.build_unet_backbone(name=backbone,
                                             input_tensor=input_tensor,
                                             n_levels=5,
                                             weights=weights)

    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False

    decoders = [_DecoderBlock(filters=f, 
                              use_batchnorm=use_batchnorm,
                              upsample_strategy=upsample_strategy,
                              name=f'decoder_{i}')
                for i, f in enumerate(decoder_filters, start=1)]

    out = _unet_forward(input_tensor, 
                        backbone=backbone, 
                        decoders=decoders, 
                        n_channels=n_channels, 
                        activation=activation)

    return tf.keras.Model(input_tensor, out, name=name)


def _unet_forward(x: tf.Tensor,
                  backbone: tf.keras.Model, 
                  decoders: Sequence[_DecoderBlock],
                  n_channels: int,
                  activation: str) -> tf.Tensor:

    residuals = backbone.output[::-1]

    x = residuals[0]
    for i in range(5):
        if i < len(residuals) - 1:
            residual = residuals[i + 1]
        else:
            residual = None
        x = decoders[i]([x, residual])

    return tf.keras.layers.Conv2D(n_channels,
                                  kernel_size=3,
                                  activation=activation,
                                  padding='same',
                                  name='unet_output')(x)
