"""This module contains different utilities to build the backbones used by
other models. These backbones may come with already pre-trained weights, which
can save a significant amount of training time.
"""

from typing import Optional, Sequence

import tensorflow as tf


################################################################################

_MODELS = {

    # ResNets
    'resnet50': [tf.keras.applications.ResNet50,
                 tf.keras.applications.resnet.preprocess_input],
    'resnet101': [tf.keras.applications.ResNet101,
                  tf.keras.applications.resnet.preprocess_input],
    'resnet152': [tf.keras.applications.ResNet152,
                  tf.keras.applications.resnet.preprocess_input],

    'resnet50v2': [tf.keras.applications.ResNet50V2,
                   tf.keras.applications.resnet_v2.preprocess_input],
    'resnet101v2': [tf.keras.applications.ResNet101V2,
                    tf.keras.applications.resnet_v2.preprocess_input],
    'resnet152v2': [tf.keras.applications.ResNet152V2,
                    tf.keras.applications.resnet_v2.preprocess_input],

    # VGG
    'vgg16': [tf.keras.applications.VGG16,
              tf.keras.applications.vgg16.preprocess_input],
    'vgg19': [tf.keras.applications.VGG19,
              tf.keras.applications.vgg19.preprocess_input],

    # Densnet
    'densenet121': [tf.keras.applications.DenseNet121,
                    tf.keras.applications.densenet.preprocess_input],
    'densenet169': [tf.keras.applications.DenseNet169,
                    tf.keras.applications.densenet.preprocess_input],
    'densenet201': [tf.keras.applications.DenseNet201,
                    tf.keras.applications.densenet.preprocess_input],

    # Inception
    'inceptionresnetv2': [
        tf.keras.applications.InceptionResNetV2,
        tf.keras.applications.inception_resnet_v2.preprocess_input],
    'inceptionv3': [tf.keras.applications.InceptionV3,
                    tf.keras.applications.inception_v3.preprocess_input],
    'xception': [tf.keras.applications.xception.Xception,
                 tf.keras.applications.xception.preprocess_input],

    # Nasnet
    'nasnetlarge': [tf.keras.applications.NASNetLarge,
                    tf.keras.applications.nasnet.preprocess_input],
    'nasnetmobile': [tf.keras.applications.NASNetMobile,
                     tf.keras.applications.nasnet.preprocess_input],

    # MobileNet
    'mobilenet': [tf.keras.applications.MobileNet,
                  tf.keras.applications.mobilenet.preprocess_input],
    'mobilenetv2': [tf.keras.applications.MobileNetV2,
                    tf.keras.applications.mobilenet_v2.preprocess_input],

    # EfficientNets
    'efficientnetb0': [tf.keras.applications.EfficientNetB0,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb1': [tf.keras.applications.EfficientNetB1,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb2': [tf.keras.applications.EfficientNetB2,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb3': [tf.keras.applications.EfficientNetB3,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb4': [tf.keras.applications.EfficientNetB4,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb5': [tf.keras.applications.EfficientNetB5,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb6': [tf.keras.applications.EfficientNetB6,
                       tf.keras.applications.efficientnet.preprocess_input],
    'efficientnetb7': [tf.keras.applications.EfficientNetB7,
                       tf.keras.applications.efficientnet.preprocess_input],
}

_DEFAULT_FEATURE_LAYERS = {

    # VGG
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3',
              'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4',
              'block2_conv2', 'block1_conv2'),

    # ResNets
    'resnet50': ('conv5_block3_out', 'conv4_block6_out',
                 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet101': ('conv5_block3_out', 'conv4_block23_out',
                  'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet152': ('conv5_block3_out', 'conv4_block36_out',
                  'conv3_block8_out', 'conv2_block3_out', 'conv1_relu'),

    # TODO: Resnets v2

    # DenseNet
    'densenet121': ('relu', 'pool4_conv',
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),
    'densenet169': ('relu', 'pool4_conv',
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),
    'densenet201': ('relu', 'pool4_conv',
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),

    # Mobile Nets
    'mobilenet': ('conv_pw_13_relu', 'conv_pw_11_relu', 'conv_pw_5_relu',
                  'conv_pw_3_relu', 'conv_pw_1_relu'),
    'mobilenetv2': ('out_relu', 'block_13_expand_relu',
                    'block_6_expand_relu', 'block_3_expand_relu',
                    'block_1_expand_relu'),

    # EfficientNets
    'efficientnetb0': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb1': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb2': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb3': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb4': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb5': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb6': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
    'efficientnetb7': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation',
                       'block2a_expand_activation'),
}


def list_supported_models() -> Sequence[str]:
    """List supported backbone models."""
    return list(_MODELS)


def build_unet_backbone(name: str,
                        input_tensor: tf.Tensor,
                        n_levels: int = 4,
                        weights: Optional[str] = None) -> tf.keras.Model:
    """Builds a UNet backbone.
    Parameters
    ----------
    name : str
        Neural Network architecture to use as a backbone. To list the
        supported models call `list_supported_models()`.
    input_tensor : tf.Tensor
        Backbone input tensor.
    n_levels : int
        Number of layers to use when building the FPN network. As now all
        models support up to 5 levels, specifying a larger number won't have
        any effect.
    weights: Optional[str]
        It can either be 'imagenet' or None. If set to None the backbone is 
        randomly initialized, otherwise the backbone is initialized with the
        pretrained imagenet weights.

    Returns
    -------
    backbone : tf.keras.Model
        A model that combines different levels of the base neural network into
        a UNet architecture.
    """
    if name not in _MODELS:
        supported_models = list_supported_models()
        supported_models = '\n'.join(f'- {o}' for o in supported_models)
        raise ValueError(f"Backbone {name} is not supported. "
                         f"Supported backbones are: \n {supported_models}")

    model_cls, _ = _MODELS[name]
    model = model_cls(input_tensor=input_tensor,
                      include_top=False,
                      weights=weights)

    outputs = [model.get_layer(o).output
               for o in _DEFAULT_FEATURE_LAYERS[name][:n_levels]]

    return tf.keras.Model(inputs=model.inputs,
                          outputs=outputs[::-1],
                          name=name)
