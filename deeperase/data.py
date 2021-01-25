import functools
from typing import Union, Sequence, Tuple

import tensorflow as tf


################################################################################

@tf.function
def read_images(input_im: tf.Tensor,
                target_im: tf.Tensor,
                image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:

    input_im = tf.io.read_file(input_im)
    input_im = tf.image.decode_jpeg(input_im, channels=3)
    input_im = tf.image.resize(input_im, image_size)
    input_im = tf.image.convert_image_dtype(input_im, tf.float32) / 255.

    target_im = tf.io.read_file(target_im)
    target_im = tf.image.decode_jpeg(target_im, channels=3)
    target_im = tf.image.resize(target_im, image_size)
    target_im = tf.image.convert_image_dtype(target_im, tf.float32) / 255.

    return input_im, target_im


@tf.function
def random_flip(im, hm):
    def flip(i, h):
        i = tf.image.flip_left_right(i)
        h = tf.image.flip_left_right(h)
        return i, h

    im, hm = tf.cond(tf.random.uniform([]) < .5,
                    lambda: flip(im, hm),
                    lambda: (im, hm))

    return im, hm


@tf.function
def random_crop(im: tf.Tensor, 
                hm: tf.Tensor, 
                image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:

    cropped_image = tf.image.random_crop(
      tf.stack([im, hm], axis=0), size=[2, *image_size, 3])

    return cropped_image[0], cropped_image[1]


################################################################################

def build_dataset(
        input_images_pattern: Union[Sequence[str], str],
        target_images_pattern: Union[Sequence[str], str],
        image_size: Tuple[int, int],
        train_size: float = .9,
        do_hflip: bool = False,
        do_crop: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Builds a training and test dataset composed of image pairs.

    Parameters
    ----------
    input_images_pattern: Union[Sequence[str], str]
        Gets all files matching one or more glob patterns and use them as 
        input images for the model.
    target_images_pattern: Union[Sequence[str], str]
        Gets all files matching one or more glob patterns and use them as 
        targets images for the model.
    image_size: Tuple[int, int]
        Size to resize the images. [HEIGHT, WIDTH]
    train_size: float, default .9   
        Percentage of samples that will belong to the train set
    do_hflip: bool, default False
        Whether or not to perform random horizontal flip on the training set
    do_crop: bool, default False
        Whether or not to perform random crop on the training set

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        Training and test datasets
    """
    # If we randomly crop the image, we expand the initial input image sizes
    if do_crop:
        h, w = image_size
        train_read_images_fn = functools.partial(read_images,
                                                 image_size=(int(h * 1.1), 
                                                             int(w * 1.1)))
        test_read_images_fn = functools.partial(read_images, 
                                                image_size=image_size)
    else:
        train_read_images_fn = functools.partial(read_images, 
                                                image_size=image_size)
        test_read_images_fn = train_read_images_fn

    # List images under the given directories
    input_paths = tf.data.Dataset.list_files(input_images_pattern, 
                                             shuffle=False)
    target_paths = tf.data.Dataset.list_files(target_images_pattern, 
                                              shuffle=False)

    # Compute the training dataset size
    ds_len = sum(1 for _ in input_paths)
    train_ds_len = int(ds_len * train_size)

    ds = tf.data.Dataset.zip((input_paths, target_paths))

    # Building training set
    train_ds = ds.take(train_ds_len).shuffle(1024)
    train_ds = train_ds.map(train_read_images_fn)

    if do_hflip:
        train_ds = train_ds.map(random_flip)

    if do_crop:
        crop_fn = functools.partial(random_crop, image_size=image_size)
        train_ds = train_ds.map(crop_fn)

    # Building test set
    test_ds = ds.skip(train_ds_len)
    test_ds = test_ds.map(test_read_images_fn)

    return train_ds, test_ds
