# DeepErase - Ink Artifact Removal

This repository contains the implementation of the de-noising model described in
[DeepErase: Weakly Supervised Ink Artifact Removal in Document Text Images
](https://arxiv.org/abs/1910.07070).

Our implementation also has the ability to work with Generative Adversarial 
Networks (GAN) framework to improve the noise removal.

## Training

```python
import deeperase

input_pattern = "data/noised/*.jpg"
target_pattern = "data/clean/*.jpg"

image_size = (64, 256)

train_ds, test_ds = deeperase.data.build_dataset(
    input_pattern, target_pattern, image_size=image_size,
    do_crop=True, do_hflip=True) # Also with image augmentation 😲

generator = deeperase.build_unet(image_size + (3,), 
                                 backbone='mobilenetv2', # See deeperase/backbones.py
                                 n_channels=3) # rgb

discriminator = deeperase.build_discriminator(image_size + (3,))
# If discriminator is not None the noise removal model will also use an 
# adversarial loss

de = deeperase.DeepErase(generator, discriminator)
de.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-4, beta_1=.5), 
           discriminator_optimizer=tf.optimizers.Adam(learning_rate=2e-4, 
                                                      beta_1=.5))

de.fit(train_ds.batch(64), epochs=50, 
       callbacks=[deeperase.callbacks.TensorBoard(logdir='logs,
                                                  images_dataset=test_ds)])
# Take a look at Tensorboard to visualize how predictions evolve 📈
```

## Prediction examples



## Authors

* [Guillem Orellana Trullols](https://github.com/Guillem96)

* [Josep Pon Farreny](https://github.com/jponf)