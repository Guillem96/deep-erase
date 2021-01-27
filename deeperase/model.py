from typing import Optional, Tuple, Mapping

import tensorflow as tf

import deeperase

################################################################################

_DEFAULT_CONTENT_LOSS = tf.losses.MeanSquaredError(
    reduction=tf.losses.Reduction.SUM)

_DEFAULT_ADVERSARIAL_LOSS = tf.losses.BinaryCrossentropy()


################################################################################

class DeepErase(tf.keras.Model):

    """
    Builds the model described at 
    DeepErase: Weakly Supervised Ink Artifact Removal in Document Text Images
    (https://arxiv.org/abs/1910.07070).

    The model goal is to remove noise and artifacts for a given image containing
    text.

    Parameters
    ----------
    generator: tf.keras.Model
        Model with generation capabilities. This model should be able to 
        generate the target image given an input image. 
    discriminator: Optional[tf.keras.Model]
        Originally the DeepErase model, does not use a GAN framework to denoise
        the images. Even that, we have empirically observed that using a 
        discriminator the results greatly improve. If the discriminator is not 
        set, the model won't use an adversarial loss to improve the results.
    """
    def __init__(self, 
                 generator: tf.keras.Model, 
                 discriminator: Optional[tf.keras.Model] = None,
                 name: Optional[str] = None):
        super(DeepErase, self).__init__(name=name or 'DeepErase')
        self.generator = generator
        self.discriminator = discriminator
        self.is_gan = discriminator is not None

    def compile(
            self, 
            optimizer: tf.optimizers.Optimizer, 
            content_loss: tf.losses.Loss = _DEFAULT_CONTENT_LOSS,
            discriminator_optimizer: Optional[tf.optimizers.Optimizer] = None,
            adversarial_loss: Optional[tf.losses.Loss] = None):

        super(DeepErase, self).compile()

        if not self.is_gan and discriminator_optimizer is not None:
            raise ValueError("Cannot set discriminator optimizer since the "
                             "discriminator model is None")

        if not self.is_gan and adversarial_loss is not None:
            raise ValueError("Cannot set adversarial loss since the "
                             "discriminator model is None")

        self.optimizer = optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.content_loss_fn = content_loss
        self.adversarial_loss_fn = adversarial_loss or _DEFAULT_ADVERSARIAL_LOSS

    def _train_discriminator_step(self, 
                                  x: tf.Tensor, 
                                  y: tf.Tensor) -> tf.Tensor:
        clean_y = self.generator(x, training=False)

        with tf.GradientTape() as tape:
            # Forward through discriminator
            disc_real_x = self.discriminator([x, y], training=True)
            disc_fake_x = self.discriminator([x, clean_y], training=True)
            
            # Compute discriminator loss (Guess which images are fake)
            d_real_loss = self.adversarial_loss_fn(
                tf.ones_like(disc_real_x), disc_real_x)
            d_fake_loss = self.adversarial_loss_fn(
                tf.zeros_like(disc_fake_x), disc_fake_x)
            discriminator_loss = (d_real_loss + d_fake_loss) / 2.

        grads_discriminator = tape.gradient(
            discriminator_loss, 
            self.discriminator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(
            zip(grads_discriminator, self.discriminator.trainable_variables))

        return discriminator_loss

    def _train_generator_step(self,
                              x: tf.Tensor, 
                              y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        with tf.GradientTape() as tape:
            clean_y = self.generator(x, training=True)

            if self.is_gan:
                # Forward through discriminator
                disc_real_x = self.discriminator([x, y], training=False)
                disc_fake_x = self.discriminator([x, clean_y], training=False)

                # Compute Generator Adversarial Loss (optimize the generator to 
                # cheat the discriminator)
                adversarial_loss = self.adversarial_loss_fn(
                    tf.ones_like(disc_fake_x), disc_fake_x)
            else:
                adversarial_loss = 0

            content_loss = self.content_loss_fn(y, clean_y)
            generator_loss = content_loss + adversarial_loss

        grads_generator = tape.gradient(generator_loss, 
                                        self.generator.trainable_variables)

        self.optimizer.apply_gradients(
            zip(grads_generator, self.generator.trainable_variables))

        return content_loss, adversarial_loss

    def train_step(
            self, 
            batch: Tuple[tf.Tensor, tf.Tensor]) -> Mapping[str, tf.Tensor]:

        x, y = batch
        report_losses = {}

        if self.is_gan:
            disc_loss = self._train_discriminator_step(x, y)
            report_losses['discriminator_loss'] = disc_loss

        content_loss, adversarial_loss = self._train_generator_step(x, y)
        report_losses['generation_content_loss'] = content_loss
        if self.is_gan:
            report_losses['generator_adversarial_loss'] = adversarial_loss

        return report_losses

    def test_step(
            self, 
            batch: Tuple[tf.Tensor, tf.Tensor]) -> Mapping[str, tf.Tensor]:

        x, y = batch
        report_losses = {}

        clean_y = self.generator(x, training=False)

        if self.is_gan:
            disc_real_x = self.discriminator([x, y], training=False)
            disc_fake_x = self.discriminator([x, clean_y], training=False)

            adversarial_loss = self.adversarial_loss_fn(
                    tf.ones_like(disc_fake_x), disc_fake_x)

            d_real_loss = self.adversarial_loss_fn(tf.ones_like(disc_real_x), 
                                                   disc_real_x)
            d_fake_loss = self.adversarial_loss_fn(tf.zeros_like(disc_fake_x), 
                                                   disc_fake_x)
            discriminator_loss = (d_real_loss + d_fake_loss) / 2.

            report_losses['generator_adversarial_loss'] = adversarial_loss
            report_losses['discriminator_loss'] = discriminator_loss
        else:
            adversarial_loss = 0
            discriminator_loss = 0

        content_loss = self.content_loss_fn(y, clean_y)
        report_losses['generation_content_loss'] = content_loss

        return report_losses

    def call(self, inputs, **kwargs):
        return self.generator(inputs, **kwargs)
