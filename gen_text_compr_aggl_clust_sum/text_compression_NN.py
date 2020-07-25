import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.constraints import MinMaxNorm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
tf.keras.backend.set_floatx("float64")
tf.executing_eagerly()


class Generator(tf.keras.Model):

    def __init__(self,
                 gen_input_random_noise_size: int,
                 gen_hidden1_size: int,
                 gen_output_size: int):
        """ Definition of the Generator Layers """
        super().__init__(name='generator')
        self.input_layer = tf.keras.layers.Dense(units=gen_input_random_noise_size,
                                                 activation=tf.nn.leaky_relu,
                                                 kernel_regularizer="l1",
                                                 activity_regularizer="l1")
        self.input_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.hidden1 = tf.keras.layers.Dense(units=gen_hidden1_size,
                                             activation=tf.nn.leaky_relu,
                                             kernel_regularizer="l1",
                                             activity_regularizer="l1")
        self.hidden1_dropout = tf.keras.layers.Dropout(rate=0.2)
        self.output_layer = tf.keras.layers.Dense(units=gen_output_size,
                                                  activation=tf.nn.leaky_relu,
                                                  kernel_regularizer="l1",
                                                  activity_regularizer="l1",
                                                  kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0))
        self.output_dropout = tf.keras.layers.Dropout(rate=0.4)

    def call(self, input_tensor, **kwargs):
        """ Definition of Forward Pass """
        x = self.input_layer(input_tensor)
        x = self.input_dropout(x)
        x = self.hidden1(x)
        x = self.hidden1_dropout(x)
        x = self.output_layer(x)
        x = self.output_dropout(x)
        return x

    def generate_noise(self, batch_size: int, random_noise_size: int):
        """ Method for generate the startup noise input tensor of the generator """
        return np.random.uniform(0, 1, size=(batch_size, random_noise_size))


class Discriminator(tf.keras.Model):

    def __init__(self,
                 discr_encoder_input_size: int,
                 discr_encoder_hidden1_size: int,
                 discr_encoder_output_size: int,
                 discr_decoder_input_size: int,
                 discr_decoder_hidden1_size: int,
                 discr_decoder_output_size: int):
        """ Definition of the Discriminator Layers """
        super(Discriminator, self).__init__()

        ### ENCODER ###
        self.encoder_input = tf.keras.layers.InputLayer(input_shape=(discr_encoder_input_size,))  # not indispensable
        # self.encoder_noise = tf.keras.layers.GaussianNoise(stddev=discr_noise_std)
        self.encoder_input_dropout = tf.keras.layers.Dropout(rate=0.4)
        self.encoder_hidden1 = tf.keras.layers.Dense(units=discr_encoder_hidden1_size,
                                                     activation=tf.nn.leaky_relu,
                                                     kernel_regularizer="l1",
                                                     activity_regularizer="l1")
        self.encoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.encoder_output = tf.keras.layers.Dense(units=discr_encoder_output_size,
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_regularizer="l1",
                                                    activity_regularizer="l1",
                                                    # kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0)
                                                    )

        ### DECODER ###
        self.decoder_input = tf.keras.layers.Input(shape=(discr_decoder_input_size,))  # not indispensable
        self.decoder_hidden1 = tf.keras.layers.Dense(units=discr_decoder_hidden1_size,
                                                     activation=tf.nn.leaky_relu,
                                                     kernel_regularizer="l1",
                                                     activity_regularizer="l1")
        self.decoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.decoder_output = tf.keras.layers.Dense(units=discr_decoder_output_size,
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_regularizer="l1",
                                                    activity_regularizer="l1",
                                                    kernel_constraint=MinMaxNorm(min_value=0.0, max_value=1.0))
        self.decoder_output_dropout = tf.keras.layers.Dropout(rate=0.4)

    def encode(self, inputs):
        x = self.encoder_input_dropout(inputs)
        x = self.encoder_hidden1(x)
        x = self.encoder_hidden1_dropout(x)
        x = self.encoder_output(x)
        return x

    def decode(self, inputs):
        x = self.decoder_hidden1(inputs)
        x = self.decoder_hidden1_dropout(x)
        x = self.decoder_output(x)
        x = self.decoder_output_dropout(x)
        return x

    def call(self, x_train, **kwargs):
        latent_representation = self.encode(x_train)
        output = self.decode(latent_representation)
        return output


class GenerativeTextCompressionNN(tf.keras.Model):
    """
        Generative Text Compression neural network:
        a custom network for dimensional reduction of textual documents based on the concepts of GAN and autoencoder
    """

    def __init__(self,
                 num_epoches: int,
                 batch_size: int,
                 gen_learning_rate: float,
                 discr_learning_rate: float,
                 gen_input_random_noise_size: int,
                 gen_hidden1_size: int,
                 gen_output_size: int,
                 discr_encoder_input_size: int,
                 discr_encoder_hidden1_size: int,
                 discr_encoder_output_size: int,
                 discr_decoder_input_size: int,
                 discr_decoder_hidden1_size: int,
                 discr_decoder_output_size: int):
        super().__init__(name="gan")
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.gen_learning_rate = gen_learning_rate
        self.discr_learning_rate = discr_learning_rate
        self.gen_input_random_noise_size = gen_input_random_noise_size
        self.gen_hidden1_size = gen_hidden1_size
        self.gen_output_size = gen_output_size
        self.discr_encoder_input_size = discr_encoder_input_size
        self.discr_encoder_hidden1_size = discr_encoder_hidden1_size
        self.discr_encoder_output_size = discr_encoder_output_size
        self.discr_decoder_input_size = discr_decoder_input_size
        self.discr_decoder_hidden1_size = discr_decoder_hidden1_size
        self.discr_decoder_output_size = discr_decoder_output_size
        self.generator_optimizer = tf.optimizers.Adam(learning_rate=gen_learning_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.optimizers.Adam(learning_rate=discr_learning_rate, beta_1=0.5)
        self.is_trained = False
        self.initial_random_noise = gen_input_random_noise_size
        self.generator = None
        self.discriminator = None
        self.is_built = False

    def _set_item_size(self, num_features: int):
        self.gen_output_size = num_features
        self.discr_encoder_input_size = num_features
        self.discr_decoder_output_size = num_features

    def _check_built_status(self):
        if not self.is_built:
            raise Exception("The network has not yet been built!")

    def build_network(self, num_features: int):
        self._set_item_size(num_features)
        self.generator = Generator(gen_input_random_noise_size=self.gen_input_random_noise_size,
                                   gen_hidden1_size=self.gen_hidden1_size,
                                   gen_output_size=self.gen_output_size)
        print(self.discr_encoder_input_size)
        self.discriminator = Discriminator(discr_encoder_input_size=self.discr_encoder_input_size,
                                           discr_encoder_hidden1_size=self.discr_encoder_hidden1_size,
                                           discr_encoder_output_size=self.discr_encoder_output_size,
                                           discr_decoder_input_size=self.discr_decoder_input_size,
                                           discr_decoder_hidden1_size=self.discr_decoder_hidden1_size,
                                           discr_decoder_output_size=self.discr_decoder_output_size)
        self.is_built = True

    def compute_generator_loss(self, generated_input, fake_output):
        # mse = tf.keras.losses.MeanSquaredError()
        cos_sim = tf.keras.losses.CosineSimilarity()
        fake_loss = cos_sim(generated_input, fake_output)
        return fake_loss

    def compute_discriminator_loss(self, real_input, real_output, generated_input, fake_output):
        # mse = tf.keras.losses.MeanSquaredError()
        cos_sim = tf.keras.losses.CosineSimilarity()
        real_loss = cos_sim(real_input, real_output)
        fake_loss = cos_sim(generated_input, fake_output)
        total_loss = -(tf.abs(real_loss) + tf.abs(fake_loss))
        return total_loss

    # @tf.function()
    def training_step(self,
                      batch: np.ndarray,
                      batch_size: int):
        self._check_built_status()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            ### Produce startup noise tensor ###
            noise = self.generator.generate_noise(batch_size=batch_size,
                                                  random_noise_size=self.initial_random_noise)

            ### Generate synthetic sample with the generator ###
            generated_batch = self.generator(noise)  # training=True

            ### Calculate 'Real Output' / 'Fake Output' tensors for the discriminator ###
            real_discr_output = self.discriminator(batch)  # training=True
            # print("\n real:", real_output.numpy().shape)
            fake_discr_output = self.discriminator(generated_batch)  # training=True
            # print("\n fake:", fake_output.numpy().shape)

            ### Compute Cost Functions Losses ###
            generator_loss = self.compute_generator_loss(generated_input=generated_batch,
                                                         fake_output=fake_discr_output)
            discriminator_loss = self.compute_discriminator_loss(real_input=batch,
                                                                 real_output=real_discr_output,
                                                                 generated_input=generated_batch,
                                                                 fake_output=fake_discr_output)

            ### Calculate Gradients ###
            gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            ### Apply Gradients ###
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                         self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.trainable_variables))
            return generator_loss, discriminator_loss

    def train(self, dataset):
        self._check_built_status()
        logging.info("Start of training process:")
        # Iterate over the epochs
        for i, epoch in enumerate(range(self.num_epoches)):
            logging.info(">>> EPOCH " + str(i + 1))
            # Iterate over the batches
            epoch_generator_losses = []
            epoch_discriminator_losses = []
            for x in tqdm(range(0, len(dataset), self.batch_size)):
                batch = np.array(dataset[x: x + self.batch_size, :])
                generator_loss, discriminator_loss = self.training_step(batch=batch, batch_size=self.batch_size)
                epoch_generator_losses.append(generator_loss.numpy())
                epoch_discriminator_losses.append(discriminator_loss.numpy())
            logging.info("\t\t epoch mean generator loss: " + '{:12f}'.format(np.mean(epoch_generator_losses)) +
                         "  (-/+" + '{:12f}'.format(np.std(epoch_generator_losses)) + ")")
            logging.info("\t\t epoch mean discriminator loss: " + '{:12f}'.format(np.mean(epoch_discriminator_losses)) +
                         "  (-/+" + '{:12f}'.format(np.std(epoch_discriminator_losses)) + ")")
        self.is_trained = True

    def generate_synthetic_samples(self, x_new: np.ndarray) -> np.ndarray:
        self._check_built_status()
        if not self.is_trained:
            raise Exception("The Generative Adversarial Network is not trained!")
        return self.generator(x_new).numpy()

    def get_latent_space(self, x_new) -> np.ndarray:
        """ Getting the compressed Latent Space of the new data given in input """
        self._check_built_status()
        if not self.is_trained:
            raise Exception("The autoencoder is not trained!")
        return self.discriminator.encode(x_new).numpy()
