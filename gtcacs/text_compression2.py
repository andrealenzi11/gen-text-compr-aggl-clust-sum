import logging
import math
from collections import Counter
from typing import List, Dict, Tuple, Sequence

import gensim
import numpy as np
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.constraints import MinMaxNorm
from tqdm import tqdm

GTC_NN_LOGGER_NAME = "GTC"
logger = logging.getLogger(GTC_NN_LOGGER_NAME)
logger.setLevel(level=logging.INFO)

tf.keras.backend.set_floatx("float64")
tf.executing_eagerly()


class Generator(tf.keras.Model):
    """
        Neural Network that generates new synthetic data
    """

    def __init__(self,
                 gen_input_random_noise_size: int,
                 gen_hidden1_size: int,
                 gen_output_size: int,
                 **kwargs):
        """
            Initialization of the Generator Layers
        """
        super().__init__(name='generator', **kwargs)
        min_value = 0.
        max_value = 1.
        regularization_norm = "l1"
        self.input_layer = tf.keras.layers.Dense(units=gen_input_random_noise_size,
                                                 activation=tf.nn.relu,  # tf.nn.leaky_relu
                                                 kernel_regularizer=regularization_norm,
                                                 activity_regularizer=regularization_norm,
                                                 # kernel_constraint=MinMaxNorm(min_value=min_value,
                                                 #                              max_value=max_value),
                                                 )
        self.input_dropout = tf.keras.layers.Dropout(rate=0.2)
        self.hidden1 = tf.keras.layers.Dense(units=gen_hidden1_size,
                                             activation=tf.nn.relu,
                                             kernel_regularizer=regularization_norm,
                                             activity_regularizer=regularization_norm,
                                             # kernel_constraint=MinMaxNorm(min_value=min_value,
                                             #                              max_value=max_value)
                                             )
        self.hidden1_dropout = tf.keras.layers.Dropout(rate=0.3)
        self.output_layer = tf.keras.layers.Dense(units=gen_output_size,
                                                  activation=tf.nn.relu,
                                                  kernel_regularizer=regularization_norm,
                                                  activity_regularizer=regularization_norm,
                                                  kernel_constraint=MinMaxNorm(min_value=min_value,
                                                                               max_value=max_value)
                                                  )
        self.output_dropout = tf.keras.layers.Dropout(rate=0.4)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_layer': self.input_layer,
            'input_dropout': self.input_dropout,
            'hidden1': self.hidden1,
            'hidden1_dropout': self.hidden1_dropout,
            'output_layer': self.output_layer,
            'output_dropout': self.output_dropout,
        })
        return config

    def call(self, input_tensor, **kwargs):
        """
            Definition of Forward Pass for generation
        """
        x = self.input_layer(input_tensor)
        x = self.input_dropout(x)
        x = self.hidden1(x)
        x = self.hidden1_dropout(x)
        x = self.output_layer(x)
        x = self.output_dropout(x)
        return x

    @staticmethod
    def generate_noise(batch_size: int, random_noise_size: int):
        """
            Method for generate the startup noise input tensor of the generator
        """
        return np.random.uniform(0, 1, size=(batch_size, random_noise_size))


class Discriminator(tf.keras.Model):
    """
        Neural Network (Auto-Encoder) for data compression
    """

    def __init__(self,
                 discr_encoder_input_size: int,
                 discr_encoder_hidden1_size: int,
                 discr_encoder_output_size: int,
                 discr_decoder_input_size: int,
                 discr_decoder_hidden1_size: int,
                 discr_decoder_output_size: int,
                 **kwargs):
        """
            Initialization of the Discriminator Layers
        """
        super().__init__(name='discriminator', **kwargs)
        min_value = 0.
        max_value = 1.
        regularization_norm = "l1"

        # ==================== ENCODER ==================== #
        self.encoder_input = tf.keras.layers.InputLayer(input_shape=(discr_encoder_input_size,))  # not indispensable

        # self.encoder_noise = tf.keras.layers.GaussianNoise(stddev=discr_noise_std)
        self.encoder_input_dropout = tf.keras.layers.Dropout(rate=0.4)
        self.encoder_hidden1 = tf.keras.layers.Dense(units=discr_encoder_hidden1_size,
                                                     activation=tf.nn.relu,
                                                     kernel_regularizer=regularization_norm,
                                                     activity_regularizer=regularization_norm,
                                                     # kernel_constraint=MinMaxNorm(min_value=min_value,
                                                     #                              max_value=max_value)
                                                     )
        self.encoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.encoder_output = tf.keras.layers.Dense(units=discr_encoder_output_size,
                                                    activation=tf.nn.relu,
                                                    kernel_regularizer=regularization_norm,
                                                    activity_regularizer=regularization_norm,
                                                    # kernel_constraint=MinMaxNorm(min_value=min_value,
                                                    #                              max_value=max_value)
                                                    )

        # ==================== DECODER ==================== #
        self.decoder_input = tf.keras.layers.Input(shape=(discr_decoder_input_size,))  # not indispensable
        self.decoder_hidden1 = tf.keras.layers.Dense(units=discr_decoder_hidden1_size,
                                                     activation=tf.nn.relu,
                                                     kernel_regularizer=regularization_norm,
                                                     activity_regularizer=regularization_norm,
                                                     # kernel_constraint=MinMaxNorm(min_value=min_value,
                                                     #                              max_value=max_value)
                                                     )
        self.decoder_hidden1_dropout = tf.keras.layers.Dropout(rate=0.1)
        self.decoder_output = tf.keras.layers.Dense(units=discr_decoder_output_size,
                                                    activation=tf.nn.relu,
                                                    kernel_regularizer=regularization_norm,
                                                    activity_regularizer=regularization_norm,
                                                    kernel_constraint=MinMaxNorm(min_value=min_value,
                                                                                 max_value=max_value)
                                                    )
        self.decoder_output_dropout = tf.keras.layers.Dropout(rate=0.4)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "encoder_input": self.encoder_input,
            "encoder_input_dropout": self.encoder_input_dropout,
            "encoder_hidden1": self.encoder_hidden1,
            "encoder_hidden1_dropout": self.encoder_hidden1_dropout,
            "encoder_output": self.encoder_output,
            "decoder_input": self.decoder_input,
            "decoder_hidden1": self.decoder_hidden1,
            "decoder_hidden1_dropout": self.decoder_hidden1_dropout,
            "decoder_output": self.decoder_output,
            "decoder_output_dropout": self.decoder_output_dropout,
        })
        return config

    def encode(self, inputs):
        """
            Encoding: from input samples to compressed latent spaces vectors
        """
        x = self.encoder_input_dropout(inputs)
        x = self.encoder_hidden1(x)
        x = self.encoder_hidden1_dropout(x)
        x = self.encoder_output(x)
        return x

    def decode(self, inputs):
        """
            Decoding: from compressed latent spaces vectors to reconstructed samples
        """
        x = self.decoder_hidden1(inputs)
        x = self.decoder_hidden1_dropout(x)
        x = self.decoder_output(x)
        x = self.decoder_output_dropout(x)
        return x

    def call(self, x_train, **kwargs):
        """
            Encoding and Decoding
        """
        latent_representation = self.encode(x_train)
        output = self.decode(latent_representation)
        return output


class GenerativeTextCompressionNN(tf.keras.Model):
    """
        Generative Text Compression neural network:
        a custom network for dimensional reduction of textual documents based on the concepts of GAN and auto-encoder
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
                 discr_decoder_output_size: int,
                 **kwargs):
        super().__init__(name="gtc", **kwargs)
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
        self.generator_loss = tf.keras.losses.MeanSquaredError()
        self.discriminator_loss = tf.keras.losses.CosineSimilarity()
        self.real_min_v = None
        self.real_mean_v = None
        self.real_max_v = None
        self.fake_min_v = None
        self.fake_mean_v = None
        self.fake_max_v = None

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "num_epoches": self.num_epoches,
            "batch_size": self.batch_size,
            "gen_learning_rate": self.gen_learning_rate,
            "discr_learning_rate": self.discr_learning_rate,
            "gen_input_random_noise_size": self.gen_input_random_noise_size,
            "gen_hidden1_size": self.gen_hidden1_size,
            "gen_output_size": self.gen_output_size,
            "discr_encoder_input_size": self.discr_encoder_input_size,
            "discr_encoder_hidden1_size": self.discr_encoder_hidden1_size,
            "discr_encoder_output_size": self.discr_encoder_output_size,
            "discr_decoder_input_size": self.discr_decoder_input_size,
            "discr_decoder_hidden1_size": self.discr_decoder_hidden1_size,
            "discr_decoder_output_size": self.discr_decoder_output_size,
            "generator_optimizer": self.generator_optimizer,
            "discriminator_optimizer": self.discriminator_optimizer,
            "is_trained": self.is_trained,
            "initial_random_noise": self.initial_random_noise,
            "generator": self.generator,
            "discriminator": self.discriminator,
            "is_built": self.is_built,
            "current_loss": self.current_loss,
        })
        return config

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
        logger.info(f"auto-encoder input size: {str(self.discr_encoder_input_size)}")
        self.discriminator = Discriminator(discr_encoder_input_size=self.discr_encoder_input_size,
                                           discr_encoder_hidden1_size=self.discr_encoder_hidden1_size,
                                           discr_encoder_output_size=self.discr_encoder_output_size,
                                           discr_decoder_input_size=self.discr_decoder_input_size,
                                           discr_decoder_hidden1_size=self.discr_decoder_hidden1_size,
                                           discr_decoder_output_size=self.discr_decoder_output_size)
        self.is_built = True

    def compute_generator_loss(self,
                               generated_data: EagerTensor,
                               real_data: EagerTensor):
        t1 = tf.concat([
            tf.math.reduce_min(generated_data, axis=0),
            tf.math.reduce_mean(generated_data, axis=0),
            tf.math.reduce_variance(generated_data, axis=0),
            tf.math.reduce_max(generated_data, axis=0),
        ], axis=0)
        t2 = tf.concat([
            tf.math.reduce_min(real_data, axis=0),
            tf.math.reduce_mean(real_data, axis=0),
            tf.math.reduce_variance(real_data, axis=0),
            tf.math.reduce_max(real_data, axis=0),
        ], axis=0)
        t3 = tf.concat([
            tf.math.reduce_min(generated_data, axis=1),
            tf.math.reduce_mean(generated_data, axis=1),
            tf.math.reduce_variance(generated_data, axis=1),
            tf.math.reduce_max(generated_data, axis=1),
        ], axis=0)
        t4 = tf.concat([
            tf.math.reduce_min(real_data, axis=1),
            tf.math.reduce_mean(real_data, axis=1),
            tf.math.reduce_variance(real_data, axis=1),
            tf.math.reduce_max(real_data, axis=1),
        ], axis=0)
        return self.generator_loss(t1, t2) + self.generator_loss(t3, t4)

    def compute_discriminator_loss(self,
                                   real_input: EagerTensor,
                                   real_output: EagerTensor,
                                   generated_input: EagerTensor,
                                   fake_output: EagerTensor):
        # print("real_input", real_input.shape, type(real_input))
        # print("real_output", real_output.shape, type(real_output))
        # print("gen", generated_data.shape, type(generated_data))
        # print("fake", real_data.shape, type(real_data))
        real_loss = self.discriminator_loss(real_input, real_output)
        fake_loss = self.discriminator_loss(generated_input, fake_output)
        return real_loss + fake_loss

    def training_step(self,
                      batch: EagerTensor,
                      batch_size: int, ):
        self._check_built_status()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # === Produce startup noise tensor === #
            noise = self.generator.generate_noise(batch_size=batch_size,
                                                  random_noise_size=self.initial_random_noise)

            # === Generate synthetic sample with the generator === #
            generated_batch = self.generator(noise)  # training=True
            if generated_batch.shape[0] > batch.shape[0]:
                generated_batch = generated_batch[:batch.shape[0], :]

            # === Calculate 'Real Output' / 'Fake Output' tensors for the discriminator === #
            real_discr_output = self.discriminator(batch)  # training=True
            self.real_min_v = tf.reduce_min(batch)
            self.real_mean_v = tf.reduce_mean(batch)
            self.real_max_v = tf.reduce_max(batch)
            fake_discr_output = self.discriminator(generated_batch)  # training=True
            self.fake_min_v = tf.reduce_min(generated_batch)
            self.fake_mean_v = tf.reduce_mean(generated_batch)
            self.fake_max_v = tf.reduce_max(generated_batch)

            # === Compute Cost Functions Losses === #
            generator_loss = self.compute_generator_loss(generated_data=generated_batch,
                                                         real_data=batch)
            discriminator_loss = self.compute_discriminator_loss(real_input=batch,
                                                                 real_output=real_discr_output,
                                                                 generated_input=generated_batch,
                                                                 fake_output=fake_discr_output)

            # === Calculate Gradients === #
            gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

            # === Apply Gradients === #
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                         self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.trainable_variables))
            return generator_loss, discriminator_loss

    def train(self, dataset):
        self._check_built_status()
        logger.info("Start of training process:")
        # Iterate over the epochs
        for i, epoch in enumerate(range(self.num_epoches)):
            logger.info(">>> EPOCH " + str(i + 1))
            # Iterate over the batches
            epoch_generator_losses = []
            epoch_discriminator_losses = []
            for x in tqdm(range(0, len(dataset), self.batch_size)):
                batch = np.array(dataset[x: x + self.batch_size, :])
                batch = tf.convert_to_tensor(batch, dtype=tf.float64)
                generator_loss, discriminator_loss = self.training_step(batch=batch, batch_size=self.batch_size)
                epoch_generator_losses.append(generator_loss.numpy())
                epoch_discriminator_losses.append(discriminator_loss.numpy())
            logger.info("\t\t epoch mean generator loss: " + '{:12f}'.format(np.mean(epoch_generator_losses)) +
                        "  (-/+" + '{:12f}'.format(np.std(epoch_generator_losses)) + ")")
            logger.info("\t\t epoch mean discriminator loss: " + '{:12f}'.format(np.mean(epoch_discriminator_losses)) +
                        "  (-/+" + '{:12f}'.format(np.std(epoch_discriminator_losses)) + ")")
        self.is_trained = True
        print("\n values stats:")
        print("\t - real:")
        print("\t\t", self.real_min_v.numpy(), self.real_mean_v.numpy(), self.real_max_v.numpy())
        print("\t - fake:")
        print("\t\t", self.fake_min_v.numpy(), self.fake_mean_v.numpy(), self.fake_max_v.numpy())

    def generate_synthetic_samples(self, x_new: np.ndarray) -> np.ndarray:
        self._check_built_status()
        if not self.is_trained:
            raise Exception("The Generative Adversarial Network is not trained!")
        return self.generator(x_new).numpy()

    def get_latent_space(self,
                         x_new: Sequence[str],
                         apply_softmax: bool = False) -> np.ndarray:
        """ Getting the compressed Latent Space of the new data given in input """
        self._check_built_status()
        if not self.is_trained:
            raise Exception("The auto-encoder is not trained!")
        encoded_latent_space = self.discriminator.encode(x_new)
        if apply_softmax:
            return tf.keras.activations.softmax(encoded_latent_space, axis=-1).numpy()
        else:
            return encoded_latent_space.numpy()

    # def get_topics_words1(self,
    #                       corpus: List[str],
    #                       latent_spaces: np.ndarray,
    #                       num_topics: int,
    #                       num_top_words: int = 50) -> List[List[str]]:
    #
    #     words = sorted(list(set([word for doc in corpus for word in doc.split()])))  # Vocabulary words
    #     word_id_map = {words[i]: i for i in range(len(words))}  # map that associate to each word its identifier
    #     id_word_map = {i: words[i] for i in range(len(words))}  # map that associate to each identifier its word
    #
    #     words_df = np.zeros(shape=len(words))  # Document Frequency of each word
    #     words_td = np.zeros(shape=(len(words), num_topics))  # Topics Distribution of each word
    #
    #     for i in range(len(corpus)):
    #         doc_words = set()  # unique words of the current document
    #         for word in corpus[i].split():
    #             idx = word_id_map[word]
    #             if word not in doc_words:
    #                 words_df[idx] += 1
    #                 doc_words.add(word)
    #             words_td[idx] += latent_spaces[i]
    #
    #     for w in [
    #         'stephanopoulos', 'mov', 'oname', 'ptr', 'uccxkvb', 'bullock',
    #         'subject', 'organization', 'line', 'mail', 'software', 'power',
    #     ]:
    #         print(f"{w}:",
    #               np.min(words_td[word_id_map[w]]),
    #               np.mean(words_td[word_id_map[w]]),
    #               np.max(words_td[word_id_map[w]]))
    #     # === normalization 1 === #
    #     words_td_normalized = words_td / (np.log(words_df)[:, None] + 1)
    #
    #     # === normalization 2 === #
    #     words_td_normalized = words_td_normalized - words_td_normalized.mean(axis=1)[:, None]
    #
    #     words_td_normalized_argsorted = np.argsort(-words_td_normalized, axis=0)
    #     words_td_normalized_sorted = -np.sort(-words_td_normalized, axis=0)
    #
    #     topics_ids_matrix = words_td_normalized_argsorted.transpose()[:, :num_top_words]
    #     print(topics_ids_matrix.shape)
    #
    #     topics_matrix = np.full(shape=(topics_ids_matrix.shape[0], topics_ids_matrix.shape[1]),
    #                             fill_value="",
    #                             dtype=np.object)
    #     print(topics_matrix.shape)
    #     print(len(word_id_map))
    #
    #     for i in range(topics_ids_matrix.shape[0]):
    #         for j in range(topics_ids_matrix.shape[1]):
    #             idx = topics_ids_matrix[i, j]
    #             topics_matrix[i, j] = id_word_map[idx]
    #
    #     print(topics_matrix)
    #     return topics_matrix

    # def get_topics_words2(self,
    #                       corpus: List[str],
    #                       latent_spaces: np.ndarray,
    #                       num_topics: int,
    #                       num_top_words: int = 50) -> List[List[str]]:
    #     clustering_model = AgglomerativeClustering(
    #         n_clusters=num_topics,
    #         affinity="euclidean",  # "euclidean"
    #         linkage="ward"  # "ward"
    #     )
    #     clusters_labels = clustering_model.fit_predict(X=latent_spaces, y=None)
    #     print(Counter(clusters_labels), "\n")
    #     terms_frequency_map = self._compute_terms_frequencies_map(corpus=corpus)
    #     clusters_partition = self._compute_clusters_partition(corpus=corpus, clusters_labels=clusters_labels)
    #     topics_matrix = self._compute_topics_matrix(clusters_partition=clusters_partition,
    #                                                 terms_frequencies_map=terms_frequency_map,
    #                                                 num_top_words=num_top_words)
    #     result = list()
    #     for i, row in enumerate(topics_matrix):
    #         row_new = list()
    #         for w, s in row:
    #             row_new.append(w)
    #         print(i + 1, len(row_new), row_new)
    #         result.append(row_new)
    #     return result

    # def get_topics_words3(self,
    #                       corpus: List[str],
    #                       words: List[str],
    #                       latent_spaces_corpus: np.ndarray,
    #                       latent_spaces_words: np.ndarray,
    #                       num_topics: int,
    #                       num_top_words: int = 50) -> List[List[str]]:
    #
    #     # clustering
    #     clustering_model = AgglomerativeClustering(
    #         n_clusters=num_topics,
    #         affinity="cosine",
    #         linkage="average"
    #     )
    #     clusters_labels_corpus = clustering_model.fit_predict(X=latent_spaces_corpus, y=None)
    #
    #     # partition
    #     label_vectors_map = dict()
    #     for label, compressed_vector in zip(clusters_labels_corpus, latent_spaces_corpus):
    #         try:
    #             label_vectors_map[label].append(compressed_vector)
    #         except KeyError:
    #             label_vectors_map[label] = [compressed_vector]
    #
    #     # compute cluster centroid
    #     centroids = list()
    #     for label, vectors in label_vectors_map.items():
    #         centroids.append(np.mean(vectors, axis=0))
    #
    #     print("words_size:", len(words))
    #     cosine_sim_matrix = list()
    #     for i, centroid_vector in enumerate(centroids):
    #         print(i + 1, " | ", centroid_vector.shape)
    #         cos_sim_scores = cosine_similarity(latent_spaces_words,
    #                                            np.array([list(centroid_vector)])).flatten()
    #         print(cos_sim_scores.shape, cos_sim_scores)
    #         print("min:", round(np.min(cos_sim_scores), 4), "  |  ",
    #               "mean:", round(np.mean(cos_sim_scores), 4), "  |  ",
    #               "max:", round(np.max(cos_sim_scores), 4))
    #         cosine_sim_matrix.append(cos_sim_scores)
    #
    #     cosine_sim_matrix = np.array(cosine_sim_matrix)
    #     cosine_sim_matrix_normalized = cosine_sim_matrix - cosine_sim_matrix.mean(axis=0)[None, :]
    #
    #     topics_matrix = list()
    #     for row in cosine_sim_matrix_normalized:
    #         top_indices = np.argsort(row)[:num_top_words]
    #         topic_top_words = [words[index] for index in top_indices]
    #         print(topic_top_words)
    #         topics_matrix.append(topic_top_words)
    #
    #     return topics_matrix

    def get_topics_words_final(self,
                               corpus: List[str],
                               latent_spaces: np.ndarray,
                               num_topics: int,
                               keyed_vectors: gensim.models.keyedvectors.KeyedVectors,
                               num_top_words: int = 50) -> List[List[str]]:
        print(type(keyed_vectors))
        clustering_model = AgglomerativeClustering(
            n_clusters=num_topics,
            affinity="euclidean",  # "euclidean"
            linkage="ward"  # "ward"
        )
        clusters_labels = clustering_model.fit_predict(X=latent_spaces, y=None)
        print(Counter(clusters_labels), "\n")
        terms_frequency_map = self._compute_terms_frequencies_map(corpus=corpus)
        clusters_partition = self._compute_clusters_partition(corpus=corpus, clusters_labels=clusters_labels)
        factor = 3
        topics_matrix = self._compute_topics_matrix(clusters_partition=clusters_partition,
                                                    terms_frequencies_map=terms_frequency_map,
                                                    num_top_words=num_top_words * factor)
        result = list()
        for i, row in enumerate(topics_matrix):
            row_new = list()
            for w, s in row:
                row_new.append(w)
            print(i + 1, len(row_new), row_new)
            result.append(row_new)

        result2 = list()
        for i, topic in enumerate(result):
            original_size = len(topic)
            topic = [word for word in topic.copy() if word in keyed_vectors.vocab]
            print(i + 1, ")  words in vocabulary: ", len(topic), " / ", original_size)
            while len(topic) > num_top_words:
                topic.remove(keyed_vectors.doesnt_match(words=topic))
            result2.append(topic)

            # while len(topic) > num_top_words - 10:
            #     topic.remove(keyed_vectors.doesnt_match(words=topic))
            # most_similar_words = keyed_vectors.most_similar(positive=topic, negative=None, topn=10)
            # print(most_similar_words)
            # for w, s in most_similar_words:
            #     topic.append(w)
            # result2.append(topic)

        print(result2)
        print([len(row) for row in result2])

        return result2

    @staticmethod
    def _compute_terms_frequencies_map(corpus: List[str]) -> Dict[str, int]:
        vec = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                              lowercase=True, max_df=1.0, min_df=1, max_features=None, )
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq_map = {word: sum_words[0, idx] for word, idx in vec.vocabulary_.items()}
        return words_freq_map

    @staticmethod
    def _compute_clusters_partition(corpus: List[str], clusters_labels) -> Dict[str, List[str]]:
        clusters_partition = dict()
        for i, label in enumerate(clusters_labels):
            if label in clusters_partition:
                clusters_partition[label].append(corpus[i])
            else:
                clusters_partition[label] = [corpus[i]]
        return clusters_partition

    def _compute_topics_matrix(self,
                               clusters_partition: Dict[str, List[str]],
                               terms_frequencies_map: Dict[str, int],
                               num_top_words: int = 100) -> List[List[Tuple[str, float]]]:
        topics_matrix = []
        for label, cluster_corpus in clusters_partition.items():
            top_cluster_tokens = self._compute_top_tokens(corpus=cluster_corpus,
                                                          terms_frequencies_map=terms_frequencies_map,
                                                          num_top_words=num_top_words)
            topics_matrix.append([(token, score) for token, score in top_cluster_tokens])
        return topics_matrix

    @staticmethod
    def _compute_top_tokens(corpus: List[str],
                            terms_frequencies_map: Dict[str, int],
                            num_top_words: int) -> List[Tuple[str, float]]:
        vec = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                              lowercase=True, max_df=0.8, min_df=5, max_features=None)
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        corpus_words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        corpus_words_freq_normalized = [(term, freq / math.sqrt(terms_frequencies_map[term])) for term, freq in
                                        corpus_words_freq]
        corpus_words_freq_normalized_sorted = sorted(corpus_words_freq_normalized, key=lambda x: x[1], reverse=True)
        return corpus_words_freq_normalized_sorted[0:num_top_words]
