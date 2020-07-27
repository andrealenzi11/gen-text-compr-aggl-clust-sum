import math
from typing import List, Tuple, Dict, Union, Set

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from gtcacs.text_compression_NN import GenerativeTextCompressionNN


class GTCACS:
    """
        Generative Text Compression with Agglomerative Clustering Summarization (GTCACS):
        a NLP model for topic extraction presented in the paper
        'Therapy Analytics using a Patient-centered Perspective: an application to Hypothyroidism'
    """

    def __init__(self,
                 num_topics: int,
                 max_num_words: int = 100,
                 max_df: Union[float, int] = 1.0,
                 min_df: Union[int, float] = 2,
                 stopwords: Set[str] = None,
                 ngram_range: Tuple[int, int] = (1, 1),
                 lowercase: bool = True,
                 max_features: Union[int, None] = None,
                 num_epoches: int = 5,
                 batch_size: int = 64,
                 gen_learning_rate: float = 0.005,
                 discr_learning_rate: float = 0.005,
                 random_seed_size: int = 100,
                 generator_hidden_dim: int = 512,
                 document_dim: int = None,
                 latent_space_dim: int = 64,
                 discriminator_hidden_dim: int = 256):
        """
            Initialization

            Attributes
            ----------
            :param num_topics:
            :param max_num_words:
            :param max_df:
            :param min_df:
            :param stopwords:
            :param ngram_range:
            :param lowercase:
            :param max_features:
            :param num_epoches:
            :param batch_size:
            :param gen_learning_rate:
            :param discr_learning_rate:
            :param random_seed_size:
            :param generator_hidden_dim:
            :param document_dim:
            :param latent_space_dim:
            :param discriminator_hidden_dim:
        """

        self.num_topics = num_topics
        self.max_num_words = max_num_words

        # vectorization
        self.max_df = max_df
        self.min_df = min_df
        self.stopwords = stopwords
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase

        # compression NN
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.gen_learning_rate = gen_learning_rate
        self.discr_learning_rate = discr_learning_rate
        self.random_seed_size = random_seed_size
        self.generator_hidden_dim = generator_hidden_dim
        self.document_dim = document_dim
        self.latent_space_dim = latent_space_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim

        # states
        self.corpus_transformed_shape = None
        self.is_fitted = False
        self.clusters_labels = None
        self.terms_frequency_map = None
        self.topics_matrix = None
        self.topics_distribution = None

        # models
        self.vectorizer_model = Pipeline(
            steps=[
                ("vect", TfidfVectorizer(ngram_range=self.ngram_range, stop_words=self.stopwords,
                                         max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         max_features=self.max_features)),
                ("dense", FunctionTransformer(csr_matrix.todense))
            ]
        )
        self.dim_red_model = GenerativeTextCompressionNN(
            num_epoches=self.num_epoches,
            batch_size=self.batch_size,
            gen_learning_rate=self.gen_learning_rate,
            discr_learning_rate=self.discr_learning_rate,
            gen_input_random_noise_size=self.random_seed_size,
            gen_hidden1_size=self.generator_hidden_dim,
            gen_output_size=self.document_dim,
            discr_encoder_input_size=self.document_dim,
            discr_encoder_hidden1_size=self.discriminator_hidden_dim,
            discr_encoder_output_size=self.latent_space_dim,
            discr_decoder_input_size=self.latent_space_dim,
            discr_decoder_hidden1_size=self.discriminator_hidden_dim,
            discr_decoder_output_size=self.document_dim
        )
        self.clustering_model = AgglomerativeClustering(
            n_clusters=self.num_topics,
            affinity="euclidean",
            linkage="ward"
        )

    def _build_compression_network(self, num_features: int):
        """

            Parameters
            ----------
            :param num_features:
        """
        self.corpus_transformed_shape = num_features
        if (not self.document_dim) or (self.document_dim > self.corpus_transformed_shape):
            self.dim_red_model.build_network(num_features=self.corpus_transformed_shape)
        else:
            self.dim_red_model.build_network(num_features=self.document_dim)

    def extract_topics(self, corpus: List[str]):
        """

            Parameters
            ----------
            :param corpus:
        """
        corpus_transformed = self.vectorizer_model.fit_transform(X=corpus, y=None)
        self._build_compression_network(num_features=corpus_transformed.shape[1])
        self.dim_red_model.train(dataset=corpus_transformed)
        corpus_transformed_compressed = self.dim_red_model.get_latent_space(x_new=corpus_transformed)
        self.clusters_labels = self.clustering_model.fit_predict(X=corpus_transformed_compressed, y=None)
        self.terms_frequency_map = self._compute_terms_frequencies_map(corpus=corpus)
        clusters_partition = self._compute_clusters_partition(corpus=corpus)
        self.topics_matrix = self._compute_topics_matrix(clusters_partition=clusters_partition,
                                                         terms_frequencies_map=self.terms_frequency_map,
                                                         num_top_words=self.max_num_words)
        self.topics_distribution = self._compute_topics_distribution(corpus=corpus)
        self.is_fitted = True

    def _check_is_fitted(self):
        """

        """
        if not self.is_fitted:
            raise ValueError("The topics are not already extracted: call 'extract_topics' first!")

    def get_topics_distribution_scores(self) -> np.ndarray:
        """

            :return:
        """
        self._check_is_fitted()
        return self.topics_distribution

    def get_topics_words(self) -> List[List[Tuple[str, float]]]:
        """
            :return:
        """
        self._check_is_fitted()
        return self.topics_matrix

    def _compute_topics_distribution(self, corpus: List[str]) -> np.ndarray:
        """
            Parameters
            ----------
            :param corpus:
            :return:
        """
        vec = CountVectorizer(ngram_range=self.ngram_range, stop_words=self.stopwords, lowercase=self.lowercase,
                              max_df=self.max_df, min_df=self.min_df, max_features=self.max_features)
        word_tokenizer_fun = vec.build_tokenizer()
        topics_matrix_tmp = [dict(topic_list) for topic_list in self.topics_matrix]
        result = np.zeros(shape=(len(corpus), self.num_topics))
        for i, doc in enumerate(corpus):
            for token in word_tokenizer_fun(doc):
                for j, topic in enumerate(topics_matrix_tmp):
                    try:
                        result[i, j] += topic[token]
                    except KeyError:
                        continue
        return result

    def _compute_clusters_partition(self, corpus: List[str]) -> Dict[str, List[str]]:
        """

            Parameters
            ----------
            :param corpus:
            :return:
        """
        clusters_partition = dict()
        for i, label in enumerate(self.clusters_labels):
            if label in clusters_partition:
                clusters_partition[label].append(corpus[i])
            else:
                clusters_partition[label] = [corpus[i]]
        return clusters_partition

    def _compute_topics_matrix(self,
                               clusters_partition: Dict[str, List[str]],
                               terms_frequencies_map: Dict[str, int],
                               num_top_words: int = 100) -> List[List[Tuple[str, float]]]:
        """
            Parameters
            ----------
            :param clusters_partition:
            :param terms_frequencies_map:
            :param num_top_words:
            :return:
        """
        topics_matrix = []
        for label, cluster_corpus in clusters_partition.items():
            top_cluster_tokens = self._compute_top_tokens(corpus=cluster_corpus,
                                                          terms_frequencies_map=terms_frequencies_map,
                                                          num_top_words=num_top_words)
            topics_matrix.append([(token, score) for token, score in top_cluster_tokens])
        return topics_matrix

    def _compute_terms_frequencies_map(self, corpus: List[str]) -> Dict[str, int]:
        """
            Parameters
            ----------
            :param corpus:
            :return:
        """
        vec = CountVectorizer(ngram_range=self.ngram_range, stop_words=self.stopwords,
                              lowercase=self.lowercase, max_df=1.0, min_df=1, max_features=None, )
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq_map = {word: sum_words[0, idx] for word, idx in vec.vocabulary_.items()}
        return words_freq_map

    def _compute_top_tokens(self,
                            corpus: List[str],
                            terms_frequencies_map: Dict[str, int],
                            num_top_words: int) -> List[Tuple[str, float]]:
        """
            Parameters
            ----------
            :param corpus:
            :param terms_frequencies_map:
            :param num_top_words:
            :return:
        """
        vec = CountVectorizer(ngram_range=self.ngram_range, stop_words=self.stopwords,
                              lowercase=self.lowercase, max_df=1.0, min_df=1, max_features=None)
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        corpus_words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        corpus_words_freq_normalized = [(term, freq / math.sqrt(terms_frequencies_map[term])) for term, freq in
                                        corpus_words_freq]
        corpus_words_freq_normalized_sorted = sorted(corpus_words_freq_normalized, key=lambda x: x[1], reverse=True)
        return corpus_words_freq_normalized_sorted[0:num_top_words]
