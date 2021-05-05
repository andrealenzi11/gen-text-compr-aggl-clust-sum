import copy
import math
from typing import List, Tuple, Dict, Union, Set, Optional, Sequence

import numpy as np
from gensim.models import KeyedVectors
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from gtcacs.text_compression2 import GenerativeTextCompressionNN


class GTCACS:
    """
        Generative Text Compression with Agglomerative Clustering Summarization (GTCACS):
        a NLP model for topic extraction, presented in the paper
        'Therapy Analytics using a Patient-centered Perspective: an application to Hypothyroidism'
    """

    def __init__(
            self,
            num_topics: int,
            max_num_words: int = 50,
            max_df: Union[float, int] = 1.0,
            min_df: Union[int, float] = 2,
            stopwords: Set[str] = None,
            ngram_range: Tuple[int, int] = (1, 1),
            lowercase: bool = True,
            max_features: Union[int, None] = None,
            num_epoches: int = 5,
            batch_size: int = 64,
            gen_learning_rate: float = 0.001,
            discr_learning_rate: float = 0.005,
            random_seed_size: int = 128,
            generator_hidden_dim: int = 256,
            discriminator_hidden_dim: int = 256,
            document_dim: Optional[int] = None,
            # latent_space_dim: int = 64,
            embeddings: Optional[KeyedVectors] = None,
    ):
        """
            Initialization

            Parameters
            ----------
            num_topics : (*int*)
                number of topics

            max_num_words : (*int*)
                maximum number of terms to consider for topic

            max_df : (*Union[float, int]*)
                maximum document frequency

            min_df : (*Union[float, int]*)
                minimum document frequency

            stopwords : (*Set[str]*)
                stopwords set

            ngram_range: (**)
                range for ngram

            lowercase: (**)
                flag for convert to lowercase

            max_features: (**)
                maximum number of terms to consider (max vocabulary size)

            num_epoches: (**)
                number of epochs

            batch_size: (**)
                number of documents in a batch

            gen_learning_rate: (**)
                learning rate for optimize the generative part

            discr_learning_rate: (**)
                learning rate for optimize the discriminative part

            random_seed_size: (**)
                dimension of generator input layer

            generator_hidden_dim: (**)
                dimension of generator hidden layer

            document_dim: (**)
                dimension of generator output layer and discriminator's input/output layer

            #latent_space_dim: (**)
                dimension of discriminator latent space

            discriminator_hidden_dim: (**)
                dimension of discriminator hidden layer
        """

        self.num_topics = num_topics
        self.max_num_words = max_num_words

        # Vectorization Parameters
        self.max_df = max_df
        self.min_df = min_df
        self.stopwords = stopwords
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase

        # Compression NN Parameters
        self.num_epoches = num_epoches
        self.batch_size = batch_size
        self.gen_learning_rate = gen_learning_rate
        self.discr_learning_rate = discr_learning_rate
        self.random_seed_size = random_seed_size
        self.generator_hidden_dim = generator_hidden_dim
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.document_dim = document_dim
        self.latent_space_dim = self.num_topics
        # self.latent_space_dim = latent_space_dim

        # Embeddings optimization Parameters
        self.embeddings = embeddings  # gensim keyed vectors
        if self.embeddings:
            self.multiplicative_factor = 3
        else:
            self.multiplicative_factor = 1

        # States
        self.corpus_transformed_shape = None
        self.is_fitted = False
        self.clusters_labels = None
        self.terms_frequency_map = None
        self.topics_matrix = None
        self.topics_distribution = None

        # Models that will be fitted on data
        self.vectorizer_model = Pipeline(
            steps=[
                ("vect", TfidfVectorizer(ngram_range=self.ngram_range,
                                         stop_words=self.stopwords,
                                         max_df=self.max_df,
                                         min_df=self.min_df,
                                         lowercase=self.lowercase,
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
        self.corpus_transformed_shape = num_features
        if (not self.document_dim) or (self.document_dim > self.corpus_transformed_shape):
            self.dim_red_model.build_network(num_features=self.corpus_transformed_shape)
        else:
            self.dim_red_model.build_network(num_features=self.document_dim)

    def fit(self, corpus: Sequence[str]):
        """
            Extract topics from the corpus of documents given in input
            (firstly call this method for fit on corpus)

            :param corpus: list of textual documents
        """
        # === text encoding === #
        corpus_transformed = self.vectorizer_model.fit_transform(X=corpus, y=None)
        print(f"\t - corpus size: {corpus_transformed.shape[0]}")
        print(f"\t - vocabulary size: {corpus_transformed.shape[1]}")

        # === dimensional reduction === #
        self._build_compression_network(num_features=corpus_transformed.shape[1])
        self.dim_red_model.train(dataset=corpus_transformed)
        corpus_transformed_compressed = self.dim_red_model.get_latent_space(x_new=corpus_transformed,
                                                                            apply_softmax=False)  # NO probabilities

        # === clustering === #
        print("\n > clustering...")
        try:
            self.clusters_labels = self.clustering_model.fit_predict(X=corpus_transformed_compressed,
                                                                     y=None)
        except MemoryError as ex_mem_clustering:
            print(str(ex_mem_clustering))
            self.clustering_model = MiniBatchKMeans(n_clusters=self.num_topics,
                                                    init='k-means++',
                                                    max_iter=100,
                                                    batch_size=256,
                                                    verbose=0,
                                                    compute_labels=True,
                                                    random_state=None,
                                                    tol=0.0,
                                                    max_no_improvement=10,
                                                    n_init=5)
            self.clusters_labels = self.clustering_model.fit_predict(X=corpus_transformed_compressed,
                                                                     y=None)

        # === extraction of top terms for each topic === #
        print("\n > extraction of top terms for each topic...")
        self.terms_frequency_map = self._compute_terms_frequencies_map(corpus=corpus)
        clusters_partition = self._compute_clusters_partition(corpus=corpus)
        self.topics_matrix = self._compute_topics_matrix(clusters_partition=clusters_partition,
                                                         terms_frequencies_map=self.terms_frequency_map,
                                                         num_top_words=self.max_num_words * self.multiplicative_factor)

        # === perform embeddings optimization === #
        if self.embeddings:
            print("\n > embeddings optimization...")
            self._apply_embeddings_optimization()

        # === NOT apply embeddings optimization, so transform the topics matrix (remove the scores)
        else:
            self.topics_matrix = [[w for w, s in topic_row] for topic_row in copy.deepcopy(self.topics_matrix)]

        # === set fitted flag === #
        self.is_fitted = True

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise ValueError("Not already fitted: call the method 'fit' first!")

    def transform(self, corpus: Sequence[str], apply_softmax: bool = False) -> np.ndarray:
        """
            Return the topics distribution scores in the corpus
        """
        self._check_is_fitted()
        corpus_transformed = self.vectorizer_model.transform(X=corpus)
        return self.dim_red_model.get_latent_space(x_new=corpus_transformed, apply_softmax=apply_softmax)

    def get_topics_words(self) -> List[List[str]]:
        """
            Return the clusters of terms representing discussion topics
        """
        self._check_is_fitted()
        return self.topics_matrix

    def _compute_clusters_partition(self, corpus: Sequence[str]) -> Dict[str, List[str]]:
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
                               num_top_words: int = 50) -> List[List[Tuple[str, float]]]:
        topics_matrix = list()
        for label, cluster_corpus in clusters_partition.items():
            top_cluster_tokens = self._compute_top_tokens(corpus=cluster_corpus,
                                                          terms_frequencies_map=terms_frequencies_map,
                                                          num_top_words=num_top_words)
            topics_matrix.append([(token, score) for token, score in top_cluster_tokens])
        return topics_matrix

    def _compute_terms_frequencies_map(self, corpus: Sequence[str]) -> Dict[str, int]:
        vec = CountVectorizer(ngram_range=self.ngram_range,
                              stop_words=self.stopwords,
                              lowercase=self.lowercase,
                              max_df=1.0, min_df=1,
                              max_features=None)
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq_map = {word: sum_words[0, idx] for word, idx in vec.vocabulary_.items()}
        return words_freq_map

    def _compute_top_tokens(self,
                            corpus: Sequence[str],
                            terms_frequencies_map: Dict[str, int],
                            num_top_words: int) -> List[Tuple[str, float]]:
        vec = CountVectorizer(ngram_range=self.ngram_range,
                              stop_words=self.stopwords,
                              lowercase=self.lowercase,
                              max_df=1.0, min_df=1,
                              max_features=None)
        vec.fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        corpus_words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        corpus_words_freq_normalized = [(term, freq / math.sqrt(terms_frequencies_map[term]))
                                        for term, freq in corpus_words_freq]
        corpus_words_freq_normalized_sorted = sorted(corpus_words_freq_normalized, key=lambda x: x[1], reverse=True)
        return corpus_words_freq_normalized_sorted[0:num_top_words]

    def _apply_embeddings_optimization(self):
        result = list()
        for i, topic in enumerate(self.topics_matrix):
            original_size = len(topic)

            # === Consider only words for which we have an embeddings vector === #
            try:
                topic_words = [word for word, score in copy.deepcopy(topic) if word in self.embeddings.vocab]
            except AttributeError:
                topic_words = [word for word, score in copy.deepcopy(topic) if word in self.embeddings.key_to_index]
            print(i + 1, ")  words in vocabulary: ", len(topic_words), " / ", original_size)

            # === Removing insignificant words === #
            while len(topic_words) > self.max_num_words:
                topic_words.remove(self.embeddings.doesnt_match(words=topic_words))

            # === Add significant words === #
            if len(topic_words) < self.max_num_words:
                for ms_w, ms_s in self.embeddings.most_similar(positive=topic_words,
                                                               topn=self.max_num_words - len(topic_words)):
                    topic_words.append(ms_w)
            result.append(topic_words)

        del self.topics_matrix
        self.topics_matrix = result
