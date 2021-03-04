****************************************************************************************************************
gtcacs
****************************************************************************************************************

A library for topic modeling based on the algorithm: 
*Generative Text Compression with Agglomerative Clustering Summarization (GTCACS)*.

|

Installation
########################################################################################################


Use the package manager pip to install *gtcacs*.

.. code:: bash

	pip3 install gtcacs


Tested Python version:

.. code:: bash

    python3.8


Tested dependencies:

.. code:: bash

    scikit-learn==0.24.1
    tensorflow==2.4.1
    tqdm==4.58.0

|

Usage
################################################################################################################################################

.. code:: python

	from sklearn.datasets import fetch_20newsgroups
	from gtcacs.topic_modeling import GTCACS

	# load dataset
	corpus, labels = fetch_20newsgroups(subset='all', return_X_y=True, download_if_missing=False)

	# set stop words
	eng_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

	# instantiate the GTCACS object
	gtcacs_obj = GTCACS(
		num_topics=20,                # number of topics
		max_num_words=50,             # maximum number of terms to consider
		max_df=0.95,                  # maximum document frequency
		min_df=15,                    # minimum document frequency
		stopwords=eng_stopwords,      # stopwords set
		ngram_range=(1, 2),           # range for ngram
		max_features=None,            # maximum number of terms to consider (max vocabulary size)
		lowercase=True,               # flag for convert to lowercase
		num_epoches=5,                # number of epochs
		batch_size=128,               # number of documents in a batch
		gen_learning_rate=0.005,      # learning rate for optimize the generative part
		discr_learning_rate=0.005,    # learning rate for optimize the discriminative part
		random_seed_size=100,         # dimension of generator input layer
		generator_hidden_dim=512,     # dimension of generator hidden layer
		document_dim=None,            # dimension of generator output layer and discriminator's input/output layer
		latent_space_dim=64,          # dimension of discriminator latent space
		discriminator_hidden_dim=256  # dimension of discriminator hidden layer
	)

	# compuation on corpus (dimensional reduction, clustering, summarization)
	gtcacs_obj.extract_topics(corpus=corpus)

	# get the extracted clusters of words
	topics = gtcacs_obj.get_topics_words()
	for i, topic in enumerate(topics):
	    print(">>> TOPIC", i + 1, topic)

	# get the topics distribution scores for each document
	corpus_transf = gtcacs_obj.get_topics_distribution_scores()
	print(corpus_transf)

|

License
################################################################################################################

`MIT <https://choosealicense.com/licenses/mit/>`_
