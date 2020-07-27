from sklearn.datasets import fetch_20newsgroups

from gtcacs.GTCACS import GTCACS

if __name__ == '__main__':
    corpus, labels = fetch_20newsgroups(subset='all', return_X_y=True, download_if_missing=False)
    eng_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
        'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
        "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }
    print(len(set(labels)))
    gtcacs_obj = GTCACS(num_topics=20,
                        max_num_words=50,
                        max_df=0.95,
                        min_df=15,
                        stopwords=eng_stopwords,
                        ngram_range=(1, 2),
                        max_features=None,
                        lowercase=True,
                        num_epoches=5,
                        batch_size=128,
                        gen_learning_rate=0.005,
                        discr_learning_rate=0.005,
                        random_seed_size=100,
                        generator_hidden_dim=512,
                        document_dim=None,
                        latent_space_dim=64,
                        discriminator_hidden_dim=256)

    gtcacs_obj.extract_topics(corpus=corpus)

    print(gtcacs_obj.corpus_transformed_shape)

    print("\n\n Topics words")
    topics = gtcacs_obj.get_topics_words()
    for i, topic in enumerate(topics):
        print(">>> TOPIC", i + 1, topic)

    print("\n\n Topics distribution scores")
    corpus_transf = gtcacs_obj.get_topics_distribution_scores()
    print(corpus_transf)
