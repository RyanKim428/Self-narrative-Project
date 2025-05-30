import pickle
import little_mallet_wrapper as lmw

path_to_mallet = '/Users/ryan/mallet-2.0.8/bin/mallet'

corpus = pickle.load(open('./corpus.pkl', 'rb'))

num_topics = 15
# 20, 25, 30

output = './output/'
topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet,
                                                              output,
                                                              num_topics,
                                                              corpus)
