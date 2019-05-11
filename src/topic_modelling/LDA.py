"""
Latent Dirichlet Allocation (LDA) is generative approach in classifying texts. It is a
three level hierarchical Bayesian model where it creates probabilities on word level, on
document level and on corpus level (corpus means all documents). The objective function
looks pretty complex, but it's just a way of describing the probability of a corpus
"""
from gensim.models.ldamulticore import LdaMulticore
import numpy as np


def init(corpus, dictionary):

    num_topics = 150
    LDAmodel = LdaMulticore(corpus=corpus,
                            id2word=dictionary,
                            num_topics=num_topics,
                            workers=4,
                            chunksize=4000,
                            passes=7,
                        alpha='asymmetric')
    return LDAmodel



'''
return the proportion of topics among each document 
'''
def document_to_lda_features(lda_model, document):
    ti_rank = lda_model.get_document_topics(document, minimum_probability=0)
    ti_rank_arr = np.array(ti_rank)
    return ti_rank_arr[:, 1]

