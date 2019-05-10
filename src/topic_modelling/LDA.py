"""
Latent Dirichlet Allocation (LDA) is generative approach in classifying texts. It is a
three level hierarchical Bayesian model where it creates probabilities on word level, on
document level and on corpus level (corpus means all documents). The objective function
looks pretty complex, but it's just a way of describing the probability of a corpus
"""

def document_to_lda_features(lda_model, document):
