from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
op_wrds = []
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def preprocess_pipeline(self):
        pipeline = super(LemmaCountVectorizer,self).preprocess_pipeline()
        return lambda doc: (lemm.lematize(w) for w in pipeline(doc))



