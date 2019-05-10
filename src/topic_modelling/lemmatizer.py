from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer



op_wrds = []
lemm = WordNetLemmatizer()
'''
Extend the sklearn implementation into the Vectorizer.
this tokenize input raw texts while discarding all single chracter terms('a', 'w')
Also built-in english stopwords is used to filtering out stopwords.
'''
class LemmaCountVectorizer(CountVectorizer):
    '''
    Inherited and subclassed the original Sklearn's CountVectorizer class and overwritten
    the build_analyzer method by implementing the lemmatizer for each list in the raw text matrix
    '''
    def build_analyzer(self):
        pipeline = super(LemmaCountVectorizer,self).build_analyzer()
        return lambda doc: (lemm.lematize(w) for w in pipeline(doc))



def print_top_words(train):
    # Storing the entire training text in a list
    text = list(train.text.values)
    # Calling our overwritten Count vectorizer
    tf_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                         min_df=2,
                                         stop_words='english',
                                         decode_error='ignore')
    tf = tf_vectorizer.fit_transform(text)
