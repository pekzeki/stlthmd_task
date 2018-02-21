import string
from collections import defaultdict

import gensim
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier


def top_tfidf(X_train):

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
    tfidf_matrix = tf.fit_transform(X_train)
    feature_names = tf.get_feature_names()
    doc = 0
    feature_index = tfidf_matrix[doc, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        print w, '         ', s


def naive_bayes_unigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "Naive Bayes (BernoulliNB + Unigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BernoulliNB()),
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def svm_gridsearch(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "LinearSVC (GridSearchCV + Balanced)"
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC(class_weight='balanced')),
                         ])

    params = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                  tfidf__use_idf=(True, False),
                  clf__C=(0.01, 0.1, 1, 10))

    grid_search = GridSearchCV(text_clf, param_grid=params)
    grid_search.fit(X_train, y_train)

    print 'Best score: ', grid_search.best_score_
    print 'Best params: ', grid_search.best_params_

    test_results(grid_search, X_test, y_test)


def sgd_gridsearch(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SGD (GridSearchCV + Balanced)"

    text_clf = Pipeline([('vect', CountVectorizer(token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               random_state=42,
                                               max_iter=5, tol=None, class_weight='balanced'))
                         ])

    params = dict(vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                  tfidf__use_idf=(True, False),
                  clf__alpha=(1e-2, 1e-3))

    grid_search = GridSearchCV(text_clf, param_grid=params)
    grid_search.fit(X_train, y_train)

    print 'Best score: ', grid_search.best_score_
    print 'Best params: ', grid_search.best_params_

    test_results(grid_search, X_test, y_test)


def sgd_word2vec(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Handcrafted Features)"
    estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    pipe = Pipeline(estimators)
    params = dict(reduce_dim__n_components=[2, 3],
                  clf__C=[0.1, 10, 100])

    grid_search = GridSearchCV(pipe, param_grid=params)
    grid_search.fit(X_train, y_train)
    test_results(grid_search, X_test, y_test)

    print grid_search.best_score_


# get_text_data = FunctionTransformer(lambda x: x['sentence'].values, validate=False)
# get_numeric_data = FunctionTransformer(lambda x: x[['domain', 'section', 'line']].values, validate=False)
get_text_data = FunctionTransformer(lambda x: x[:, 0], validate=False)
domain = FunctionTransformer(lambda x: x[:, 1:2].astype('float64'), validate=False)
section = FunctionTransformer(lambda x: x[:, 2:3].astype('float64'), validate=False)
line = FunctionTransformer(lambda x: x[:, 3:4].astype('float64'), validate=False)
word_length = FunctionTransformer(lambda x: x[:, 4:5].astype('float64'), validate=False)
has_citation = FunctionTransformer(lambda x: x[:, 5:6].astype('float64'), validate=False)
has_symbol = FunctionTransformer(lambda x: x[:, 6:7].astype('float64'), validate=False)
has_number = FunctionTransformer(lambda x: x[:, 7:8].astype('float64'), validate=False)


def sgd_extra_features(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SGD (Handcrafted Features)"

    estimators = [('reduce_dim', TruncatedSVD()), ('clf', LinearSVC(class_weight='balanced'))]
    estimators_pipe = Pipeline(estimators)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('ngram_tf_idf', Pipeline([
                ('selector', get_text_data),
                ('vect', CountVectorizer()),
                ('transformer', TfidfTransformer())
            ])),
            ('domain', domain),
            ('section', section),
            ('line', line),
            ('word_length', word_length),
            ('has_citation', has_citation),
            ('has_symbol', has_symbol),
            ('has_number', has_number)
        ])),
        ('estimators', estimators_pipe)
    ])

    params = dict(features__ngram_tf_idf__vect__ngram_range=[(1, 1), (1, 2), (1, 3)],
                  features__ngram_tf_idf__transformer__use_idf=(True, False),
                  estimators__clf__C=(0.1, 1, 10, 100),
                  estimators__reduce_dim__n_components=[100, 110])

    grid_search = GridSearchCV(pipeline, param_grid=params)
    grid_search.fit(X_train, y_train)

    print 'Best score: ', grid_search.best_score_
    print 'Best params: ', grid_search.best_params_

    test_results(grid_search, X_test, y_test)


def word2vec(X_train, y_train, X_test, y_test):
    # Load Google's pre-trained Word2Vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/Users/arge/Downloads/stlthmd_task/data/raw/GoogleNews-vectors-negative300.bin', binary=True)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    #etree_w2v = Pipeline([
    #    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    #    ("extra trees", ExtraTreesClassifier(class_weight='balanced', n_estimators=200))])

    etree_w2v = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(class_weight='balanced', n_estimators=200))])

    text_clf = etree_w2v.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def test_results(clf, X_test, y_test):
    predictions = clf.predict(X_test)

    print "Accuracy is: ", clf.score(X_test, y_test)
    print classification_report(y_test, predictions)
    print "--------------------------"



class SentenceLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, sentence):
        """Helper code to compute average word length of a name"""
        sentence.translate(None, string.punctuation)
        words = sentence.split()
        return len(words)

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        df['len'] = df['sentence'].apply(self.average_word_length)
        return df['sentence'].apply(self.average_word_length)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class HasEntity(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self, entity):
        pass

    def has_entity(self, line, entity):
        if entity in line:
            return 1
        else:
            return 0

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['sentence'].apply(self.has_entity)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


'''
    text_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                          ('tfidf', TfidfTransformer()),
                          ('only_sentence', FunctionTransformer(select_cols, validate=False, kw_args={'cols': 3}))
                          ])

    final_pipe = Pipeline([
        ('feats', FeatureUnion([
            ('text_pipe', text_pipe),  # can pass in either a pipeline
            #('numeric_pipe', numeric_pipe)  # or a transformer
        ])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None))  # classifier
    ])
'''


def sgd_unigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Unigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None, class_weight='balanced')),
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)

def sgd_bigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SGD (Bigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None, class_weight='balanced'))
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)