from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import gensim


def naive_bayes_unigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "Naive Bayes (Unigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])
    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def svm_unigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Unigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)),
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def svm_bigram(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Bigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None))
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def svm_gridsearch(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (GridSearchCV)"

    text_clf = Pipeline([('vect', CountVectorizer(token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               random_state=42,
                                               max_iter=5, tol=None))
                         ])

    params = dict(vect__ngram_range=[(1, 1), (1, 2)],
                  tfidf__use_idf=(True, False),
                  clf__alpha=(1e-2, 1e-3))

    grid_search = GridSearchCV(text_clf, param_grid=params)
    grid_search.fit(X_train, y_train)

    print 'Best score: ', grid_search.best_score_
    print 'Best params: ', grid_search.best_params_
    test_results(grid_search, X_test, y_test)


def svm_word2vec(X_train, y_train, X_test, y_test):
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
get_text_data = FunctionTransformer(lambda x: x[:, 3], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[:, :3].astype('float'), validate=False)


def svm_extra_features(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Handcrafted Features)"

    process_and_join_features = Pipeline([
        ('features', FeatureUnion([
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vect', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                ('tfidf', TfidfTransformer())
            ])),
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                #('ave', AverageWordLengthExtractor()),
                #('caster', ArrayCaster())
            ])),
            ('extra_features', Pipeline([
                ('selector', get_text_data),
                ('ave', AverageWordLengthExtractor()),
                ('caster', ArrayCaster())
            ]))
        ])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)) # classifier)
    ])

    process_and_join_features = process_and_join_features.fit(X_train, y_train)
    test_results(process_and_join_features, X_test, y_test)


def word2vec(X_train, y_train, X_test, y_test):
    # Load Google's pre-trained Word2Vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '/Users/arge/Downloads/stlthmd_task/data/raw/GoogleNews-vectors-negative300.bin', binary=True)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    etree_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    etree_w2v = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    text_clf = etree_w2v.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def test_results(clf, X_test, y_test):
    predictions = clf.predict(X_test)

    print "Accuracy is: ", clf.score(X_test, y_test)
    print classification_report(y_test, predictions)
    print "--------------------------"


class ArrayCaster(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return np.transpose(np.matrix(data))


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, sentence):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in sentence.split()])

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        func = np.vectorize(self.average_word_length)
        return func(X)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class SentenceLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, sentence):
        """Helper code to compute average word length of a name"""
        return len(sentence.split())

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['sentence'].apply(self.average_word_length)

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
