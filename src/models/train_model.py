from sklearn.feature_extraction.text import CountVectorizer
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
import numpy as np


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
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)),
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def svm_gridsearch(X_train, y_train, X_test, y_test):
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


# get_text_data = FunctionTransformer(lambda x: x['sentence'], validate=False)
# get_numeric_data = FunctionTransformer(lambda x: x[['domain', 'section', 'line']], validate=False)
get_text_data = FunctionTransformer(lambda x: x[:, 3], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[:, :3], validate=False)


def svm_extra_features(X_train, y_train, X_test, y_test):
    print "--------------------------"
    print "SVM (Handcrafted Features)"

    process_and_join_features = Pipeline([
        ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data)
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vect', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None))  # classifier)
    ])

    X_train = np.sparse.hstack((X_train, OtherColumn.values))

    print X_train.shape
    print X_train[:, :3]

    process_and_join_features = process_and_join_features.fit(X_train, y_train)
    test_results(process_and_join_features, X_test, y_test)



def test_results(clf, X_test, y_test):
    predictions = clf.predict(X_test)

    print "Accuracy is: ", clf.score(X_test, y_test)
    print classification_report(y_test, predictions)
    print "--------------------------"


class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def average_word_length(self, sentence):
        """Helper code to compute average word length of a name"""
        return np.mean([len(word) for word in sentence.split()])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        return df['sentence'].apply(self.average_word_length)

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