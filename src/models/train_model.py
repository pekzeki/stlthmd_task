from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


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
    print "SVM (Unigram + TFIDF)"
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42,
                                               max_iter=5, tol=None)),
                         ])

    text_clf = text_clf.fit(X_train, y_train)
    test_results(text_clf, X_test, y_test)


def test_results(clf, X_test, y_test):
    predictions = clf.predict(X_test)

    print "Accuracy is: ", clf.score(X_test, y_test)
    print classification_report(y_test, predictions)
    print "--------------------------"
