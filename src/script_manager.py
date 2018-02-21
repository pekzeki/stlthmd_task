from features import build_features
from models import train_model
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import os

def main():

    # Read final data set
    df = pd.read_csv(os.environ.get('FINAL_DATASET_PATH'),
                     sep='\t',
                     lineterminator='\n',
                     header=None,
                     names=['domain', 'doc_id', 'line', 'section', 'sentence', 'annotator_1', 'annotator_2', 'annotator_3'])

    # Extract a single label from inter-annotation information
    df = build_features.set_final_label(df)
    df = build_features.extra_features(df)
    #print df.corr(method='pearson')

    # Process sentences
    df['sentence'] = df['sentence'].apply(
        lambda x: build_features.process_text(x, stopwords_file=os.environ.get('STOP_WORDS_PATH'), stemming=False, lemmetization=False))
        #lambda x: build_features.process_text(x, stopwords_file=os.environ.get('STOP_WORDS_PATH'), stemming=True, lemmetization=False))
    X_train, X_test, y_train, y_test = build_features.split_data(df)


    # Train classifiers
    #train_model.svm_unigram(X_train[:, 0], y_train, X_test[:, 3], y_test)
    #train_model.svm_bigram(X_train[:, 0], y_train, X_test[:, 3], y_test)
    #train_model.top_tfidf(X_train[:, 0])

    #train_model.naive_bayes_unigram(X_train[:, 0], y_train, X_test[:, 3], y_test)
    #train_model.svm_gridsearch(X_train[:, 0], y_train, X_test[:, 3], y_test)
    #train_model.sgd_gridsearch(X_train[:, 0], y_train, X_test[:, 3], y_test)

    #train_model.sgd_extra_features(X_train, y_train, X_test, y_test)
    train_model.word2vec(X_train[:, 0], y_train, X_test[:, 0], y_test)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()











