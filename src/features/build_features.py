from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer


def calculate_label(a1, a2, a3):
    count = Counter([a1, a2, a3])
    if len(count) == 3:
        label = a3
    elif len(count) == 2 and a1 == a2:
        label = a2
    else:
        label = a3

    return label


def set_final_label(df):
    df['final_label'] = df.apply(
        lambda row: calculate_label(row['annotator_1'], row['annotator_2'], row['annotator_3']), axis=1)
    df = df.drop(columns=["annotator_1", "annotator_2", "annotator_3"])
    # label_conversion = {"final_label": {"AIMX": 1, "OWNX": 2, "CONT": 3, "BASE": 4, "MISC": 5}}
    # df.replace(label_conversion, inplace=True)

    return df


def split_data(df):

    X = df.iloc[:, 4].values
    y = df.iloc[:, 5].values

    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_stop_words(stopwords_file):
    """
    Reads the provided stopwords file

    :param stopwords_file:
    :return:
    """

    stopwords = []
    with open(stopwords_file) as input_data:
        for line in input_data:
            stopwords.append(line.strip())
    return stopwords


def process_text(text, stopwords_file=None, stemming=False, lemmetization=False):

    filtered_words = []

    entity_list = ['CITATION', 'NUMBER', 'SYMBOL']
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()

    if stopwords_file is not None:
        stopwords = get_stop_words(stopwords_file)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    for w in tokens:

        # Do not touch entities
        if any(entity in w for entity in entity_list):
            w = w.upper()
        else:
            w = w.lower()

        if not w in stopwords:
            if stemming:
                w = stemmer.stem(w)
            if lemmetization:
                w = lemmatiser.lemmatize(w)
            filtered_words.append(w)


    return " ".join(filtered_words)