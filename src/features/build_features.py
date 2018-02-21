from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import re
import string
from sklearn.preprocessing import LabelEncoder



def line_word_length(line):
    line.translate(None, string.punctuation)
    words = line.split()
    # words = re.split("[\p{Punct}\s]+", line)

    return len(words)

def has_entity(line, entity):
    if entity in line:
        return 1
    else:
        return 0


def extra_features(df):

    df['has_citation'] = df['sentence'].apply(lambda row: has_entity(row, 'CITATION'))
    df['has_symbol'] = df['sentence'].apply(lambda row: has_entity(row, 'SYMBOL'))
    df['has_number'] = df['sentence'].apply(lambda row: has_entity(row, 'NUMBER'))
    df['word_length'] = df['sentence'].apply(line_word_length)

    lb_make = LabelEncoder()
    df["domain"] = lb_make.fit_transform(df["domain"])
    df["section"] = lb_make.fit_transform(df["section"])

    return df


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


def split_sentence_data(df):

    X = df.iloc[:, 4].values
    y = df.iloc[:, 5].values

    return train_test_split(X, y, test_size=0.25, random_state=42)


def split_data(df):

    X = df[['sentence', 'domain', 'section', 'line', 'word_length', 'has_citation', 'has_symbol', 'has_number']].values
    y = df['final_label'].values

    return train_test_split(X, y, test_size=0.25, random_state=42)


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

    text = ''.join([i for i in text if not i.isdigit()])
    filtered_words = []

    entity_list = ['CITATION', 'NUMBER', 'SYMBOL']
    #stemmer = SnowballStemmer("english")
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

    else:
        return text



