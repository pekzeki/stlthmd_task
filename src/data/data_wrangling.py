import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from dotenv import find_dotenv, load_dotenv
import os

def show_basic_info(df):

    print '------------Dataset Info-----------------'

    print 'Data Shape: ', df.shape
    print 'Max Line: ', df["line"].max()

    print pd.isnull(df).any()

def show_kappa_score(df):

    print '------------Kappa Score------------------'

    print "kappa_score (1-2): ", cohen_kappa_score(df["annotator_1"], df["annotator_2"], labels=None, weights=None)
    print "kappa_score (1-3): ", cohen_kappa_score(df["annotator_1"], df["annotator_3"], labels=None, weights=None)
    print "kappa_score (2-3): ", cohen_kappa_score(df["annotator_2"], df["annotator_3"], labels=None, weights=None)



def show_stats(df):

    print '------------Descriptives-----------------'

    #doc_max_line = df.groupby(['domain', 'doc_id']).agg(np.max)['line']
    #doc_max_line.hist()
    #plt.show()




'''
print 'Columnt Types', df.dtypes
print df.shape
print df.iloc[0:5,:]
print df.loc[:5,"section"]
print df.loc[:5,["section", "annotator_3"]]
print df[["annotator_1", "annotator_2", "annotator_3"]]

print df["line"] / 2
print df["line"].max()
print df["line"].std()

line_filter = df["line"] > 7
filtered_lines = df[line_filter]
print filtered_lines

jdm_w_line_filter = (df["line"] > 10) & (df["domain"] == "arxiv")
filtered_rows = df[jdm_w_line_filter]
filtered_rows.head()

df = df[df["domain"] == "arxiv"]["line"]
df.hist()
plt.show()
'''


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Read final data set
    df = pd.read_csv(os.environ.get('FINAL_DATASET_PATH'),
                     sep='\t',
                     lineterminator='\n',
                     header=None,
                     names=['domain', 'doc_id', 'line', 'section', 'sentence', 'annotator_1', 'annotator_2', 'annotator_3'])

    show_basic_info(df)


