import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

def decribe_before_processing(df):

    print '------------Dataset Info-----------------'

    print 'Data Shape: ', df.shape
    print 'Columnt Types', df.dtypes
    print 'Max Line: ', df["line"].max()

    print '------------Kappa Score------------------'

    print "kappa_score (1-2): ", cohen_kappa_score(df["annotator_1"], df["annotator_2"], labels=None, weights=None)
    print "kappa_score (1-3): ", cohen_kappa_score(df["annotator_1"], df["annotator_3"], labels=None, weights=None)
    print "kappa_score (2-3): ", cohen_kappa_score(df["annotator_2"], df["annotator_3"], labels=None, weights=None)

    print '------------Descriptives-----------------'

    #doc_max_line = df.groupby(['domain', 'doc_id']).agg(np.max)['line']
    #doc_max_line.hist()
    #plt.show()

    print '-----------------------------'



'''
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


