import nltk
import numpy as np
import pandas as pd
from collections import Counter

def filter_and_lemmatize_df(df, col_name='text', stopwords='nltk'):
    if stopwords == 'nltk':
        stopwords = nltk.corpus.stopwords.words('english')
    df[col_name] = df[col_name].str.lower()

    # Remove punctuation
    df[col_name] = df[col_name].str.replace('[^\w\s]|_','')

    # Remove numbers
    df[col_name] = df[col_name].str.replace('\d+', '')
    
    # Remove stopwords
    df[col_name] = df[col_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    # Lemmatize text column

    lemmatizer = nltk.stem.WordNetLemmatizer()
    df[col_name] = df[col_name].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    return df


