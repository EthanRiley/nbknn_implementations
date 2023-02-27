import nltk
import pandas as pd
from collections import Counter
import numpy as np

def create_reporting_df(df, text_col='text', class_col='class', class_1='0', class_2='1'):
    '''
    Creates a table reporting the following information:
    word: The word
    count: The number of times the word appears in the corpus
    class_1_count: The number of times the word appears in the corpus for class 1
    class_2_count: The number of times the word appears in the corpus for class 2
    phi: The phi coefficient for the word
    '''
    reporting_df = pd.DataFrame(columns=['word', 'count', f'{class_1}_count', f'{class_2}_count', 'phi'])
    # Create a list of all unique words in the corpus
    vocab = get_word_set(df, text_col)

    # Find the word count for each word in the corpus
    word_count = get_word_count(df, text_col=text_col, vocab=vocab)

    # Find word count for each word in each class
    class_1_word_count = get_word_count(df[df[class_col] == class_1], text_col=text_col, vocab=vocab)
    class_2_word_count = get_word_count(df[df[class_col] == class_2], text_col=text_col, vocab=vocab)

    # Create the reporting dataframe
    for word in vocab:
        reporting_df = reporting_df.append({'word': word, 
                                            'count': word_count[word], 
                                            f'{class_1}_count': class_1_word_count[word], 
                                            f'{class_2}_count': class_2_word_count[word], 
                                            'phi': get_phi_coefficient(word_count[word], class_1_word_count[word], class_2_word_count[word])}, 
                                            ignore_index=True)
        
    return reporting_df



def get_word_set(df, text_col='text'):
    '''
    :input df: a dataframe with a column containing text
    :input col_name: the name of the column containing text
    Returns a list of all words in the corpus
    '''
    words = [word for text in df[text_col] for word in text.split()]
    return sorted(list(set(words)))

def get_word_count(df, text_col='text', vocab=None):
    '''
    Returns a dictionary of word counts for each word in the corpus
    input df: a dataframe with a column containing text
    input col_name: the name of the column containing text
    input vocab: a list of words to count
    '''
    if vocab is None:
        vocab = get_word_set(df, text_col)
    word_count = {}
    for word in vocab:
        word_count[word] = 0
    for text in df[text_col]:
        for word in text.split():
            word_count[word] += 1
    return word_count

def get_phi_coefficient(word_count, class_1_word_count, class_2_word_count):
    '''
    Returns the phi coefficient for a given word
    input word_count: the total number of times the word appears in the corpus
    input class_1_word_count: the number of times the word appears in the corpus for class 1
    input class_2_word_count: the number of times the word appears in the corpus for class 2
    '''
    # Calculate phi coefficient for each word
    phi = (word_count * (class_1_word_count - class_2_word_count)) / ((word_count + class_1_word_count) * (word_count + class_2_word_count))
    return phi
    








