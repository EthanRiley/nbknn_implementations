import numpy as np
from math import dist, sqrt
from collections import Counter
from functools import partial

class kNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.training_array = np.array([])
        self.training_classes = np.array([])
        self.idf_dict = {}

    def train_on_df(self, df, text_col='text', class_col='class', vectorizer='frequency'):
        self.vocab = self.get_word_set(df, text_col=text_col)
        self.training_array, self.training_classes = self.vectorize_df(df, vectorizer=vectorizer, text_col=text_col, class_col=class_col)

    def train(self, training_array, training_classes):
        self.training_array = training_array
        self.training_classes = training_classes
    
    def predict(self, test_array, distance_type='euclid'):
        distance_functions = {'euclid': self.euclid_distance,
                              'cosine': self.cosine_distance,
                              'manhattan': self.manhattan_distance,
                              'minkowski': self.minkowski_distance,
                              'chebyshev': self.chebyshev_distance}
        predictions = []
        for i in range(len(test_array)):
            distances = []
            for j in range(len(self.training_array)):
                distances.append(distance_functions[distance_type](test_array[i], self.training_array[j]))
            k_nearest = np.argsort(distances)[:self.k]
            k_nearest_classes = []
            for index in k_nearest:
                k_nearest_classes.append(self.training_classes[index])
            predictions.append(Counter(k_nearest_classes).most_common(1)[0][0])
        return predictions


    def vectorize_df(self, df, vectorizer='frequency', text_col='text', class_col='class', type='train'):
        vectorize = {'frequency': self.frequency_vectorizer,
                      'count': self.count_vectorizer,
                      'tfidf': self.tfidf_vectorizer,
                      'binary': self.binary_vectorizer,
                      'log': self.log_vectorizer}

        if vectorizer == 'tfidf' and type == 'train':
            self.create_idf_dict(df, text_col=text_col)

        #print(self.idf_dict)

        text_as_vector = np.zeros((len(df), len(self.vocab)))
        for i in range(len(df)):
            split_text = df.iloc[i][text_col].split()
            text_counts = Counter(split_text)
            #print(text_counts)
            for j in range(len(self.vocab)):
                if vectorizer == 'tfidf':
                    text_as_vector[i][j] = vectorize[vectorizer](idf_dict=self.idf_dict, text_counts=text_counts, word=self.vocab[j])
                else:
                    text_as_vector[i][j] = vectorize[vectorizer](text_counts=text_counts, word=self.vocab[j])
        class_array = df[class_col].values
        return text_as_vector, class_array

    @staticmethod
    def euclid_distance(x, y):
        return dist(x, y)
    
    @staticmethod
    def cosine_distance(x, y):
        return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    
    @staticmethod
    def manhattan_distance(a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))
    
    @staticmethod
    def minkowski_distance(a, b, p):
        return sum(abs(val1-val2)**p for val1, val2 in zip(a,b))**(1/p)
    
    @staticmethod
    def chebyshev_distance(a, b):
        return max(abs(val1-val2) for val1, val2 in zip(a,b))
    

    @staticmethod
    def count_vectorizer(text_counts, word):
        return text_counts[word]

    @staticmethod
    def frequency_vectorizer(text_counts, word):
        # Sum values in text_counts Counter object
        total = sum(text_counts.values())
        return text_counts[word]/total
        

    @staticmethod
    def get_word_set(df, text_col='text'):
        '''
        :input df: a dataframe with a column containing text
        :input col_name: the name of the column containing text
        Returns a list of all words in the corpus
        '''
        words = [word for text in df[text_col] for word in text.split()]
        return sorted(list(set(words)))

    @staticmethod
    def tfidf_vectorizer(idf_dict, text_counts, word):
        return text_counts[word] * idf_dict[word]

    def create_idf_dict(self, df, text_col='text'):
        document_count = len(df)
        for word in self.vocab:
            # Find the number of documents that contain the word
            word_count = 0
            for row in df.iterrows():
                words = row[1][text_col].split()
                if word in words:
                    word_count += 1
            # Calculate the idf for the word
            self.idf_dict[word] = np.log((document_count/word_count + 1))

    def binary_vectorizer(self, df, text_col='text', class_col='class'):
        pass

    def log_vectorizer(self, text_col='text', class_col='class'):
        pass


