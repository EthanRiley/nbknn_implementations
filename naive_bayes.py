import numpy as np
import pandas as pd


class naiveBayes:
    def __init__(self):
        self.class_list = []
        self.vocab = []
        self.class_dict = {}
        self.class_word_total = {}
        self.priors = {}
        self.idf_dict = {}

    def train(self, training_df, text_col='text', class_col='class'):
        # Determine the classes
        self.class_list = training_df[class_col].unique()
        # Determine the vocabulary
        self.vocab = self.get_word_set(training_df)
        # Instantiate a class dictionary containing the vocabulary as the keys and the probabilties as the values
        for class_ in self.class_list:
            self.class_dict[class_] = {word: 0 for word in self.vocab}
        # Get the word counts for each class
        for class_ in self.class_list:
            self.class_dict[class_] = self.get_word_count(training_df[training_df[class_col] == class_], vocab=self.vocab)
        # Get the total number of words in each class
        self.class_word_total = {}
        for class_ in self.class_list:
            self.class_word_total[class_] = 0
            for text in training_df[training_df[class_col] == class_][text_col]:
                self.class_word_total[class_] += self.num_words(text)
        # Determine priors
        self.priors = {}
        for class_ in self.class_list:
            # Get total number of members of class_ found in training_df
            num_class = len(training_df[training_df[class_col] == class_])
            self.priors[class_] = num_class / len(training_df)
        
    def predict(self, test_df, text_col='text'):
        '''
        Returns an array of predictions for the test_df
        '''
        predictions = []
        # Iterate through each row in the test_df
        classification_probabilities_list = []
        for row in test_df.iterrows():
            # Create a dictionary that holds the probability of each class
            classification_probabilities = {}
            # Iterate through each class
            for class_ in self.class_list:
                classification_probabilities[class_] = self.priors[class_]
                for word in row[1][text_col].split():
                    if word in self.vocab:
                        classification_probabilities[class_] *= np.log10(1 + self.class_dict[class_][word] / self.class_word_total[class_])
            classification_probabilities_list.append(classification_probabilities)
            # Get the class with the highest probability
            predictions.append(max(classification_probabilities, key=classification_probabilities.get))
        return predictions, classification_probabilities_list
                
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
    def get_word_count(df, vocab, text_col='text'):
        '''
        :input df: a dataframe with a column containing text
        :input vocab: a list of words
        :input col_name: the name of the column containing text
        Returns a dictionary containing the word counts for each word in the vocabulary
        '''
        word_count = {word: 0 for word in vocab}
        for text in df[text_col]:
            for word in text.split():
                if word in vocab:
                    word_count[word] += 1
        return word_count
    
    @staticmethod
    def num_words(text):
        '''
        :input text: a string
        Returns the number of words in the string
        '''
        return len(text.split())
    
    @staticmethod
    def accuracy(predictions, actual):
        '''
        :input predictions: a list of predictions
        :input actual: a list of actual values
        Returns the accuracy of the predictions
        '''
        return np.sum(predictions == actual) / len(predictions)

    @staticmethod
    def sensitivity(predictions, actual):
        pass
    
    @staticmethod
    def specificity(predictions, actual):
        pass

    @staticmethod
    def precision(predictions, actual):
        pass

    @staticmethod
    def f1_score(predictions, actual):
        pass

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        # finding number of unqiue classes
        unique_classes = set(y_true) | set(y_pred)
        n_classes = len(unique_classes)

        # initialize the confusiuon matrix
        matrix = np.zeros(shape=(n_classes, n_classes), dtype=int)

        # Pair up labels and predictions
        pairs = zip(y_true, y_pred)

        # Populate the confusion matrix
        for i, j in pairs:
            matrix[i,j] += 1

        return matrix
    
    @staticmethod
    def metrics(y_true, y_pred):
        scores = {}
        C = naiveBayes.confusion_matrix(y_true, y_pred)
        scores['accuracy'] = C.diagonal().sum() / C.sum()
        TN, FP, FN, TP = C.ravel()
        scores['sensitivity'] = TP / (TP + FN)
        scores['precision'] = TP / (TP + FP)
        scores['specificity'] = TN / (TN + FP)
        scores['f1-score'] = 2 * scores['sensitivity'] * scores['precision'] / (scores['sensitivity'] + scores['precision'])
        return scores

