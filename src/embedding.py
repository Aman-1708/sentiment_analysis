import pandas as pd
from sentence_transformers import SentenceTransformer, util


class Embedding:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def encode(self, bert=True):
        if bert:
            print("Encoding using BERT..")
            bert = SentenceTransformer('all-mpnet-base-v2')

            print('Encoding Train data..')
            self.X_train = bert.encode(self.X_train, show_progress_bar=True)
            print('Complete!..')

            print('Encoding Test data..')
            self.X_test = bert.encode(self.X_test, show_progress_bar=True)
            print('Complete!..')

        else:
            print('This class only supports BERT embeddings for now. More updates to come soon.')
