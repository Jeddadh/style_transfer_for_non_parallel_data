import torch
from sentence_preprocessing import   Preprocessing
import numpy as np


train_positive_path = "data/sentiment.train.1.txt"
train_negative_path = "data/sentiment.train.0.txt"

text_file_path_list = [train_positive_path, train_negative_path] 


def preprocessing(text_file_path_list = text_file_path_list):
    corpuses = []
    for j, path in enumerate(text_file_path_list):
        with open(path) as f :
            corpus = []
            for i, line in enumerate(f):
                corpus.append(line[:-2].strip().split())
        corpuses.append(corpus)
        print( 'Corpus {} out of {} loaded.'.format(j+1, len(text_file_path_list)))

    corpus = corpuses[0] + corpuses[1]
    pre_processing = Preprocessing(corpus)

    embedded_corpuses = []
    for i, corpus in enumerate(corpuses):
        # embedded_corpus = torch.tensor(pre_processing.sentence_to_embeddings(corpus), dtype=torch.long)
        embedded_corpus = pre_processing.sentences_to_indices(corpus)
        embedded_corpuses.append(embedded_corpus)
        print( 'Corpus {} out of {} pre-processed.'.format(i+1, len(text_file_path_list)))
    return embedded_corpuses

def sample_mini_batch(X, k):
    X_batch_indices = np.random.choice(len(X), k, replace = False)
    X_batch_indices = torch.from_numpy(X_batch_indices)
    X_batch_indices = X_batch_indices.type(torch.LongTensor)

    return X[X_batch_indices]
