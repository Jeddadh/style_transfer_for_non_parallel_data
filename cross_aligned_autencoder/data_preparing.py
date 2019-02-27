import torch
from sentence_preprocessing import  Word2vec, Preprocessing
import numpy as np


train_positive_path = "data/sentiment.train.1.txt"
train_negative_path = "data/sentiment.train.0.txt"

text_file_path_list = [train_positive_path, train_negative_path] 
w2v_path = '/Users/rarecarat/Downloads/nlp_project 2/data/crawl-300d-2M.vec'


def preprocessing(text_file_path_list = text_file_path_list, w2v_path = w2v_path):

    corpuses = []
    for j, path in enumerate(text_file_path_list):
        with open(path) as f :
            corpus = []
            for i, line in enumerate(f):
                corpus.append(line[:-2])
        corpuses.append(corpus)
        print( 'Corpus {} out of {} loaded.'.format(j+1, len(text_file_path_list)))

    pre_processing = Preprocessing(w2v_path)

    embedded_corpuses = []
    for i, corpus in enumerate(corpuses):
        embedded_corpus = torch.tensor(pre_processing.sentence_to_embeddings(corpus))
        embedded_corpuses.append(embedded_corpus)
        print( 'Corpus {} out of {} pre-processed.'.format(i+1, len(text_file_path_list)))
    return embedded_corpuses

def sample_mini_batch(X, k):
    X_batch_indices = np.random.choice(len(X), k, replace = False)
    X_batch_indices = torch.from_numpy(X_batch_indices)
    X_batch_indices = X_batch_indices.type(torch.LongTensor)

    return X[X_batch_indices]


# with open(train_positive_path) as f :
#     train_positive_sentences = []
#     for i, line in enumerate(f):
#         train_positive_sentences.append(line[:-2])

# print('Positive Data loaded ...')
# with open(train_negative_path) as f :
#     train_negative_sentences = []
#     for i, line in enumerate(f):
#         train_negative_sentences.append(line[:-2])

# print('Negative Data loaded ...')

# preprocessing = Preprocessing(w2v_path)

# embedded_positive_train = torch.tensor(preprocessing.sentence_to_embeddings(train_positive_sentences))
# embedded_negative_train = torch.tensor(preprocessing.sentence_to_embeddings(train_negative_sentences))

# print('Preprocessing complete')
