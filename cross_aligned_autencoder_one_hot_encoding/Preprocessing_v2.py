import numpy as np
from keras.preprocessing import sequence
from gensim.corpora import Dictionary
import torch


train_positive_path = "data/sentiment.train.1.txt"
train_negative_path = "data/sentiment.train.0.txt"

text_file_path_list = [train_positive_path, train_negative_path] 


class Preprocessing():
    
    def __init__(self, text_file_path_list = text_file_path_list, nmax=10000, maxlen=12):
        
        self.maxlen   = maxlen
        self.nmax = nmax
        self.embedded_corpuses = self.embed_corpuses()
        self.depth    = len(self.dct)

    def embed_corpuses(self):
        corpuses = self.load_data(text_file_path_list)
        self.dct = self.construct_dict(corpuses, prune_at= self.nmax)
        embedded_corpuses = []
        for i, corpus in enumerate(corpuses):
            embedded_corpus = self.sentences_to_indices(corpus)
            embedded_corpuses.append(embedded_corpus)
            print( 'Corpus {} out of {} pre-processed.'.format(i+1, len(corpuses)))
        return embedded_corpuses

    def load_data(self, text_file_path_list = text_file_path_list):
        corpuses = []
        for j, path in enumerate(text_file_path_list):
            with open(path) as f :
                corpus = []
                for i, line in enumerate(f):
                    corpus.append(line[:-2].strip().split())
            corpuses.append(corpus)
            print( 'Corpus {} out of {} loaded.'.format(j+1, len(text_file_path_list)))
        return corpuses

    def construct_dict(self, corpuses, prune_at):
        corpus = [["<_bos_>", "<_eos_>","<_unk_>" ]] + corpuses[0] + corpuses[1]
        dct = Dictionary(corpus, prune_at = prune_at)
        return dct

    def sentence_to_indices(self, sentence, unknown_word_index):
        sentence = ['<_bos_>'] + sentence + ['<_eos_>']
        sent_indices = self.dct.doc2idx(sentence,unknown_word_index = unknown_word_index)
        return sent_indices

    def sentences_to_indices(self, list_of_sentences):
        unknown_word_index = self.dct.doc2idx(['<_unk_>'])[0]
        value = self.dct.doc2idx(['<_eos_>'])[0]
        sentences_indices = [self.sentence_to_indices(sentence, unknown_word_index) for sentence in list_of_sentences]
        sentences_indices = sequence.pad_sequences(sentences_indices, maxlen = self.maxlen, padding = 'post', truncating='post', value = value)
        return sentences_indices
    
    def sparse_representation(self, x, dense = False):
        dim_1  = x.shape[0]
        dim_2  = x.shape[1]
        numel  = dim_1 * dim_2

        a      = torch.arange(dim_1, dtype = torch.long).view(-1,1).repeat(1,dim_2)
        x_idx  = a.view(a.numel()).numpy()

        y_idx  = torch.arange(dim_2, dtype = torch.long).repeat(dim_1)
        z_idx  = x.reshape(numel)

        idx    = torch.LongTensor([x_idx, y_idx, z_idx])
        values = torch.ones(numel, dtype = torch.long)

        if dense:
            values = torch.ones(numel, dtype = torch.double)
            return torch.sparse.DoubleTensor(idx, values, torch.Size([dim_1, dim_2, self.depth])).to_dense()
        else:
            return torch.sparse.LongTensor(idx, values, torch.Size([dim_1, dim_2, self.depth]))

    def sample_mini_batch(self, positive, k):
        i = 1 - int(positive)
        embedded_corpus = self.embedded_corpuses[i]
        X_batch_indices = np.random.choice(embedded_corpus.shape[0], k, replace = False)
        X_batch_indices = torch.from_numpy(X_batch_indices)
        X_batch_indices = X_batch_indices.type(torch.LongTensor)
        return self.sparse_representation(embedded_corpus[X_batch_indices],dense = True),embedded_corpus[X_batch_indices]
    

# x = Preprocessing()
# y = x.sample_mini_batch(True, 128)
