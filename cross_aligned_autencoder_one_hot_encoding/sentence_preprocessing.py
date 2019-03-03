import numpy as np
from keras.preprocessing import sequence
import io
from gensim.corpora import Dictionary
import torch
        
class Preprocessing():
    
    def __init__(self, corpus, nmax=10000, maxlen=12):
        self.dct = self.construct_dict(corpus, prune_at=nmax)
        self.depth = len(self.dct)
        self.maxlen = maxlen
        
    def sentence_to_indices(self, sentence, unknown_word_index):
        sentence = ['<_bos_>'] + sentence + ['<_eos_>']
        sent_indices = self.dct.doc2idx(sentence,unknown_word_index = unknown_word_index)
        return sent_indices

    def sparse_representation(self,x, dense = False):
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
            return torch.sparse.LongTensor(idx, values, torch.Size([dim_1, dim_2, self.depth])).to_dense()
        else:
            return torch.sparse.LongTensor(idx, values, torch.Size([dim_1, dim_2, self.depth]))

    def sentences_to_indices(self, list_of_sentences):
        unknown_word_index = self.dct.doc2idx(['<_unk_>'])[0]
        value = self.dct.doc2idx(['<_eos_>'])[0]
        sentences_indices = [self.sentence_to_indices(sentence, unknown_word_index) for sentence in list_of_sentences]
        sentences_indices = sequence.pad_sequences(sentences_indices, maxlen = self.maxlen, padding = 'post', truncating='post', value = value)
        return sentences_indices

    def sentence_to_embeddings(self, sentences_indices):
        sentences_indices = self.sparse_representation(sentences_indices, True)
        return sentences_indices

    def construct_dict(self, corpus, prune_at):
        vocab = [["<_bos_>", "<_eos_>","<_unk_>" ]] + corpus
        dct = Dictionary(vocab, prune_at = prune_at)
        return dct

