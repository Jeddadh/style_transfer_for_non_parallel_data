import numpy as np
from keras.preprocessing import sequence
import io

class Word2vec():
    def __init__(self, fname, nmax=100000):
        self.load_wordvec(fname, nmax)
        self.word2id = {k:v for k,v in zip(self.word2vec.keys(),range(nmax+3))}
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.embeddings = np.array(list(self.word2vec.values()))
            
    def load_wordvec(self, fname, nmax):
        self.word2vec = {}
        with io.open(fname,  encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break

        vector_shape = list(self.word2vec.values())[0].shape[0]

        self.word2vec['<_bos_>'] = np.ones(vector_shape)
        self.word2vec['<_eos_>'] = np.zeros(vector_shape)
        self.word2vec['<_unk_>'] = - np.ones(vector_shape)

        print('Loaded %s pretrained word vectors' % (len(self.word2vec)))
        
class Preprocessing():
    
    def __init__(self,w2v_path, nmax=100000, maxlen=12):
        self.w2v = Word2vec(w2v_path,nmax=nmax)
        self.maxlen = maxlen
        
    def preprocess_sentence(self, sentence):
        sentence = sentence.strip().split(" ")
        return sentence

    def sentence_to_indices(self, sentence):
        pre_sentence = self.preprocess_sentence(sentence)
        sent_indices = []

        sent_indices.append(self.w2v.word2id['<_bos_>'])

        for word in pre_sentence : 
            if word in self.w2v.word2id :
                sent_indices.append(self.w2v.word2id[word])
            else:
                sent_indices.append(self.w2v.word2id['<_unk_>'])

        sent_indices.append(self.w2v.word2id['<_eos_>'])

        return sent_indices

    def sentence_to_embeddings(self, list_of_sentences):
        value = self.w2v.word2id['<_eos_>']
        sentences_indices = [self.sentence_to_indices(sentence) for sentence in list_of_sentences]
        sentences_indices = sequence.pad_sequences(sentences_indices, maxlen = self.maxlen, padding = 'post', value = value)

        sentences_embeddings = []
        for sentence in sentences_indices :
            sentences_embeddings.append([self.w2v.embeddings[w_index] for w_index in sentence])
        return np.array(sentences_embeddings)
        
        