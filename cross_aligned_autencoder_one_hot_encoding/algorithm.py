import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from structure import SimpleEncoder, SimpleGenerator, Discriminator
from Preprocessing_v2 import Preprocessing

torch.set_default_tensor_type('torch.DoubleTensor')


preprocessing = Preprocessing()

# preprocessing.corpuses 

# preparing Data
# [embedded_positive_train, embedded_negative_train] = preprocessing()

# setting dimensions
x_dimension = preprocessing.maxlen
y_dimension = 100
z_dimension = 100
emb_dimension = preprocessing.depth
input_size, hidden_size = x_dimension*(y_dimension + z_dimension), 100


discriminator1 = Discriminator(input_size, hidden_size)
discriminator2 = Discriminator(input_size, hidden_size)
encoder = SimpleEncoder(x_dimension, y_dimension, z_dimension, emb_dimension)
generator = SimpleGenerator(x_dimension, y_dimension,z_dimension, emb_dimension)

optim_encoder = torch.optim.Adam(encoder.parameters(), lr = 0.001, betas=(0.5, 0.9))
optim_generator = torch.optim.Adam(generator.parameters(), lr = 0.001, betas=(0.5, 0.9))
optim_discriminator1 = torch.optim.Adam(discriminator1.parameters(), lr = 0.001, betas=(0.5, 0.9))
optim_discriminator2 = torch.optim.Adam(discriminator2.parameters(), lr = 0.001, betas=(0.5, 0.9))

N_ITER = 1000
BATCHLEN = 128
lamda = 0.1

# dimension 2
y_0 = torch.ones((BATCHLEN,y_dimension ), dtype = torch.float64)
y_1 = - torch.ones((BATCHLEN,y_dimension ), dtype = torch.float64)

y = [y_0, y_1]
Lrecs = []
LD1s = []
LD2s= []
for i in range(N_ITER):
    encoder.zero_grad()
    generator.zero_grad()
    discriminator2.zero_grad()
    discriminator1.zero_grad()
    H_p = [None, None]
    transfered_H_p = [None, None]
    
    Outputs = [None, None]

    X_batch_pos, target_pos = preprocessing.sample_mini_batch(positive=True, k=BATCHLEN)
    X_batch_neg, target_neg = preprocessing.sample_mini_batch(positive=False, k=BATCHLEN)

    target_pos = torch.tensor(target_pos).type(torch.LongTensor)
    target_neg = torch.tensor(target_neg).type(torch.LongTensor)

    X_batch = [X_batch_pos,X_batch_neg] # sample a minibatch
    
    for p in range(2):
        q = 1-p
        X_batch_p = X_batch[p]
        y_p = y[p]
        _ , Z = encoder(X_batch_p,y_p)
        Outputs[p], H_p[p]  = generator(y_p, Z, X_batch_p, force=True)
        transfered_H_p[p] = generator(y[q], Z, X_batch_p, force=False)[1]
        
        H_p[p] = H_p[p].reshape((BATCHLEN, x_dimension*(y_dimension + z_dimension)))
        transfered_H_p[p] = transfered_H_p[p].reshape((BATCHLEN, x_dimension*(y_dimension + z_dimension)))
        
    criterion = nn.NLLLoss(reduce=True, size_average=False)
    # Lrec = torch.sqrt(criterion(Outputs[0], X_batch[0])) + torch.sqrt(criterion(Outputs[1], X_batch[1])) # MSE

    X_batch[0] = torch.transpose(X_batch[0], 1, 2)
    X_batch[1] = torch.transpose(X_batch[1], 1, 2)

    Outputs[0] = torch.transpose(Outputs[0], 1, 2)
    Outputs[1] = torch.transpose(Outputs[1], 1, 2)

    Lrec = - criterion(Outputs[0], target_pos) - criterion(Outputs[1], target_neg) # Cross_entropy
    LD1 =  torch.mean(- torch.log(discriminator1(H_p[0])) - torch.log(1 - discriminator1(transfered_H_p[1])))
    LD2 =  torch.mean(- torch.log(discriminator2(H_p[1])) - torch.log(1 - discriminator2(transfered_H_p[0])))

    (Lrec - lamda*(LD1)+LD2).backward(retain_graph=True)
    optim_encoder.step()
    optim_generator.step()
    LD1.backward(retain_graph=True)
    optim_discriminator1.step()
    LD2.backward()
    optim_discriminator2.step()
    if i%10 ==0:
        print(' ------- Epoch :', i, '-------')
        print('')
        print('Lrec :', Lrec.item())
        print('LD1 :', LD1.item())
        print('LD2 :', LD2.item())
        print('')
    
    Lrecs.append(Lrec.item())
    LD1s.append(LD1.item())
    LD2s.append(LD2.item())

    if i%100 == 0 and i!= 0 :
        plt.plot(Lrecs, label ='Lrec' )
        plt.plot(LD1s , label = 'LD1')
        plt.plot(LD2s , label = 'LD2')
        plt.legend()
        plt.show()