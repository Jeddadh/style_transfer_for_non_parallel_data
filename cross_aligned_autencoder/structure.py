import torch
import torch.nn as nn
import torch.nn.functional as F



def transform_y_z_to_hidden(z,y, axis = 1):
    return torch.cat([y,z], dim = axis)

class SimpleGenerator(nn.Module):
    def __init__(self, x_dimension, y_dimension, z_dimension,emb_dimension):
        super(SimpleGenerator, self).__init__()
        self.y_dimension = y_dimension
        self.emb_dimension = emb_dimension
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        self.h_dimension = self.y_dimension + self.z_dimension
        
        
        self.gru = nn.GRUCell(input_size = self.emb_dimension,
                         hidden_size = self.h_dimension
                             )
        self.fc1 = nn.Linear(self.h_dimension,self.emb_dimension)
        
    def step_teacher_forced(self, z, y, x):
        h0 = transform_y_z_to_hidden(z,y)
        outputs = []
        hidden_states = []
        for i in range(self.x_dimension) :
            xi = x[:,i]
            if i == 0 :
                h = h0
            h = self.gru(xi,h)
            output = self.fc1(h)
            outputs.append(output)
            hidden_states.append(h)
        
        
        outputs = torch.cat(outputs, dim = 1).reshape(z.shape[0],self.x_dimension, self.emb_dimension )
        hidden_states = torch.cat(hidden_states, dim = 1).reshape(z.shape[0],self.x_dimension, self.h_dimension )
        
        return outputs, hidden_states
            
    def step_self_fed(self, z, y,temp):
        h0 = transform_y_z_to_hidden(z,y)
        self.bos = torch.ones((z.shape[0], self.emb_dimension))
        outputs = []
        hidden_states = []
        for i in range(self.x_dimension) :
            if i == 0 :
                h = h0
                xi = self.bos
            h = self.gru(xi,h)
            xi = self.fc1(h)/temp
            outputs.append(xi)
            hidden_states.append(h)
            
        outputs = torch.cat(outputs, dim = 1).reshape(z.shape[0],self.x_dimension, self.emb_dimension )
        hidden_states = torch.cat(hidden_states, dim = 1).reshape(z.shape[0],self.x_dimension, self.h_dimension )
            
        return outputs, hidden_states
       
    def forward(self, y, z, x = None, force=True,temp = 1):
        if force :
            return self.step_teacher_forced(z, y, x)
        else :
            return self.step_self_fed(z, y, temp)



def initialize_h(y, z_dimension , dim = 1):
    z0 = torch.zeros((y.shape[0],z_dimension), dtype = torch.float64)
    return torch.cat([y,z0], dim = dim)

class SimpleEncoder(nn.Module):
    def __init__(self, x_dimension, y_dimension, z_dimension, emb_dimension):
        super(SimpleEncoder, self).__init__()
        self.y_dimension = y_dimension
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        self.emb_dimension = emb_dimension
        self.h_dimension = self.y_dimension #+ self.z_dimension
        
#         self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.gru = nn.GRUCell(input_size = emb_dimension,
                         hidden_size = self.h_dimension
                             )
        self.fc1 = nn.Linear(self.h_dimension,self.x_dimension)
        
        
    def step(self,x, y):
        h0 = y #initialize_h(y, self.z_dimension)
        for i in range(self.x_dimension) :
            xi = x[:,i]
            if i == 0 :
                h = h0
            h = self.gru(xi,h)
            output = self.fc1(h)
        return output, h
                   
    def forward(self,x, y):
        return self.step(x, y)

# plusieurs couches
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,hidden_states):
        hidden1 = self.fc1(hidden_states)
        hidden2 = self.fc2(hidden1)
        return F.sigmoid(hidden2)