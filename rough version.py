import torch
from torch import nn
import numpy as np
# embedding = nn.Embedding(1000,128)
# embedding(torch.LongTensor([3,4]))
l = nn.Linear(4,4)
b = nn.BatchNorm1d(num_features = 4)
v = l.forward(torch.rand([2,4]))
r = nn.ReLU()
# v = r(v)
# t = [-0.12679759,  0.01196998,  0.178731 ,   0.1574521,  -0.14898884, -0.36681685, -0.07811235,  0.36594403]
# print(v.detach().numpy())
# r = b(np.array(t))
r = b(v)

#%%
dtype = torch.FloatTensor
class encoder_rnn(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim):
        super(encoder_rnn,self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,bidirectional = True,non_linearity = "Relu")
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.fc  = nn.Sequential(
                  nn.Linear(hidden_dim*2, hidden_dim*4),
                   nn.BatchNorm1d(num_features = (hidden_dim*4)),
                  nn.ReLU(),
                  nn.Linear(hidden_dim*4,output_dim),
                  nn.Sigmoid()
                  )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features = hidden_dim*2)

    def forward(self,x):
        # print(len(x))
        batch_output = torch.zeros(1,self.hidden_dim*2)
        for each in range(len(x)):
            row = x[each]
            for e in range(len(row)):
                X =  (torch.from_numpy(row[e])[None,None,:]).type(dtype)   
                if e == 0:
    #                 h0 = (torch.randn(self.layer_dim*2, 1, self.hidden_dim).type(dtype),torch.randn(self.layer_dim*2, 1, self.hidden_dim).type(dtype))
                    h0 = torch.randn(self.layer_dim*2, 1, self.hidden_dim).type(dtype)
                    output,hidden = self.rnn(X,h0)
                else:
                    output,hidden = self.rnn(X,hidden)
            output = (output).view(-1)
            output = output[None,:]
            batch_output = torch.cat((batch_output,output),0)
            
        batch_output = self.relu(self.bn1(batch_output[1:]))
        final = self.fc(batch_output)
        
        return final


    
model = encoder_rnn(300,20,2,1)
ans = model.forward(np.random.rand(10,3,300))
print(ans.shape)


#%%
import pandas as pd
d = pd.read_csv("train.csv",index_col = None,encoding = "iso-8859-1")

l = [len(each.split(" ")) for each in d["question_text"]]
y = d.target
