
import torch
import numpy as np

import torch.nn as nn

import torch.nn.functional as F

class Prob_FB(nn.Module):
 
    def __init__(self, state_dim,condition_dim,max_k):
        super().__init__()

        '''
        This function calcualte the P_F and P_B for trjactory balance
        it takes in state, condition and k
        it outputs probability of next state
        '''

        self.embedding_dim = 32
        self.hidden_dim=256

        self.state_dim = state_dim
        self.max_k=max_k+1#the initial state (all ones) included

        self.condition_dim=condition_dim
        
        self.embed = nn.Embedding(self.max_k, self.embedding_dim)


        
        self.FFL1= nn.Linear(self.embedding_dim+state_dim+condition_dim, self.hidden_dim)
        self.FFL2= nn.Linear(self.hidden_dim, self.hidden_dim//4)
        self.FFL3= nn.Linear(self.hidden_dim//4, self.hidden_dim)
        self.outputlayer= nn.Linear(self.hidden_dim, 2*state_dim)


    def forward(self, state,ks,conditions):       
        '''
        state has shape (bsz,state_dim)
        ks are indices and have shape (bsz,)
        conditions has shape (bsz, condition_dim)
        '''

        emebeded_k=self.embed_code(ks)
        
       
        x=torch.cat([state,emebeded_k,conditions],1)


        x=nn.LeakyReLU()(self.FFL1(x))
        x=nn.LeakyReLU()(self.FFL2(x))
        x=nn.LeakyReLU()(self.FFL3(x))
        output=nn.Sigmoid()(self.outputlayer(x))


        return output


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)



class Partition_logZ(nn.Module):
 
    def __init__(self, condition_dim):
        super().__init__()
        '''
        this function takes in conditions of shape (bsz, dim)
        and output partition functio Z of shape (bsz,)

        '''
        self.hidden_dim=16

        self.condition_dim=condition_dim

        self.FFL1= nn.Linear(condition_dim, self.hidden_dim)
        self.FFL2= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.outputlayer= nn.Linear(self.hidden_dim, 1)


    def forward(self, conditions):       
    
        x=nn.LeakyReLU()(self.FFL1(conditions))
        x=nn.LeakyReLU()(self.FFL2(x))
        output=self.outputlayer(x)

        return output.squeeze(1)

if __name__ == "__main__":
    P_FB=Prob_FB(state_dim=4,condition_dim=5,max_k=7)
    F_logZ=Partition_logZ(5)

    N_units=2
    states=torch.randn((12,4))
    ks=torch.ones(12).long()
    conditions=torch.randn((12,5))

    predicted=P_FB(states,ks,conditions)

    Predicted_F=predicted[:,:N_units]

    Predicted_B=predicted[:,N_units:] 

    LogZ=F_logZ(conditions)

    print("Predicted_F")
    print(Predicted_F)

    print("Predicted_B")
    print(Predicted_B)

    print("LogZ")
    print(LogZ)