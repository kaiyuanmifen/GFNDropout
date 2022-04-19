
import torch
import numpy as np

import torch.nn as nn

import torch.nn.functional as F

class FlowFunction(nn.Module):
 
    def __init__(self, state_dim,n_action,condition_dim):
        super().__init__()

        self.embedding_dim = 16
        self.hidden_dim=128

        self.state_dim = state_dim
        self.n_action=n_action

        self.condition_dim=condition_dim
        
        self.embed = nn.Embedding(self.n_action, self.embedding_dim)

        self.FFL1= nn.Linear(self.embedding_dim+state_dim+condition_dim, self.hidden_dim)
        self.FFL2= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.outputlayer= nn.Linear(self.hidden_dim, 1)


    def forward(self, state,action,conditions):       
        '''
        state has shape (bsz,state_dim)
        action is a one hot vector has shape (bsz,n_action_space)
        conditions has shape (bsz, condition_dim)
        This function output Log(flow) for numeric issues
        '''

        emebeded_actions=self.embed_code(torch.argmax(action,1))
        
        # print("state")
        # print(state.shape)
        # print(emebeded_actions.shape)
        x=torch.cat([state,emebeded_actions,conditions],1)


        x=nn.LeakyReLU()(self.FFL1(x))
        x=nn.LeakyReLU()(self.FFL2(x))
        output=self.outputlayer(x)


        return output


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)




class DBModel(nn.Module):
 
    def __init__(self, state_dim,n_action,condition_dim):
        super().__init__()
        '''
        detailed balance model takes in state(binary mask),condition
        output P_F (size of mask) and P_B(size of mask) and flow of s (size 1)
        '''

        
        self.state_dim = state_dim
        self.n_action=n_action

        self.condition_dim=condition_dim

        self.hidden_dim=128
        
        
        self.FFL1= nn.Linear(state_dim+condition_dim, self.hidden_dim)
        self.FFL2= nn.Linear(self.hidden_dim, self.hidden_dim)
        self.outputlayer= nn.Linear(self.hidden_dim, 2*self.n_action+2)#P_F(n_action) + terminate(1) +P_B(n_action) + Flow(1)


    def forward(self, state,conditions):       
        '''
        state has shape (bsz,state_dim)
        conditions has shape (bsz, condition_dim)
        '''

        x=torch.cat([state,conditions],1)

        x=nn.LeakyReLU()(self.FFL1(x))
        x=nn.LeakyReLU()(self.FFL2(x))
        output=self.outputlayer(x)

        return output


if __name__ == "__main__":
    Fnet=FlowFunction(state_dim=3, n_embed=8)

    states=torch.zeros((2,3))
    actions=torch.tensor([[0,1,0,0,0,0],[0,0,0,0,1,0]])

    output=Fnet.forward(states,actions)

    print("output")
    print(output)