
import torch
import numpy as np
import torch.optim as optim 
from GFN_SampleMask import GFN_SamplingMask

from FlowFunction import FlowFunction

import matplotlib.pyplot as plt
import matplotlib



matplotlib.use('AGG')


#reward function

def FakeData(batch_size,dim):

    Data=torch.randn(batch_size,dim)

    return(Data)

def get_reward(states):

    rewards=(states[:,0]==states[:,1]).sum().float()+torch.abs(torch.randn(1))

    return rewards


####parameter settings

dim=8  
N_epsiodes=200
batch_size=64#batch size better to be larger than codebook size for initalizaiton

N_units=6

p=0.5

GFN_operation=GFN_SamplingMask(N_units=N_units,batch_size=batch_size,p=p,Gamma=0.1)

Fnet=FlowFunction(state_dim=N_units, n_action=N_units,condition_dim=dim)


GFN_operation.reset()

optimizer_GFN = optim.Adam(Fnet.parameters(), lr=1e-3)
#optimizer_task=optim.Adam(Quantizer.parameters(), lr=1e-3)

#optimizer=optim.Adam(list(Fnet.parameters()), lr=1e-3)

Match_loss_all=[]
AllRewards=[]


###fake data with certain pattern
conditions=FakeData(batch_size,dim)

for Episode in range(N_epsiodes):
    print("episode",Episode)
    GFN_operation.reset()


    #####forward, build the trajectory


    states=GFN_operation.forward(FlowFunction=Fnet,conditions=conditions)

    rewards=get_reward(states.detach().clone())#decouple GFN training and other training
  

    
    ####calculate rewards
    AllRewards.append(rewards.mean().item())


    ##optimization
    #joint optimizer
    # optimizer.zero_grad()

    # Match_loss=GFN_operation.CalculateFlowMatchingLoss(Fnet,rewards)
    # Match_loss_all.append(Match_loss.item())

    # AllLoss=Match_loss

    # AllLoss.backward()


    # optimizer.step()

    ##separate optimizer
    optimizer_GFN.zero_grad()
    #optimizer_other.zero_grad()
    #####calculate flow match loss update the GFN 


    Match_loss=GFN_operation.CalculateFlowMatchingLoss(Fnet,rewards,conditions)
    Match_loss_all.append(Match_loss.item())

    Match_loss.backward()

    optimizer_GFN.step()

    ####other loss 


    #optimizer_GFN.step()
    #optimizer_other.step()

    #######

#######evaluation 
AllModes=[]
for Episode in range(20):
    print("eval episode",Episode)
    GFN_operation.reset()
    GFN_operation.Train=False#switch off Gamma random action picking
    #####forward, build the trajectory
    with torch.no_grad():
        
        states=GFN_operation.forward(FlowFunction=Fnet,conditions=conditions)

        AllModes.append(states[:,:2].sum(1).unsqueeze(0))




AllModes=torch.cat(AllModes,1).flatten()
print("AllModes")
print(AllModes)
print(AllModes.shape)


plt.plot(list(range(N_epsiodes)), Match_loss_all)
plt.title("matching loss vs episode")
plt.xlabel("episode")
plt.ylabel("matching loss")
plt.savefig('matchinglossEpisode.png')

plt.clf()
    
plt.plot(list(range(N_epsiodes)), AllRewards)
plt.title("Rewards vs episode")
plt.xlabel("episode")
plt.ylabel("Rewards(mean)")
plt.savefig('RewardEpisode.png')


plt.clf()

plt.plot(list(range(N_epsiodes)), (-torch.log(torch.tensor(AllRewards))).tolist())
plt.title("TaskLoss vs episode")
plt.xlabel("episode")
plt.ylabel("TaskLoss(mean)")
plt.savefig('TaskLossEpisode.png')


plt.clf()

    
plt.hist(AllModes)
plt.title("values")
plt.xlabel("values")
plt.ylabel("Frequency")
plt.savefig('Modes.png')