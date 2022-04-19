
import torch
import numpy as np
import torch.nn as nn

class GFN_SamplingMask(object):
    def __init__(self, N_units,device="cpu",batch_size=32,p=0.5,Gamma=0.1):
        '''
        in this version the DAG is not a tree (a state may have mutiple parents) 

        a state is a binary vector indicating whether a unit is active (not dropped) or inactive

        an action is to deactivate an unit so that the DAG in acyclic  
        
        '''
        self.N_units = N_units
        self.batch_size=batch_size


        self.action_indice=torch.arange(0,N_units).to(device)#which unit to deactivate

        self.p=p #percentage of decativation , to be consistant with traditional dropout method

        self.action_space=torch.nn.functional.one_hot(self.action_indice.flatten()).to(device)

        self.Gamma=Gamma#chances of randomly choose allowed actions

        self.Train=True ###switch on/ff random actions selection

        self.device=device
    def reset(self):

        
        self.action_indice=torch.arange(self.N_units).to(self.device)#which unit to deactivate
        
        self.action_space=torch.nn.functional.one_hot(self.action_indice.flatten()).to(self.device)

        self.Train==True

    def UpdateState(self,state,action):
        '''
        state has shape  (dim,)
        action are in the form of one hot vectors
        '''

        action_index=torch.argmax(action)

        state[action_index]=0# drop one unit in the dropout


        return state

    def get_flow_in(self,state,FlowFunction,condition):
        '''
        state has shape (dim,)
        get the in flow of a state, in this env all state transition are deterministic
        
        In this version with a tree structured of DAG, each state only has one parental state
        
        The order does not matter (a set), there fore the parental states can be any state with the current state + backward action
        '''
   

        if (state==0).sum()==0:
            print("initial state ,no parents")

        else:
            backward_action_indice=[]
            parent_states=[]
            
            for Position_index in range(state.shape[0]):
                if Position_index==0:        
                    parent_state=state.detach().clone()
                    parent_state[Position_index]=1
                    
                backward_action_index=Position_index

                parent_states.append(parent_state.unsqueeze(0))

                backward_action_indice.append(backward_action_index)

                
            parent_states=torch.cat(parent_states,0)
            backward_action_indice=torch.tensor(backward_action_indice)


        backward_actions=self.action_space.clone().detach()[backward_action_indice,:]

        conditions=condition.unsqueeze(0).repeat(parent_states.shape[0],1)

    
        LogFlows=FlowFunction(parent_states,backward_actions,conditions)

        return parent_states,backward_actions,LogFlows


    def get_flow_out(self,state,FlowFunction,condition):
        

        PossibleActions=self.action_space[state==1,:]

        state_vec=state.unsqueeze(0).repeat((state==1).sum(),1)

        condition=condition.unsqueeze(0).repeat((state==1).sum(),1)

       
        LogFlows=FlowFunction(state_vec,PossibleActions,condition)
       
        return state,PossibleActions,LogFlows


    def forwardFM(self,FlowFunction,conditions):
        '''
        the flow function conditions on inputs into the task model , the condition has shape (batch_sz, dim)
        '''

        ###initialize state
        self.states=torch.ones((conditions.shape[0],self.N_units)).to(self.device) #all ones,intial states of the GFN where all units are active
        
        self.trajectories=[] # trjactories may have differnet length
        

        #run the forward



        terminal=False #if the trajectories has come to an end 
        
        for step in range(0,int(self.N_units*self.p)):
            
            terminal=len(self.trajectories)==int(self.N_units*self.p)#reserved for later improvement


            if not terminal:
                states_vec=[]
                for i in range(conditions.shape[0]):#in the last batch, N_samples may be < batch size
                    state=self.states.clone().detach()[i,:]
               
                    _,PossibleActions,LogFlows=self.get_flow_out(state,FlowFunction,conditions[i,:])
                    
          
                    Flows=torch.exp(LogFlows)
          
                    action_Prob=torch.nn.Softmax(0)(Flows)
          

                    if np.random.uniform()<self.Gamma and self.Train==True:#with some chances of random actions
                        #print("using random actions")
                        action_Prob=torch.zeros_like(action_Prob).fill_(1.0/action_Prob.shape[0])

                    action_index_chosen=np.random.choice(np.arange(0, PossibleActions.shape[0]), p=action_Prob.flatten().detach().cpu().numpy())
                    
                    action=PossibleActions[action_index_chosen,:]

                    state=self.UpdateState(state,action)

                    states_vec.append(state.unsqueeze(0))
                    
                   
                self.states=torch.cat(states_vec,0)


            self.trajectories.append(self.states.detach().clone())




        return self.states


    def CalculateFlowMatchingLoss(self, FlowFunction,reward_terminal,conditions):
        '''
        first run the step_forward function to generate batch of trajectgories ending with terminal states
        then run this function to calculate flow matching loss 
        the reward_terminal is calcualted from terminate states(self.states afte running forward function), it has the shape (batch_size,)

        '''

        ####low match on all but initial state
        Epsilon=1e-8 #small number to avoid numeric problem to avoid log on tiny number, and balance large and small flow, larger number smooth out the distribution (more difficult for peaky modes)

        Match_loss_all=0
        for t in range(1,len(self.trajectories)):
            states_t=self.trajectories[t] #shape Batch_size X N_slots Xstate_dim

            for i in range(conditions.shape[0]):#loop through the batch dimension
                

                state=states_t[i,:]                

                terminal=t==int(self.N_units*self.p)   
           
                if not terminal:
                    _,_,LogFlowsOut=self.get_flow_out(state,FlowFunction,conditions[i,:])
                    
                    Reward=torch.tensor(0.0)
                else:
                    LogFlowsOut=torch.tensor(0.0)
                    Reward=torch.tensor(reward_terminal[i]).float()
                _,_,LogFlowsIn=self.get_flow_in(state,FlowFunction,conditions[i,:])
                

                

                ####hand-written version:
                # In_log_sum=torch.log(Epsilon+torch.exp(LogFlowsIn).sum())
                # Out_log_sum=torch.log(Epsilon+Reward+torch.exp(LogFlowsOut).sum())


                ###in torch logsumexp version is more numerically stable:


                In_log_sum=torch.logsumexp(torch.cat([torch.log(torch.tensor(Epsilon).reshape(1,1)).to(self.device),
                                            LogFlowsIn],0),0)
                Out_log_sum=torch.logsumexp(torch.cat([torch.log(torch.tensor(Epsilon).reshape(1,1)).to(self.device),
                                            torch.log(torch.tensor(Reward).reshape(1,1).to(self.device)),LogFlowsOut],0),0)
                
 
                Match_loss_ti=(In_log_sum-Out_log_sum)**2
 
                Match_loss_all+=Match_loss_ti

        return Match_loss_all


    def forwardDB(self,model_DB,conditions):
        '''
        in detailed balance the NN predict F, P_F and P_B , the condition has shape (batch_sz, dim)
        '''
        batch_size,dim=conditions.shape

        inf = 1000000000
        minus_inf = -inf
        logZ_DB = torch.zeros((1,)).to(self.device) # log (initial state flow), Z = 1

        # set PB to uniform distribution (and learn just PF) or not 
        uniform_PB = False

        trajectory_max_length=int(self.N_units*(1.0-self.p))      

  
        self.loss_DB = torch.zeros((batch_size, trajectory_max_length)).to(self.device)
        # finished trajectories
        dones = torch.full((batch_size,), False, dtype=torch.bool).to(self.device)
        #start with state all zeros, fill in 1 ones
        states = torch.zeros((batch_size, self.N_units), dtype=torch.long).to(self.device)
        actions = None # (current_batch_size,)

        i = 0
        while torch.any(~dones):

            ### Forward pass ### 
            current_batch_size = (~dones).sum()
            non_terminal_states = states[~dones] # (current_batch_size, ndim)
            logits = model_DB(non_terminal_states,conditions) # (current_batch_size, output_dim) 

            ### logF ### 
            logF = logits[...,2*self.N_units+1] # (current_batch_size,)
          


            self.loss_DB[~dones,i] += logF # add logF(s_i) to the loss L(s_i, s_{i+1})
            if i>0: self.loss_DB[~dones,i-1] -= logF # add -logF(s_i) to the loss L(s_{i-1}, s_i)
            #else : logZ_DB[:] = logF.mean().item() # initial state : Z = F(s_0)
            else: logZ_DB[:] = logF[0].item() # initial state : Z = F(s_0)
            
            ### Backward Policy ### 
            PB_logits = logits[...,(self.N_units+1):(2*self.N_units+1)] # (current_batch_size, ndim)
            PB_logits = PB_logits * (0 if uniform_PB else 1) # (current_batch_size, ndim)
            # Being in a edge cell -- (a zero coordinate), we can't move backward
            edge_mask = (non_terminal_states == 1).float() # (current_batch_size, ndim)
            logPB = (PB_logits + minus_inf*edge_mask).log_softmax(1) # (current_batch_size, ndim)
            # add -logPB(s_{i-1} | s_i) to the loss L(s_{i-1}, s_i)
            if actions is not None: 
                """
                Gather along the parents' dimension (1) to select the logPB of the previously chosen actions, while avoiding the actions leading 
                to terminal states (action==ndim). The reason of using the previous chosen actions () is that PB is calculated on the same trajectory as PF
                See below for the calculation of `action`. We avoid actions leading to terminal states because a terminal state can't be parent of another 
                state
                """
                #self.loss_DB[~dones,i-1] -= logPB.gather(1, actions[actions!=ndim].unsqueeze(1)).squeeze(1)
           
                self.loss_DB[~dones,i-1] -= logPB.gather(1, actions).squeeze(1)
        
            ### Forward Policy ### 
            PF_logits = logits[...,:(self.N_units)] # (current_batch_size, ndim+1) 
            # Being in a edge cell ++ (a coordinate that is H), we can't move forward
            edge_mask = (non_terminal_states == 1).float() # (current_batch_size, ndim)
            # but any cell can be a terminal cell
            stop_action_mask = torch.zeros((current_batch_size, 1), device=self.device) # (current_batch_size, 1)
            # Being in a edge cell, we can't move forward, but any cell can be a terminal cell
            #PF_mask = torch.cat([edge_mask, stop_action_mask], 1) # (current_batch_size, ndim+1)
            PF_mask = edge_mask
            # logPF (with mask)
            logPF = (PF_logits + minus_inf*PF_mask).log_softmax(1) # (current_batch_size, ndim+1)
            # choose next states
            sample_temperature = 1
            #exp_weight= 0.
            #sample_ins_probs = (1-exp_weight)*(logPF/sample_temperature).softmax(1) + exp_weight*(1-PF_mask) / (1-PF_mask+0.0000001).sum(1).unsqueeze(1) # (current_batch_size, ndim+1)
            sample_ins_probs = (logPF/sample_temperature).softmax(1) # (current_batch_size, ndim+1)
            #actions = torch.distributions.categorical.Categorical(probs = sample_ins_probs).sample() # (current_batch_size,)
            #actions = torch.multinomial(probs = sample_ins_probs, 1).squeeze(1) # (current_batch_size,) # (current_batch_size,)
            actions = sample_ins_probs.multinomial(1) # (current_batch_size,)


            # add logPF(s_i | s_{i+1}) to the loss L(s_i, s_{i+1}) : gather along the children's dimension (1) to select the logPF for the chosen actions
            self.loss_DB[~dones,i] += logPF.gather(1, actions).squeeze(1)

            # terminates = (actions==ndim).squeeze(1)
            # for state in non_terminal_states[terminates]: 
            #     state_index = get_state_index(state.cpu())
            #     if first_visit_DB[state_index]<0: first_visit_DB[state_index] = it
            #     all_visited_DB.append(state_index)

            ####take action
            with torch.no_grad():
                #non_terminates = actions[~terminates]
                #states[~dones] = states[~dones].scatter_add(1, non_terminates, torch.ones(non_terminates.shape, dtype=torch.long, device=self.device))
                #action ares deactivation of neuron units
                for j in range(states.shape[0]): 
                    states[j,actions[j]]=1

            ### select terminal states ### 

            terminates = ((states==1).sum(1)==trajectory_max_length)
            dones |= terminates
       
            i += 1

        #lens = states.sum(1)+1 # number of actions taken for each trajectory
        #loss = (loss_DB**2).sum()/lens.sum() 
        #loss = (loss_DB**2).sum(dim=1).sum()/batch_size

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        # losses_DB.append(loss.item())
        # R = reward_function(states.float())
        # rewards_DB.append(R.mean().cpu())
        # logZ_DB_list.append(logZ_DB.item())

        # if it%100==0: 
        #     print('\nloss =', np.array(losses_DB[-100:]).mean(), 'logZ =', logZ_DB.item(), "R =", np.array(rewards_DB[-100:]).mean())
            
        #     #emp_dist = np.bincount(all_visited_DB[-200000:], minlength=len(true_dist)).astype(float)
        #     #emp_dist /= emp_dist.sum()
        #     #l1 = np.abs(true_dist-emp_dist).mean()
        #     #print('L1 =', l1)
        #     #l1log_DB.append((len(all_visited_DB), l1))

        # return np.array(losses_DB[-100:]).mean(),logZ_DB.item(),np.array(rewards_DB[-100:]).mean()

        return states
          

    def DB_train(self,terminal_rewards,optimizer):
        '''
        in detailed balance the NN predict F, P_F and P_B , the condition has shape (batch_sz, dim)
        '''
        batch_size=terminal_rewards.shape[0]

        self.loss_DB[:,-1] -= (terminal_rewards + 1e-8).log()
        #lens = states.sum(1)+1 # number of actions taken for each trajectory
        #loss = (loss_DB**2).sum()/lens.sum() 
        loss = (self.loss_DB**2).sum(dim=1).sum()/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # losses_DB.append(loss.item())
        # R = reward_function(states.float())
        # rewards_DB.append(R.mean().cpu())
        # logZ_DB_list.append(logZ_DB.item())

        # if it%100==0: 
        #     print('\nloss =', np.array(losses_DB[-100:]).mean(), 'logZ =', logZ_DB.item(), "R =", np.array(rewards_DB[-100:]).mean())
            
        #     #emp_dist = np.bincount(all_visited_DB[-200000:], minlength=len(true_dist)).astype(float)
        #     #emp_dist /= emp_dist.sum()
        #     #l1 = np.abs(true_dist-emp_dist).mean()
        #     #print('L1 =', l1)
        #     #l1log_DB.append((len(all_visited_DB), l1))

        # return np.array(losses_DB[-100:]).mean(),logZ_DB.item(),np.array(rewards_DB[-100:]).mean()

        return loss.detach().clone()
          


    
    



class GFN_SamplingMask_AutoencoderTB(object):
    def __init__(self, N_units,device="cpu",batch_size=32,p=0.5,Gamma=0.1):
        '''
        This is trajecgory balance version
        in this version the DAG is not a tree (a state may have mutiple parents) 

        a state is a binary vector indicating whether a unit is active (not dropped) or inactive

        an action is to deactivate an unit so that the DAG in acyclic  
        
        '''
        self.N_units = N_units
        self.batch_size=batch_size


        self.action_indice=torch.arange(0,N_units).to(device)#which unit to deactivate

        self.p=p #percentage of decativation , to be consistant with traditional dropout method

        self.action_space=torch.nn.functional.one_hot(self.action_indice.flatten()).to(device)

        self.Gamma=Gamma#chances of randomly choose allowed actions

        self.Train=True ###switch on/ff random actions selection

        self.k=5 #total number of steps
        self.device=device
    def reset(self):

        
        self.action_indice=torch.arange(self.N_units).to(self.device)#which unit to deactivate
        
        self.action_space=torch.nn.functional.one_hot(self.action_indice.flatten()).to(self.device)

        self.Train==True

    def forward(self,P_FB,conditions):
        '''
        the flow function conditions on inputs into the task model , the condition has shape (batch_sz, dim)
        '''

        ###initialize state
        self.states=torch.ones((conditions.shape[0],self.N_units)).to(self.device) #all ones,intial states of the GFN where all units are active
        
        self.trajectories=[] # trjactories may have differnet length
        
        self.trajectories.append(self.states)#put initial state in

        #run the forward

       
        terminal=False #if the trajectories has come to an end 
        
        for step in range(0,self.k):
            
            terminal=len(self.trajectories)==self.k#reserved for later improvement


            if not terminal:

                predicted=P_FB(self.states,torch.tensor(step).repeat(self.states.shape[0]),conditions)

                Predicted_F=predicted[:,:self.N_units]

                if step==self.k-1:#limit the last step action space to only p% active
                    Threshold=torch.topk(Predicted_F,int(self.N_units*self.p))[0].min()
                    states_next=(Predicted_F>Threshold).float()

                else:        
                    states_next=(Predicted_F>torch.ones(Predicted_F.shape).uniform_(0,1)).float().detach().clone()#sample the output, think about better ways to do this

                self.states=states_next.detach().clone()


            self.trajectories.append(self.states.detach().clone())



        return self.states

    def CalculateTrajectoryLogP(self,trajectories,conditions,P_FB):
        '''
        trajectories has shape (T,bsz,dim)
        '''
        All_Log_P_F=[]
        All_Log_P_B=[]
        for step in range(1,trajectories.shape[0]):
          
            predicted_F=P_FB(trajectories[step-1,:,:],torch.tensor(step).repeat(self.states.shape[0]),conditions)

            Predicted_F=predicted_F[:,:self.N_units]
            

            Log_P_F=torch.log(trajectories[step,:,:]*(Predicted_F)+(1-trajectories[step,:,:])*(1-Predicted_F)).sum(1)
            
            predicted_B=P_FB(trajectories[step,:,:],torch.tensor(step).repeat(self.states.shape[0]),conditions)
            Predicted_B=predicted_B[:,self.N_units:]

            Log_P_B=torch.log(trajectories[step,:,:]*(Predicted_B)+(1-trajectories[step,:,:])*(1-Predicted_B)).sum(1)
            
            All_Log_P_F.append(Log_P_F.unsqueeze(0))
            All_Log_P_B.append(Log_P_B.unsqueeze(0))

        All_Log_P_F=torch.cat(All_Log_P_F,0)
        All_Log_P_B=torch.cat(All_Log_P_B,0)

        return All_Log_P_F,All_Log_P_B #shape(T,Bsz)


    def CalculateTrajectoryBalance(self,P_FB,F_logZ,reward_terminal,conditions):
        '''
        first run the step_forward function to generate batch of trajectgories ending with terminal states
        then run this function to calculate TB
        the reward_terminal is calcualted from terminate states(self.states afte running forward function), it has the shape (batch_size,)

        '''
        trajectories=torch.cat([item.unsqueeze(0) for item in self.trajectories])#shape(T,bsz,dim)

        All_Log_P_F,All_Log_P_B=self.CalculateTrajectoryLogP(trajectories,conditions,P_FB)

        # print("All_Log_P_F")
        # print(torch.exp(All_Log_P_F))
        # print("All_Log_P_B")
        # print(torch.exp(All_Log_P_B))


        Log_Z=F_logZ(conditions)#estimate the partition function 
        # print("Log_Z")
        # print(Log_Z[0])
        # print("reward_terminal")
        # print(torch.log(reward_terminal[0]))
        numerator=Log_Z+All_Log_P_F.sum(0)
      
        
        MIN_REW=1e-9#minimum reward to avoid numerical problem
        denominators=torch.log(torch.tensor(reward_terminal).clamp_min(MIN_REW))+All_Log_P_F.sum(0)
  
        loss = nn.MSELoss()(numerator, denominators)

        return loss

          



    
    