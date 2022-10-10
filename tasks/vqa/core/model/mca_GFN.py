# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)


    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        #att_map = self.dropout(att_map)
        att_map = self.GFN_Dropout(att_map,query)#in transformer, not using previous binary masks as input due to the large number,but it still implicitly conditioned on previous masks

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(0)#because GFN maks is used
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask,GFN_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask*GFN_mask)#overlap the masks
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(0)#as GFN mask is used
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask,GFN_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask*GFN_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])



        ###construct the mask for GFN
        layer_dims=[512 for p in range(__C.LAYER)]

        self.rand_mask_generator=RandomMaskGenerator(dropout_rate=opt.mlp_dr)


        self.p_zx_mask_generators=construct_conditional_mask_generators(layer_dims=layer_dims,
                                                        additional_input_dims=[0 for j in layer_dims],
                                                        hiddens=[32,32]).to(device)#p(z|x)

        self.q_zxy_mask_generators=construct_multiinput_conditional_mask_generators(layer_dims=layer_dims,
                                                        additional_input_dims=[num_classes for j in layer_dims],
                                                        hiddens=[32,32]).to(device)#q(z|x,y) 

    
        self.p_z_mask_generators=construct_unconditional_mask_generators(layer_dims=layer_dims,
                                                        hiddens=[32,32]).to(device)#p(z) use small capacity to regularize q_z

        self.q_z_mask_generators=construct_unconditional_mask_generators(layer_dims=layer_dims,
                                                        hiddens=[32,32]).to(device)#q(z)



        self.activation = activation

        mg_activation=nn.LeakyReLU
        

        ###partition functions
        self.LogZ_total_flowestimator = MLP(in_dim=input_dim+num_classes,out_dim=1,hidden=[16],
                                    activation=mg_activation).to(device)#paritization function when the GFN condition on both x and y 


        self.LogZ_unconditional=nn.Parameter(torch.tensor(0.0))#paritization function when the GFN does not condition on any input 




        ######optimizer
        z_lr=1e-1
        mg_lr_z=1e-3
        mg_lr_mu=1e-3
        lr=opt.lr
        self.beta=self.opt.beta #temperature on rewards


        
        p_z_param_list = [{'params': self.p_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
        self.p_z_optimizer = optim.Adam(p_z_param_list)
       
        q_z_param_list = [{'params': self.q_z_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                            {'params': self.LogZ_unconditional, 'lr': z_lr,"weight_decay":0.1}]
        self.q_z_optimizer = optim.Adam(q_z_param_list)


        p_zx_param_list = [{'params': self.p_zx_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},]
        self.p_zx_optimizer = optim.Adam(p_zx_param_list)
       
        q_zxy_param_list = [{'params': self.q_zxy_mask_generators.parameters(), 'lr': mg_lr_mu,"weight_decay":0.1},
                            {'params': self.LogZ_total_flowestimator.parameters(), 'lr': z_lr,"weight_decay":0.1}]
        self.q_zxy_optimizer = optim.Adam(q_zxy_param_list)



        # GFN_Z_param_list = [{'params': self.LogZ_total_flowestimator, 'lr': z_lr,"weight_decay":0.1},
        # {'params': self.LogZ_unconditional.parameters(), 'lr': z_lr,"weight_decay":0.1}]

        # self.GFN_Z_optimizer = optim.Adam(GFN_Z_param_list)

        # self.GFN_Z_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.GFN_Z_optimizer, milestones=[200,300,350],gamma=0.2)


        taskmodel_param_list = [{'params': self.fc.parameters(), 'lr': lr},
        {'params': self.out_layer.parameters(), 'lr': lr}]

        self.taskmodel_optimizer = optim.Adam(taskmodel_param_list)
        self.taskmodel_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.taskmodel_optimizer,
         milestones=self.opt.schedule_milestone,gamma=0.2)


        self.to(self.device)


      



    def forward(self, x, y, x_mask, y_mask,ans):

        '''
        x is txt 
        y is image 
        ans is the answer 
        ''' 

        x=x.reshape(x.shape[0],-1)
        ans=torch.nn.functional.one_hot(y, self.num_classes)#convert to one hot vector
        batch_size,input_dim=x.shape[0],x.shape[1]
        
            
        LogZ_unconditional=self.LogZ_unconditional#top down mask has partition functionindepdent of input x
        LogZ_conditional=self.LogZ_total_flowestimator(torch.cat([x,y],1))#partition function for bottomp mask is input dependent
        
        LogPF_qz=torch.zeros(batch_size).to(device)# forward probability, unconditonal mask
        LogPB_qz=torch.zeros(batch_size).to(device)# backward prob
        LogR_qz=torch.zeros(batch_size).to(device)

        LogPF_BNN=torch.zeros(batch_size).to(device)
        LogPB_BNN=torch.zeros(batch_size).to(device)

        LogPF_qzxy=torch.zeros(batch_size).to(device)# forward probability, for contional mask generator
        LogR_qzxy=torch.zeros(batch_size).to(device)
        LogPB_qzxy=torch.zeros(batch_size).to(device)

        Log_pzx=torch.zeros(batch_size).to(device)#part of log R for bottom up mask
        Log_pz=torch.zeros(batch_size).to(device)#part of log R for topdown mask

        '''
        mu mask generation, indepedent of the input x 
        '''

        #initialize masks as all zeros(dropout them all)
        #one batch share the same mu mask
        if self.train:#use tempered version of the policy q(z) or q(z|x,y) during training
            temperature=self.temperature
        else:
            temperature=1.0
        masks_qz=[[] for _ in range(len(self.enc_list+self.dec_list))]
        
        for layer_idx in range(len(self.enc_list+self.dec_list)):for enc in self.enc_list:

            layer=(self.enc_list+self.dec_list)[layer_idx]

            if ("topdown" in mask) or ("upNdown" in mask):
                if layer_idx==0:
                    qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](torch.zeros(batch_size,input_dim).to(device),temperature)      
                    
                else:
                    ##concatenate all previous masks
                    previous_mask=[]
                    for j in range(layer_idx):
                        previous_mask.append(masks_qz[j][-1])
                    previous_mask=torch.cat(previous_mask,1)
                    qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](previous_mask,temperature)
                
                masks_qz[layer_idx].append(qz_mask_l.detach().clone())
                

                
                LogPF_qz+=(qz_mask_l*torch.log(qz_p_l)+(1-qz_mask_l)*torch.log(1-qz_p_l)).sum(1)

                LogPB_qz+=0 #uniform backward P
            else:
                masks_qz[layer_idx].append(torch.ones(self.layer_dims[layer_idx]).to(device))
                

        '''
        forward pass
        '''
        actual_masks=[]
        masks_conditional=[]
        for layer_idx in range(len(self.enc_list)):
            enc=self.enc_list[layer_idx]

            # x=layer(x)
            # x = self.activation()(x)


            #####different masks generator
            if "bottomup" in mask or "upNdown" in mask:

                if self.train:
                    #during training use q(z|x,y;phi) to sample mask
                    if layer_idx==0:

                        m_conditional_l,qzxy_p_l= self.q_zxy_mask_generators[layer_idx](torch.zeros(batch_size,x.shape[1]).to(device),x.clone().detach(),y.float().clone().detach(),temperature)#generate mask based on activation from previous layer, detach from BNN training
                        
                        
                    else:
                        previous_actual_mask=[]#use previous actual masks
                        for j in range(layer_idx):
                            previous_actual_mask.append(actual_masks[j])
                        previous_actual_mask=torch.cat(previous_actual_mask,1)
                        
                        m_conditional_l,qzxy_p_l = self.q_zxy_mask_generators[layer_idx](previous_actual_mask,x.clone().detach(),ans.float().clone().detach(),temperature)
            

                    masks_conditional.append(m_conditional_l)

                    ###add log P_F_Z to the GFN loss

                    LogPF_qzxy+=(m_conditional_l*torch.log(qzxy_p_l)+(1-m_conditional_l)*torch.log(1-qzxy_p_l)).sum(1)

                    LogPB_qzxy-=0
                

                else:
                    #during inference use p(z|x;xi) to sample mask
                        if layer_idx==0:
                            m_conditional_l,_= Log_P_zx_l = self.p_zx_mask_generators[layer_idx](x.clone().detach())
                        else:
                            previous_actual_mask=[]#use previous actual masks
                            for j in range(layer_idx):
                                previous_actual_mask.append(actual_masks[j])

                            ###calculate p(z|x;xi)
                            input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],1)

                            m_conditional_l,_= self.p_zx_mask_generators[layer_idx](input_pzx)#generate mask based on activation from previous layer, detach from BNN training


                        masks_conditional.append(m_conditional_l)

            else:
                    masks_conditional.append(torch.ones(x.shape).to(device))

                



            EPSILON=random.uniform(0,1)

            if mask=="random" or (EPSILON<self.random_chance  and self.training):# during training ,of a centain chance a random policy will be used to explore the space
                
                m=self.rand_mask_generator(x).to(device)

            elif mask=="topdown":
                m_qz_l=masks_qz[layer_idx][-1]  
                m=m_qz_l
            elif mask=="bottomup" or mask=="upNdown":   
                m=m_conditional_l


            elif mask=="none":
                m=torch.ones(x.shape).to(device)

            m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
            # #rescale the output
            # if mask!="none":
            #     multipliers = m.shape[1] / (m.sum(1) + 1e-6)
            #     x = torch.mul((x * m).T, multipliers).T

            actual_masks.append(m)
            ####apply the dropout mask
            x = layer(x, x_mask,m)

            ###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
            if layer_idx==0:
                
                ###calculate p(z|x;xi)
                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(x.clone().detach(),m)
                #calculate p(z|xi)
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(torch.zeros(batch_size,input_dim).to(device),m)
                
            else:
                previous_actual_mask=[]#use previous actual masks
                for j in range(layer_idx):
                    previous_actual_mask.append(actual_masks[j])

                ###calculate p(z|x;xi)
                input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],1)

                Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


                ###calculate p(zxi)
                input_pz=torch.cat(previous_actual_mask,1)
        
                Log_P_z_l = self.p_z_mask_generators[layer_idx].log_prob(input_pz,m)#generate mask based on activation from previous layer, detach from BNN training


            Log_pzx+=Log_P_zx_l
            Log_pz+=Log_P_z_l




     

        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
