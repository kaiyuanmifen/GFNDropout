# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model_GFN.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
import random

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

        self.dropout = nn.Dropout(__C.DROPOUT_R)

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
        att_map = self.dropout(att_map)

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


# ---------------------------
# ---- GFN related ----
# ---------------------------
class RandomMaskGenerator(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = torch.tensor(dropout_rate).type(torch.float32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x):
        return torch.bernoulli((1. - self.dropout_rate) * torch.ones(x.shape))

    def log_prob(self, x, m):
        dist = (1. - self.dropout_rate) * torch.ones(x.shape).to(self.device)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


class MLP_(nn.Module):
    def __init__(self, in_dim=784, out_dim=10, hidden=None, activation=nn.LeakyReLU):
        super().__init__()
        if hidden is None:
            hidden = [32, 32]
        h_old = in_dim
        self.fc = nn.ModuleList()
  
        self.LN = nn.ModuleList()
        for h in hidden:
            self.fc.append(nn.Linear(h_old, h))
            self.LN.append(torch.nn.LayerNorm(h))
            h_old = h

        ###layerNorm to cope with variation if inputs in


        self.out_layer = nn.Linear(h_old, out_dim)
        self.activation = activation

    def forward(self, x):

        for layer,ln in zip(self.fc,self.LN):
            x = self.activation()(layer(x))
            x=ln(x)
            
        x = self.out_layer(x)
        return x

class MLPMaskGenerator(nn.Module):
    def __init__(self, in_dim,out_dim, hidden=[32], activation=nn.LeakyReLU):
        super().__init__()
       

        self.mlp = MLP_(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            activation=activation,
        )

    def _dist(self, x,T=1.0):

        x = self.mlp(x)
    
        #x = torch.exp(x)/(1.0+torch.exp(x))
        x=nn.Sigmoid()(x/T)
    
        dist=x
        
        #dist = dist.clamp(1e-3, 1.0-1e-3)

        return dist

    def forward(self, x,T=1.0):

        
        probs=self._dist(x,T)

        return torch.bernoulli(probs), probs

    def log_prob(self, x, m):
        dist = self._dist(x)
        probs = dist * m + (1. - dist) * (1. - m)
        return torch.log(probs).sum(1)


def construct_random_mask_generators(
        n_layers,
        dropout_rate,
):


    mask_generators = nn.ModuleList()
    for layer in range(n_layers):
        mask_generators.append(
            RandomMaskGenerator(
                dropout_rate=dropout_rate,
            )
        )

    return mask_generators



def construct_unconditional_mask_generators(
        layer_dims,
        hiddens=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()

    for layer_idx in range(len(layer_dims)):

        if layer_idx==0:
            in_dim=512 #arbitary number
            out_dim=layer_dims[layer_idx]
        
        else:
            in_dim=0
            for j in range(layer_idx):
                 in_dim+=layer_dims[j]

            out_dim=layer_dims[layer_idx]

        
        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hiddens,
                activation=activation
            )
        )

    return mask_generators



def construct_conditional_mask_generators(
        layer_dims,
        additional_input_dims,
        hiddens=None,
        activation=nn.LeakyReLU
):
    mask_generators = nn.ModuleList()
    for layer_idx in range(len(layer_dims)):

        if layer_idx==0:
            in_dim=layer_dims[layer_idx]+additional_input_dims[layer_idx]
            out_dim=layer_dims[layer_idx]
        
        else:
            in_dim=layer_dims[layer_idx]+additional_input_dims[layer_idx]
            for j in range(layer_idx):
                 in_dim+=layer_dims[j]

            out_dim=layer_dims[layer_idx]

        mask_generators.append(
            MLPMaskGenerator(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden=hiddens,
                activation=activation
            )
        )

    return mask_generators



# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x



# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
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

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class SGA_GFN(nn.Module):
    def __init__(self, __C):
        super(SGA_GFN, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        #self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        #self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        #self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)


        ###GFlowOut related 


        self.rand_mask_generator=RandomMaskGenerator(dropout_rate=__C.DROPOUT_R)

        hiddens=[32,32]
        self.maskgenerator_input_shapes=[512,512,512]
        self.additional_input_dims=[512,512,512]#size of the answer(label)

        self.p_zx_mask_generators=construct_conditional_mask_generators(layer_dims=self.maskgenerator_input_shapes,
                                                        additional_input_dims=[0,0,0],
                                                        hiddens=hiddens)#p(z|x)

        self.q_zxy_mask_generators=construct_conditional_mask_generators(layer_dims=self.maskgenerator_input_shapes,
                                                        additional_input_dims=self.additional_input_dims,
                                                        hiddens=hiddens)#q(z|x,y) 

    
        self.p_z_mask_generators=RandomMaskGenerator(dropout_rate=__C.DROPOUT_R)

        self.q_z_mask_generators=construct_unconditional_mask_generators(layer_dims=self.maskgenerator_input_shapes,
                                                        hiddens=hiddens)#q(z)

        self.mask=__C.GFlowOut
        self.BNN=False
        self.random_chance=0.1 #chances of using random mask during traning
        self.temperature=2 # temperature in sigmoid , high temperature more close the p to 0.5 for binary mask

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, x, y, x_mask, y_mask,ans):

        x,actual_masks,masks_qz,masks_conditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz=self.GFN_forward(x, y, x_mask, y_mask,ans)

        return x,actual_masks,masks_qz,masks_conditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz

    def GFN_forward(self,x, y, x_mask, y_mask,ans):

        '''
        x shape (bsz,length,512)
        ans has shape (bsz,3129)
        '''
        '''
        mu mask generation, indepedent of the input x 
        '''

        #initialize masks as all zeros(dropout them all)

        batch_size=x.shape[0]

        LogPF_qz=torch.zeros(batch_size).to(self.device)# forward probability, unconditonal mask
        LogPB_qz=torch.zeros(batch_size).to(self.device)# backward prob
        #LogR_qz=torch.zeros(batch_size).to(self.device)

        LogPF_BNN=torch.zeros(batch_size).to(self.device)
        LogPB_BNN=torch.zeros(batch_size).to(self.device)

        LogPF_qzxy=torch.zeros(batch_size).to(self.device)# forward probability, for contional mask generator
        #LogR_qzxy=torch.zeros(batch_size).to(self.device)
        LogPB_qzxy=torch.zeros(batch_size).to(self.device)

        Log_pzx=torch.zeros(batch_size).to(self.device)#part of log R for bottom up mask
        Log_pz=torch.zeros(batch_size).to(self.device)#part of log R for topdown mask

        

        #one batch share the same mu mask

        if self.train:#use tempered version of the policy q(z) or q(z|x,y) during training
            temperature=self.temperature
        else:
            temperature=1.0

        
        masks_qz=[[] for _ in range(3)]
        
        for layer_idx in range(3):
        
            if "topdown"==self.mask:
                if layer_idx==0:
                    qz_mask_l,qz_p_l=self.q_z_mask_generators[layer_idx](torch.zeros(batch_size,512),temperature)#784 is an arbitary number here      
                    
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
                masks_qz[layer_idx].append(torch.ones(self.maskgenerator_input_shapes[layer_idx]).to(self.device))
                

        '''
        forward pass
        '''
        actual_masks=[]
        masks_conditional=[]

        if self.BNN:

            print("to be defined")

        else:
                
   
            #####different masks generator
            for layer_idx in range(3):
                
                if ("bottomup" in self.mask) or ("bNL" in self.mask):

                    if self.train:
                        #during training use q(z|x,y;phi) to sample mask
                        if layer_idx==0:
                            input_qzxy=torch.cat([ans.unsqueeze(1).repeat(1,x.shape[1],1)]+[x.clone().detach()],2)
                            m_conditional_l,qzxy_p_l= self.q_zxy_mask_generators[layer_idx](input_qzxy,temperature)#generate mask based on activation from previous layer, detach from BNN training
                            
                            
                        else:
                            previous_actual_mask=[]#use previous actual masks
                            for j in range(layer_idx):
                                previous_actual_mask.append(actual_masks[j])
                            
                            
                            
                            #concatenate this way for computational efficiency
                            input_qzxy=torch.cat(previous_actual_mask+[ans.unsqueeze(1).repeat(1,x.shape[1],1)]+[x.clone().detach()],2)
                            m_conditional_l,qzxy_p_l = self.q_zxy_mask_generators[layer_idx](input_qzxy,temperature)
                        
                  
                        masks_conditional.append(m_conditional_l)

                        ###add log P_F_Z to the GFN loss

                        LogPF_qzxy+=(m_conditional_l*torch.log(qzxy_p_l)+(1-m_conditional_l)*torch.log(1-qzxy_p_l)).sum(2).sum(1)
                        
                        LogPB_qzxy-=0
                    

                    else:
                        #during inference use p(z|x;xi) to sample mask
                            if layer_idx==0:
                                m_conditional_l,_= Log_P_zx_l = self.p_zx_mask_generators[layer_idx](out.clone().detach())
                            else:
                                previous_actual_mask=[]#use previous actual masks
                                for j in range(layer_idx):
                                    previous_actual_mask.append(actual_masks[j])

                                ###calculate p(z|x;xi)
                                input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],2)

                                m_conditional_l,_= self.p_zx_mask_generators[layer_idx](input_pzx)#generate mask based on activation from previous layer, detach from BNN training


                            masks_conditional.append(m_conditional_l)

                else:
                        masks_conditional.append(torch.ones(x.shape))

                    



                EPSILON=random.uniform(0,1)

                if self.mask=="random" or (EPSILON<self.random_chance and self.training):# during training ,of a centain chance a random policy will be used to explore the space
                    
                    m=self.rand_mask_generator(torch.zeros(x.shape).to(self.device))

                elif self.mask=="topdown":
                    m_qz_l=masks_qz[layer_idx][-1]  
                    m=m_qz_l
                elif self.mask=="bottomup":   
                    m=m_conditional_l
                elif self.mask=="bNL":
                    m_qz_l=masks_qz[layer_idx][-1]  
                    m=m_qz_l*m_conditional_l

                elif self.mask=="none":
                    m=torch.ones(x.shape[0],x.shape[1],x.shape[2]).to(self.device)

                m=m.to(self.device)
                m=m.detach().clone()#mask should not interfere backbone model(MLP or resnet etc) training
                


                ###calculate p(z;xi) and p(z|x;xi) both of which are needed for training
                if layer_idx==0:
                    
                    ###calculate p(z|x;xi)
                    Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(x.clone().detach(),m)
                    #calculate p(z|xi)
                    Log_P_z_l = self.p_z_mask_generators.log_prob(m,m)
                    
                else:
                    previous_actual_mask=[]#use previous actual masks
                    for j in range(layer_idx):
                        previous_actual_mask.append(actual_masks[j])

                    ###calculate p(z|x;xi)
                    input_pzx=torch.cat(previous_actual_mask+[x.clone().detach()],2)

                    Log_P_zx_l = self.p_zx_mask_generators[layer_idx].log_prob(input_pzx,m)#generate mask based on activation from previous layer, detach from BNN training


                    ###calculate p(z;xi)
                    
                    Log_P_z_l = self.p_z_mask_generators.log_prob(m,m)#generate mask based on activation from previous layer, detach from BNN training


                Log_pzx+=Log_P_zx_l.sum(1)
                Log_pz+=Log_P_z_l.sum(1)


                actual_masks.append(m)

                ###apply the mask

                if layer_idx==0:
                    #self-attention
                    att=self.mhatt1(x, x, x, x_mask)
                    att=att.mul(m)
                    x = self.norm1(x + att)


                elif layer_idx==1:
                    #cross attention

                    att=self.mhatt2(x, x, x, x_mask)
                    att=att.mul(m)
                    x = self.norm2(x + att)


                elif layer_idx==2:
                    ###feedforward layer
                    output=self.ffn(x)
                    output=output.mul(m)    
                    x = self.norm3(x + output)

        return x,actual_masks,masks_qz,masks_conditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz






# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------
class MCA_ED_GFN(nn.Module):
    def __init__(self, __C):
        super(MCA_ED_GFN, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER//2)])
        self.dec_list_GFN = nn.ModuleList([SGA_GFN(__C) for _ in range(__C.LAYER//2)])


        ###LogZ normalizing factor
        self.LogZ_unconditional=nn.Parameter(torch.tensor(0.0))#using condition is too much calculation in this transformer setting, therefore use unconditional version


        self.ans_projector=MLP_(in_dim=3129, out_dim=512)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, x, y, x_mask, y_mask,ans=None):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        #GFlowOut in the last few layers of decoder only

        ans=self.ans_projector(ans)
        batch_size=x.shape[0]

        all_LogPF_qz=torch.zeros(batch_size).to(self.device)# forward probability, unconditonal mask
        all_LogPB_qz=torch.zeros(batch_size).to(self.device)# backward prob
        #LogR_qz=torch.zeros(batch_size).to(self.device)

        all_LogPF_BNN=torch.zeros(batch_size).to(self.device)
        all_LogPB_BNN=torch.zeros(batch_size).to(self.device)

        all_LogPF_qzxy=torch.zeros(batch_size).to(self.device)# forward probability, for contional mask generator
        #LogR_qzxy=torch.zeros(batch_size).to(self.device)
        all_LogPB_qzxy=torch.zeros(batch_size).to(self.device)

        all_Log_pzx=torch.zeros(batch_size).to(self.device)#part of log R for bottom up mask
        all_Log_pz=torch.zeros(batch_size).to(self.device)#part of log R for topdown mask

        for dec in self.dec_list_GFN:
            y,actual_masks,masks_qz,masks_conditional,LogPF_qz,LogPB_qz,LogPF_BNN,LogPB_BNN,LogPF_qzxy,LogPB_qzxy,Log_pzx,Log_pz = dec(y, x, y_mask, x_mask,ans)

            all_LogPF_qz+=LogPF_qz
            all_LogPB_qz+=LogPB_qz

            all_LogPF_BNN+=LogPF_BNN
            all_LogPB_BNN+=LogPB_BNN

            all_LogPF_qzxy+=LogPF_qzxy
            all_LogPB_qzxy+=LogPB_qzxy

            all_Log_pzx+=Log_pzx
            all_Log_pz+=Log_pz

        return x, y,self.LogZ_unconditional,all_LogPF_qz,all_LogPB_qz,all_LogPF_BNN,all_LogPB_BNN,all_LogPF_qzxy,all_LogPB_qzxy,all_Log_pzx,all_Log_pz
        #return x, y