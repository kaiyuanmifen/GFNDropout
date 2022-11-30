# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from core.model_GFN.mca_GFN import MCA_ED_GFN

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, HP, input_dim):
        super(AttFlat, self).__init__()
        self.HP = HP

        self.mlp = MLP(
            in_size=HP.HIDDEN_SIZE,
            mid_size=HP.FLAT_MLP_SIZE,
            out_size=HP.FLAT_GLIMPSES,
            dropout_r=HP.DROPOUT_R,
            use_relu=True,
            HP=HP,
            input_dim=input_dim
        )

        self.linear_merge = nn.Linear(
            HP.HIDDEN_SIZE * HP.FLAT_GLIMPSES,
            HP.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.HP.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, HP, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.HP=HP
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=HP.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if HP.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=HP.WORD_EMBED_SIZE,
            hidden_size=HP.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            HP.IMG_FEAT_SIZE,
            HP.HIDDEN_SIZE
        )

        if self.HP.GFlowOut=="none":
        	self.backbone = MCA_ED(HP)
        else:
        	self.backbone = MCA_ED_GFN(HP)

        self.attflat_img = AttFlat(HP, [64, 100, 512])
        self.attflat_lang = AttFlat(HP, [64, 14, 512])

        self.proj_norm = LayerNorm(HP.FLAT_OUT_SIZE)
        self.proj = nn.Linear(HP.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix,ans=None):
        '''
        ans is only used during training
        '''
        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        if self.HP.GFlowOut=="none":
            lang_feat, img_feat = self.backbone(
                lang_feat,
                img_feat,
                lang_feat_mask,
                img_feat_mask
            )
        else:
            lang_feat, img_feat,LogZ_unconditional,all_LogPF_qz,all_LogPB_qz,all_LogPF_BNN,all_LogPB_BNN,all_LogPF_qzxy,all_LogPB_qzxy,all_Log_pzx,all_Log_pz = self.backbone(
                lang_feat,
                img_feat,
                lang_feat_mask,
                img_feat_mask,
                ans
            )
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        if self.HP.GFlowOut=="none":
            return proj_feat
        else:
            return proj_feat,LogZ_unconditional,all_LogPF_qz,all_LogPB_qz,all_LogPF_BNN,all_LogPB_BNN,all_LogPF_qzxy,all_LogPB_qzxy,all_Log_pzx,all_Log_pz

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
