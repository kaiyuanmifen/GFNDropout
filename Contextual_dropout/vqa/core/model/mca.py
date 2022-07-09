# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm, Dropout_variants

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, HP, dim_0, dim_1):
        super(MHAtt, self).__init__()
        self.HP = HP

        self.linear_v = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_k = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_q = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HP.HIDDEN_SIZE, HP.HIDDEN_SIZE)

        if HP.concretedp:
            self.dropout_regularizer = 2. / HP.data_size
            #print(HP.data_size)
        else:
            self.dropout_regularizer = 1.0
        self.dropout = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=1, ARM=HP.ARM,
                                        dropout_dim=1, input_dim=[1, HP.MULTI_HEAD, dim_0, dim_1],
                                        dropout_distribution=HP.dropout_distribution, ctype = HP.ctype)

        # self.dropout = nn.Dropout(HP.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.HP.MULTI_HEAD,
            self.HP.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HP.HIDDEN_SIZE
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

        att_map = F.softmax(scores, dim=-1) # 64, 8, 14, 14 // 64, 8, 100, 100, // 64, 8, 100, 14
        #print('att_map shape', att_map.shape)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, HP, input_dim):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HP.HIDDEN_SIZE,
            mid_size=HP.FF_SIZE,
            out_size=HP.HIDDEN_SIZE,
            dropout_r=HP.DROPOUT_R,
            use_relu=True,
            HP=HP,
            input_dim=input_dim
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, HP):
        super(SA, self).__init__()

        self.mhatt = MHAtt(HP, 14, 14)
        self.ffn = FFN(HP, input_dim=[64, 14, 2048])

        if HP.concretedp:
            self.dropout_regularizer = 2. / HP.data_size
            #print(HP.data_size)
        else:
             self.dropout_regularizer = 1.0
        self.dropout1 = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32,
                                        dropout_dim=2, input_dim=[1, 14, 512], ctype = HP.ctype)


        self.dropout2 = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                         concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                         k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32,
                                         dropout_dim=2, input_dim=[1, 14, 512], ctype = HP.ctype)


        # self.dropout1 = nn.Dropout(HP.DROPOUT_R)
        self.norm1 = LayerNorm(HP.HIDDEN_SIZE)

        # self.dropout2 = nn.Dropout(HP.DROPOUT_R)
        self.norm2 = LayerNorm(HP.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        #print('sa shape', (self.mhatt(x, x, x, x_mask)).shape)
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask) # 64, 14, 512
        ))
        #print('sa ffn shape', (self.ffn(x)).shape)
        x = self.norm2(x + self.dropout2(
            self.ffn(x) # 64, 14, 512
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, HP):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(HP, 100, 100)
        self.mhatt2 = MHAtt(HP, 100, 14)
        self.ffn = FFN(HP, input_dim=[64, 100, 2048])
        if HP.concretedp:
            self.dropout_regularizer = 2. / HP.data_size
            #print(HP.data_size)
        else:
            self.dropout_regularizer = 1.0
        self.dropout1 = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32,
                                         dropout_dim=2, input_dim=[1, 100, 512], ctype = HP.ctype)

        self.dropout2 = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32,
                                         dropout_dim=2, input_dim=[1, 100, 512], ctype = HP.ctype)

        self.dropout3 = Dropout_variants(dropout_regularizer=self.dropout_regularizer, dp_type=HP.dp_type,
                                        concretedp=HP.concretedp, learn_prior=HP.learn_prior,
                                        k=HP.dp_k, eta_const=HP.eta_const, cha_factor=32,
                                         dropout_dim=2, input_dim=[1, 100, 512], ctype = HP.ctype)


        # self.dropout1 = nn.Dropout(HP.DROPOUT_R)
        self.norm1 = LayerNorm(HP.HIDDEN_SIZE)

        # self.dropout2 = nn.Dropout(HP.DROPOUT_R)
        self.norm2 = LayerNorm(HP.HIDDEN_SIZE)

        # self.dropout3 = nn.Dropout(HP.DROPOUT_R)
        self.norm3 = LayerNorm(HP.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        #print('sga shape', (self.mhatt1(x, x, x, x_mask)).shape)
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask) # 64, 100, 512
        ))
        #print('sga 2 shape', (self.mhatt2(y, y, x, y_mask)).shape)
        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask) # 64, 100, 512,
        ))
        #print('sga ffn shape', (self.ffn(x)).shape)
        x = self.norm3(x + self.dropout3(
            self.ffn(x) # 64, 100, 512,
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, HP):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(HP) for _ in range(HP.LAYER)])
        self.dec_list = nn.ModuleList([SGA(HP) for _ in range(HP.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y
