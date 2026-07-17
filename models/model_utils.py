from collections import OrderedDict
from os.path import join
import pdb
# import eniops

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearFusion(nn.Module):
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):

        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            # add
            nn.LayerNorm(dim2),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class EvidenceHead(nn.Module):
    def __init__(self, dim: int = 256):
        super(EvidenceHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Softplus()          
        )

    def forward(self, x):
        # x: [B, D]
        evidence = self.net(x)                 # [B, 1]
        conf = evidence / (evidence + 1.0)     
        return conf.squeeze(-1)                # [B]

class MultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim = 256,
                 k_dim = 256,
                 v_dim = 256,
                 embed_dim = 256,
                 out_dim = 256,
                 n_head = 4,
                 dropout=0.1,
                 temperature = 1
                 ):
        super(MultiheadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = self.embed_dim//self.n_head
        self.temperature = temperature


        self.w_q = nn.Linear(self.q_dim, embed_dim)
        self.w_k = nn.Linear(self.k_dim, embed_dim)
        self.w_v = nn.Linear(self.v_dim, embed_dim)

        self.scale = (self.embed_dim//self.n_head) ** -0.5

        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q, k, v, return_attn = False):
        q_raw = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = q.shape[0] # B
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_score = F.softmax(attention_score / self.temperature, dim = -1)

        attention_score = self.attn_dropout(attention_score)

        x = torch.matmul(attention_score, v)

        attention_score = attention_score.sum(dim = 1)/self.n_head
        
        attn_out = x.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_out = self.out_proj(attn_out)

        attn_out = self.proj_dropout(attn_out)
        if return_attn:
            return attn_out, attention_score
        else:
            return attn_out
        # return out, attention_score

class DiffMultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim=256,
                 k_dim=256,
                 v_dim=256,
                 embed_dim=256,
                 out_dim=256,
                 n_head=4,
                 dropout=0.1,
                 temperature=1):
        super(DiffMultiheadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = self.embed_dim // self.n_head
        self.temperature = temperature

        self.w_q = nn.Linear(self.q_dim, 2 * embed_dim)
        self.w_k = nn.Linear(self.k_dim, embed_dim)
        self.w_v = nn.Linear(self.v_dim, embed_dim)
        self.scale = (self.embed_dim // self.n_head) ** -0.5
        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        self.lambda_val = nn.Parameter(torch.zeros(n_head))

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, return_attn=False):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = q.shape[0]

        q = q.view(batch_size, -1, 2 * self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2)

        q1 = q[:, 0::2, :, :]
        q2 = q[:, 1::2, :, :]

        attn1 = torch.matmul(q1, k.transpose(-1, -2)) * self.scale
        attn2 = torch.matmul(q2, k.transpose(-1, -2)) * self.scale

        attn1 = F.softmax(attn1 / self.temperature, dim=-1)
        attn2 = F.softmax(attn2 / self.temperature, dim=-1)

        attn1 = self.attn_dropout(attn1)
        attn2 = self.attn_dropout(attn2)

        lam = torch.sigmoid(self.lambda_val).view(1, self.n_head, 1, 1)
        attention_score = attn1 - lam * attn2

        x = torch.matmul(attention_score, v)

        attn_score_out = attention_score.sum(dim=1) / self.n_head

        attn_out = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_dropout(attn_out)

        if return_attn:
            return attn_out, attn_score_out
        else:
            return attn_out

