import math

import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

device = torch.device('cuda')

class Position_Embedding(nn.Module):
    def __init__(self, batchsize=1,img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None, dropout=0.1, num_class=2):
        super(Position_Embedding, self).__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.batchsize = batchsize
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        ##卷积+Flatten
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16, padding=0, groups=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        ##token+位置嵌入
        self.class_token = torch.randn(batchsize, 1, embed_dim, requires_grad=True).to(device)
        self.position_embeddings = nn.Parameter(
            torch.zeros(batchsize, math.prod(self.grid_size) + 1, embed_dim))  # 和class_token后的X保持一致(值是否应该全为0)
        ##torch.zeros(batchsize, grid_size * grid_size + 1, embed_dim, requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B, C, H, W = x.shape ##(1,3,224,224)
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        out = self.conv(x).flatten(2).transpose(1,2)
        out = self.norm(out)
        out = torch.cat((self.class_token, out),dim=1)##(B, 197, 768)
        out = self.position_embeddings + out##(B,197,768)
        out = self.dropout(out)

        return out


##克隆函数
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def Attention(query, key, value, mask, dropout=None):
    d_K = query.size(-1)

    attention_score = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_K)

    p_atten = torch.softmax(attention_score, dim=-1)

    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, batchsize = 1,head_num=12,embedding_dim=768,dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        self.batchsize = batchsize

        assert embedding_dim % head_num == 0

        self.d_k = embedding_dim // head_num

        self.head = head_num

        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.attention = None

        self.dropout = nn.Dropout(dropout)

    ##暂不用Mask相关代码
    def forward(self, query, key, value, mask=None):

        query, key, value =\
        [model(x).view(self.batchsize, self.head, -1, self.d_k).transpose(1, 2)##(B, 197, 头数, 词嵌入维度)
            for model, x in zip(self.linears,(query, key, value))]

        x, self.attention = Attention(query,key,value,mask=None,dropout=self.dropout)##[1, 197, 4, 192]
        #转置2,3维
        x = x.transpose(1, 2).contiguous().view(self.batchsize, -1, self.head*self.d_k)

        return self.linears[-1](x)


class MLP_Block(nn.Module):
    def __init__(self, batchsize=1, embed_dim=768, dropout=0.1):
        super(MLP_Block, self).__init__()
        self.linear1 = nn.Linear(embed_dim, 4*embed_dim)
        self.linear2 = nn.Linear(4*embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = self.linear1(x)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        return out

class SublayerConnection(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layerNorm(x)))


class Transformer(nn.Module):
    def __init__(self, batchsize=1, embed_dim=768, head_num=12, dropout=0.1):
        super(Transformer, self).__init__()

        self.self_attn =MultiHeadSelfAttention(batchsize, head_num, embed_dim, dropout)
        self.MLP = MLP_Block(batchsize, embed_dim, dropout)
        self.sublayer = clones(SublayerConnection(embed_dim, dropout), 2)
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.MLP)


class ViT(nn.Module):
    def __init__(self, *, batchsize=1, img_size=224, patch_size=16, head_num=12, embed_dim=768, dropout=0.1, num_classes=2):
        super(ViT, self).__init__()
        self.Position_Embedding = Position_Embedding(batchsize, img_size, patch_size, 3, embed_dim, None, dropout, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layers = clones(Transformer(batchsize, embed_dim, head_num, dropout), 12)

        ##self.layernorm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        ##x[1,3,224,224]
        out = self.Position_Embedding(x)
        out = self.dropout(self.Position_Embedding(x))
        for layer in self.layers:
            out = layer(out).to(device)

        x = self.to_latent(x)


        out = out[:, 0]

        return self.mlp_head(out)

if __name__ == '__main__':
    v = ViT(
        batchsize=1,
        img_size=224,
        patch_size=16,
        head_num=12,
        embed_dim=768,
        dropout=0.1,
        num_classes=4
    )

print("123")