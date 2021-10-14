
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import pdb


# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super(EmbedLayer, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.randn(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S
        x = x.transpose(1, 2)  # B E S -> B S E
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, args):
        super(SelfAttentionLayer, self).__init__()
        self.num_heads = args.n_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.num_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.num_heads, bias=False)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.num_heads, bias=False)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.num_heads, bias=False)

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):

        x_queries = self.queries(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
        x_queries = x_queries.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
        x_keys = self.keys(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
        x_keys = x_keys.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE
        x_values = self.values(x).reshape(x.shape[0], x.shape[1], self.num_heads, self.head_embed_dim)  # B, S, E -> B, S, H, HE
        x_values = x_values.transpose(1, 2)  # B, S, H, HE -> B, H, S, HE

        x_queries = x_queries.reshape([-1, x_queries.shape[2], x_queries.shape[3]])  # B, H, S, HE -> (BH), S, HE
        x_keys = x_keys.reshape([-1, x_keys.shape[2], x_keys.shape[3]])  # B, H, S, HE -> (BH), S, HE
        x_values = x_values.reshape([-1, x_values.shape[2], x_values.shape[3]])  # B, H, S, HE -> (BH), S, HE

        x_keys = x_keys.transpose(1, 2)  # (BH), S, HE -> (BH), HE, S
        x_attention = x_queries.bmm(x_keys)  # (BH), S, HE  .  (BH), HE, S -> (BH), S, S
        x_attention = x_attention / self.embed_dim ** 0.5
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(x_values)  # (BH), S, S . (BH), S, HE -> (BH), S, HE
        x = x.reshape([-1, self.num_heads, x.shape[1], x.shape[2]])  # (BH), S, HE -> B, H, S, HE
        x = x.transpose(1, 2)  # B, H, S, HE -> B, S, H, HE
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B, S, H, HE -> B, S, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.attention = SelfAttentionLayer(args)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = nn.ReLU() #GELU()
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):
        x_ = self.attention(x)
        x = x + x_
        x = self.norm1(x)
        x_ = self.fc1(x)
        x = self.activation(x)
        x_ = self.fc2(x_)
        x = x + x_
        x = self.norm2(x)
        return x


# Alternate Encoder with PyTorch Built-in Attention
class EncoderWithPyTorchAttention(nn.Module):
    def __init__(self, args):
        super(EncoderWithPyTorchAttention, self).__init__()
        self.attention = nn.MultiheadAttention(args.embed_dim, args.n_heads)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):
        x_ = x.transpose(0, 1)  # N, S, E -> S, N, E
        x_ = self.attention(x_, x_, x_)[0]  # S, N, E -> S, N, E
        x_ = x_.transpose(0, 1)  # S, N, E -> N, S, E
        x = x + x_
        x = self.norm1(x)
        x_ = self.fc1(x)
        x = self.activation(x)
        x_ = self.fc2(x_)
        x = x + x_
        x = self.norm2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)])
        self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
