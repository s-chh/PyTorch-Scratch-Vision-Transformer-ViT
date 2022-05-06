import torch
import torch.nn as nn

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super(EmbedLayer, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S
        x = x.transpose(1, 2)  # B E S -> B S E
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        return x


class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v):

        x_queries = self.queries(q).reshape(q.shape[0], q.shape[1], self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        x_queries = x_queries.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        x_keys = self.keys(k).reshape(k.shape[0], k.shape[1], self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
        x_keys = x_keys.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
        x_values = self.values(v).reshape(v.shape[0], v.shape[1], self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
        x_values = x_values.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE

        x_queries = x_queries.reshape([-1, x_queries.shape[2], x_queries.shape[3]])  # B, H, Q, HE -> (BH), Q, HE
        x_keys = x_keys.reshape([-1, x_keys.shape[2], x_keys.shape[3]])  # B, H, K, HE -> (BH), K, HE
        x_values = x_values.reshape([-1, x_values.shape[2], x_values.shape[3]])  # B, H, V, HE -> (BH), V, HE

        x_keys = x_keys.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K
        x_attention = x_queries.bmm(x_keys)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K
        x_attention = x_attention / (self.head_embed_dim ** 0.5)
        x_attention = torch.softmax(x_attention, dim=-1)

        x = x_attention.bmm(x_values)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        x = x.reshape([-1, self.n_attention_heads, x.shape[1], x.shape[2]])  # (BH), Q, HE -> B, H, Q, HE
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B, Q, H, HE -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.attention = AttentionLayer(args)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):
        x_ = self.attention(x, x, x)
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


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
        self.classifier = Classifier(args)

        # Intialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
