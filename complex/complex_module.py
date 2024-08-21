import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
import math


def apply_complex(F_r, F_i, X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F_r(X_r) - F_i(X_i), F_r(X_i) + F_i(X_r)), dim=-1)

def apply_complex_sep(F_r, F_i, X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F_r(X_r), F_i(X_i)), dim=-1)

@torch.jit.script
def complex_mul(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.mul(X_r, Y_r) - torch.mul(X_i, Y_i)
    Z_i = torch.mul(X_r, Y_i) + torch.mul(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_bmm(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.bmm(X_r, Y_r) - torch.bmm(X_i, Y_i)
    Z_i = torch.bmm(X_r, Y_i) + torch.bmm(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_bcmm(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.bmm(X_r, Y_r) + torch.bmm(X_i, Y_i)
    Z_i = torch.bmm(X_r, Y_i) - torch.bmm(X_i, Y_r)

    return torch.stack((Z_r, Z_i), dim=-1)


@torch.jit.script
def complex_softmax(X, eps:float = 1e-9):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    X_norm = torch.norm(X, dim=-1)
    X_norm_softmax = F.softmax(X_norm, dim=-1)
    X_change = X_norm_softmax.div(X_norm + eps)
    return torch.stack((X_r*X_change, X_i*X_change), dim=-1)


@torch.jit.script
def transpose_qkv(x, num_heads: int):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1, 2)
    x = x.transpose(1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3], 2)

@torch.jit.script
def transpose_output(x, num_heads: int):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2], 2)
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0], x.shape[1], -1, 2)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, X):
        device = X.device
        dtype = X.dtype
        mask = torch.ones(*X.shape[-3:], device=device, dtype=dtype)
        mask = F.dropout1d(mask, p=self.p, training=self.training)
        return torch.mul(X, mask)


class ComplexSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu_r = self.silu_i = nn.SiLU()

    def forward(self, X):
        return apply_complex_sep(self.silu_r, self.silu_i, X)



class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-9):
        super().__init__()
        
        self.beta = nn.Parameter(torch.zeros([normalized_shape, 2]))
        self.gamma_rr = nn.Parameter(torch.ones([normalized_shape]) * (1.0/math.sqrt(2)))
        self.gamma_ii = nn.Parameter(torch.ones([normalized_shape]) * (1.0/math.sqrt(2)))
        self.gamma_ri = nn.Parameter(torch.zeros([normalized_shape]))
        self.eps = eps

    def forward(self, x):

        x_mean = torch.mean(x, dim = -2, keepdim = True).expand_as(x)
        x_real_var = torch.var(x[..., 0], dim = -1)
        x_imag_var = torch.var(x[..., 1], dim = -1)
        x_var = x_real_var + x_imag_var
        x_div = torch.sqrt(x_var + self.eps)
        x_div = x_div.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = torch.sub(x, x_mean)
        x = torch.div(x, x_div)

        x_real, x_imag = x[..., 0], x[..., 1]
        x = torch.stack((x_real*self.gamma_rr+x_imag*self.gamma_ri, x_real*self.gamma_ri+x_imag*self.gamma_ii), dim=-1)
        x = x + self.beta

        return x
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l_r = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)
        self.l_i = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)

    def forward(self, X):
        return apply_complex(self.l_r, self.l_i, X)



class ComplexMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ComplexSiLU, bias=True, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = ComplexLinear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.fc2 = ComplexLinear(hidden_features, out_features, bias)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ComplexDotProductAttention(nn.Module):
    """
    Query shape: [batch_size, query_num, query_key_dim]
    Key shape: [batch_size, key_value_num, query_key_dim]
    Value shape: [batch_size, key_value_num, value_dim]
    """
    def __init__(self, dropout, **kwargs):
        super(ComplexDotProductAttention, self).__init__(**kwargs)

    def forward(self, queries, keys, values):
        query_key_dim = queries.shape[-2]

        self.attention_weights = complex_softmax(
            complex_bcmm(queries, keys.transpose(1, 2)) / math.sqrt(query_key_dim)
        )

        Y = complex_bmm(self.attention_weights, values)
        return Y


class ComplexMultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size,
        num_hiddens,
        num_heads,
        dropout,
        key_size=None,
        value_size=None,
        bias=False,
        **kwargs
    ):
        super(ComplexMultiHeadAttention, self).__init__(**kwargs)
        key_size = key_size or query_size
        value_size = value_size or query_size
        self.num_heads = num_heads
        self.attention = ComplexDotProductAttention(dropout=dropout)
        self.w_q = ComplexLinear(query_size, num_hiddens, bias=bias)
        self.w_k = ComplexLinear(key_size, num_hiddens, bias=bias)
        self.w_v = ComplexLinear(value_size, num_hiddens, bias=bias)
        self.w_o = ComplexLinear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        Y = self.w_o(output_concat)
        return Y


class ComplexPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=10000):
        super(ComplexPositionalEncoding, self).__init__()
        pcode = torch.zeros((1, max_len, hidden_dim, 2), dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_dim, dtype=torch.float32) / hidden_dim
        )
        pcode[:, :, :, 0] = torch.cos(pos)
        pcode[:, :, :, 1] = torch.sin(pos)
        self.register_buffer("pcode", pcode, persistent=False)
        self.dropout = ComplexDropout(dropout)


    def forward(self, X, step = 0):
        X_pos = max(X.shape[1], step+1)
        X = complex_mul(X, self.pcode[:, step+1: X_pos+1, :, :].to(X.device))
        return self.dropout(X)
    


class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.linear1 = ComplexLinear(input_dim, hidden_dim)
        self.act = ComplexSiLU()
        self.linear2 = ComplexLinear(hidden_dim, output_dim)

    def forward(self, X):
        Y = self.linear2(self.act(self.linear1(X)))
        return Y


class ComplexAddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(ComplexAddNorm, self).__init__(**kwargs)
        self.ln = ComplexLayerNorm(normalized_shape)
        self.dropout = ComplexDropout(dropout)

    def forward(self, X, Y):
        Y = self.dropout(Y)
        Y = self.ln(Y + X)
        return Y

class ComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape, 
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexEncoderBlock, self).__init__(**kwargs)
        self.attention = ComplexMultiHeadAttention(
            key_size = key_dim, 
            query_size = query_dim, 
            value_size = value_dim, 
            num_hiddens = hidden_dim, 
            num_heads = num_heads, 
            dropout = dropout, 
            bias = use_bias
        )

        self.addnorm1 = ComplexAddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, ffn_hidden_dim)
        self.addnorm2 = ComplexAddNorm(norm_shape, dropout)
        self.ln = ComplexLayerNorm(norm_shape)

    def forward(self, X):
        Y = self.attention(X, X, X)
        Z = self.addnorm1(X, Y)
        return self.addnorm2(Z, self.ffn(Z))



class ComplexTransformerEncoder(nn.Module):
    def __init__(
        self,
        origin_dim,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexTransformerEncoder, self).__init__(**kwargs)
        self.fc = ComplexLinear(in_features=origin_dim, out_features=hidden_dim)
        self.origin_dim = origin_dim
        self.hidden_dim = hidden_dim
        self.pos_encoding = ComplexPositionalEncoding(hidden_dim, dropout)
        self.blks = nn.Sequential()
        for n in range(num_layers):
            self.blks.add_module(
                "Block" + str(n),
                ComplexEncoderBlock(
                    key_dim = key_dim,
                    query_dim = query_dim,
                    value_dim = value_dim,
                    hidden_dim = hidden_dim,
                    norm_shape = norm_shape,
                    ffn_input_dim = ffn_input_dim,
                    ffn_hidden_dim = ffn_hidden_dim,
                    num_heads = num_heads,
                    dropout = dropout,
                    use_bias = use_bias
                ),
            )
        self.act = ComplexSiLU()
        self.mlp = ComplexMLP(in_features = self.origin_dim, out_features = self.hidden_dim)

    def forward(self, X, *args):
        X = self.act(self.mlp(X))
        X = self.pos_encoding(X * math.sqrt(self.hidden_dim))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class ComplexToReal(nn.Module):
    def __init__(self):
        super(ComplexToReal, self).__init__()
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(in_features=2, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(dim=-1)
        return x
    

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool_r = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.avg_pool_i = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, X):
        return apply_complex_sep(self.avg_pool_r, self.avg_pool_i, X)