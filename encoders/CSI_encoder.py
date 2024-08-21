import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from complex.complex_module import *

class RF_Transformer(nn.Module):
    def __init__(self, origin_dim=270, dim=512, heads=16, layers=32, dropout=0):
        super(RF_Transformer, self).__init__()

        self.encoder_origin_dim = origin_dim
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.encoder = ComplexTransformerEncoder(
                        origin_dim = self.encoder_origin_dim,
                        key_dim = self.dim,
                        query_dim = self.dim,
                        value_dim = self.dim,
                        hidden_dim = self.dim,
                        norm_shape = self.dim,
                        ffn_input_dim = self.dim,
                        ffn_hidden_dim = self.dim,
                        num_heads = self.heads,
                        num_layers = self.layers,
                        dropout = self.dropout)

        self.mlp = ComplexMLP(in_features = self.dim, out_features = self.dim)


    def forward(self, x):
        x = torch.stack((torch.real(x), torch.imag(x)), dim=-1)
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class CSI_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = RF_Transformer()
        # self.avg_pool = ComplexAvgPool2d(kernel_size=[20, 1], stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=[20, 1], stride=1, padding=0)
        self.fc2 = nn.Linear(512, 512)
        self.act = nn.SiLU()
        self.c2r = ComplexToReal()


    def forward(self, x):
        x = self.transformer(x)
        x = self.c2r(x)
        x = self.act(self.avg_pool(x))
        x = rearrange(x, "b 1 d -> b d")
        x = self.fc2(x)
        return x
    



# if __name__ == "__main__":

#     device = "cuda:0"
#     model = CSI_encoder().to(device)
#     para = sum(p.numel() for p in model.parameters())
#     print("Num params: ", para)
#     print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

#     # print(model)
#     csi = torch.randn((64,20,270), device=device, dtype=torch.complex64)

#     print(csi.shape)
#     output= model(csi)
#     print(output.shape)