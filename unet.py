import math
import torch
import torch.nn as nn
from encoders.RGB_encoder import RGB_encoder
from encoders.CSI_encoder import CSI_encoder

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_norm(num_channels, num_groups):
    return nn.GroupNorm(num_groups, num_channels)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, condition_dim):
        super(AdaptiveLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.condition_dim = condition_dim
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.mlp_gamma = nn.Linear(condition_dim, normalized_shape)
        self.mlp_beta = nn.Linear(condition_dim, normalized_shape)

    def forward(self, x, condition):
        gamma = self.mlp_gamma(condition)
        beta = self.mlp_beta(condition)
        x = self.layer_norm(x)
        x = x * gamma + beta
        return x


class AdaLNZero(nn.Module):
    def __init__(self, normalized_shape, condition_dim):
        super(AdaLNZero, self).__init__()
        self.normalized_shape = normalized_shape
        self.condition_dim = condition_dim
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, normalized_shape),
            nn.SiLU()
        )

    def forward(self, x, condition):
        gamma = self.mlp(condition).view(-1, *[1] * (len(self.normalized_shape) - 1))
        beta = gamma * torch.mean(x, dim=-1, keepdim=True)
        
        x = x - beta
        x = x / (torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-6))
        x = x * gamma
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device      = x.device
        half_dim    = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # x * self.scale和emb外积
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb = None, rgb_condition_emb = None, csi_condition_emb = None):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x, time_emb = None, rgb_condition_emb = None, csi_condition_emb = None):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = get_norm(in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w  = x.shape
        q, k, v     = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention   = torch.softmax(dot_products, dim=-1)
        out         = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out         = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)

    def forward(self, query, key_value):

        query_transformed = self.query_linear(query)
        key_transformed = self.key_linear(key_value)
        value_transformed = self.value_linear(key_value)

        scores = torch.matmul(query_transformed, key_transformed.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)

        weighted_values = torch.matmul(attention_weights, value_transformed)

        return weighted_values




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_emb_dim, rgb_condition_dim, csi_condition_dim, activation=SiLU(), num_groups=1, use_attention=False):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm_2 = get_norm(out_channels, num_groups)



        self.norm_3 = get_norm(out_channels, num_groups)
        self.conv_3 = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias  = nn.Linear(time_emb_dim, out_channels)
        self.rgb_condition_bias = nn.Linear(rgb_condition_dim, out_channels)
        self.csi_condition_bias = nn.Linear(csi_condition_dim, out_channels)

        self.residual_connection    = nn.Conv2d(in_channels, out_channels, 1)
        self.attention  = nn.Identity() if not use_attention else AttentionBlock(out_channels, num_groups)
        self.condition_bias = nn.Linear(rgb_condition_dim + csi_condition_dim, out_channels)
    
    def forward(self, x, time_emb, rgb_condition_emb, csi_condition_emb):

        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        out += self.time_bias(self.activation(time_emb))[:, :, None, None]
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out)

        condition_emb = torch.cat((csi_condition_emb, rgb_condition_emb), dim = -1)

        condition = self.condition_bias(self.activation(condition_emb))[:, :, None, None]

        out += condition

        out = self.activation(self.norm_3(out))
        out = self.conv_3(out) + self.residual_connection(x)
        out = self.attention(out)

        return out



class UNet(nn.Module):
    def __init__(
        self, img_channels = 1, base_channels=32, channel_mults=(1, 2, 4, 8),
        num_res_blocks=3, time_emb_dim=128, time_emb_scale=1.0, rgb_condition_dim = 512, csi_condition_dim = 512, activation=SiLU(),
        dropout=0.1, attention_resolutions=(1,), num_groups=1):
        super().__init__()

        self.activation = activation
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv  = nn.Conv2d(img_channels, base_channels, 3, stride = [1], padding=[1,4])

        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()

        # channels指的是每一个模块处理后的通道数
        # now_channels是一个中间变量，代表中间的通道数
        channels        = [base_channels]
        now_channels    = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels, dropout,
                        time_emb_dim=time_emb_dim, rgb_condition_dim=rgb_condition_dim, csi_condition_dim = csi_condition_dim, activation=activation,
                        num_groups=num_groups, use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)


        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, rgb_condition_dim=rgb_condition_dim, csi_condition_dim = csi_condition_dim, activation=activation,
                    num_groups=num_groups, use_attention=True,
                ),
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, rgb_condition_dim=rgb_condition_dim, csi_condition_dim = csi_condition_dim, activation=activation, 
                    num_groups=num_groups, use_attention=False,
                ),
            ]
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout, 
                    time_emb_dim=time_emb_dim, rgb_condition_dim=rgb_condition_dim, csi_condition_dim = csi_condition_dim, activation=activation, 
                    num_groups=num_groups, use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_conv1 = nn.Conv2d(base_channels, img_channels, 7, padding=[3, 0])

        self.up = Upsample(base_channels)

        self.out_conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=[1, 1])
        self.out_conv3 = nn.Conv2d(base_channels, img_channels, 1)

        self.rgb_encoder = RGB_encoder()
        self.csi_encoder = CSI_encoder()

        self.out_norm =  get_norm(32, num_groups)



    def forward(self, x, time, rgb_condition, csi_condition):

        time_emb = self.time_mlp(time)
        rgb_condition_emb = self.rgb_encoder(rgb_condition)
        csi_condition_emb = self.csi_encoder(csi_condition)

        x = self.init_conv(x)

        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb, rgb_condition_emb, csi_condition_emb)
            skips.append(x)
                
        for layer in self.mid:
            x = layer(x, time_emb, rgb_condition_emb, csi_condition_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, rgb_condition_emb, csi_condition_emb)

        x = self.out_conv1(x)
        return x
