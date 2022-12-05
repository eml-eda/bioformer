import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np 

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        bound1 = 1 / (dim ** .5)
        bound2 = 1 / (hidden_dim ** .5)
        nn.init.uniform_(self.net[0].weight, -bound1, bound1)
        nn.init.uniform_(self.net[0].bias, -bound1, bound1)
        nn.init.uniform_(self.net[3].weight, -bound2, bound2)
        nn.init.uniform_(self.net[3].bias, -bound2, bound2)

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        bound = 1 / (dim ** .5)
        nn.init.uniform_(self.to_qkv.weight, -bound, bound)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        bound = 1 / (inner_dim ** .5)
        nn.init.uniform_(self.to_out[0].weight, -bound, bound)
        nn.init.uniform_(self.to_out[0].bias, -bound, bound)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        qkv = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size1, patch_size2, patch_size3, ch_1, ch_2, ch_3, tcn_layers, num_classes, dim_patch, depth, heads, pool = 'cls', channels = 3,
                 dim_head = 64, dropout = 0., emb_dropout = 0., use_cls_token=True, 
                 sessions="ignore", subjects="ignore", training_config="ignore", pretrained="ignore", chunk_idx="ignore", chunk_i="ignore"):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height1, patch_width1 = pair(patch_size1)
        patch_height2, patch_width2 = pair(patch_size2)
        patch_height3, patch_width3 = pair(patch_size3)
        patch_width = [patch_width3, patch_width2, patch_width1]
        ch = [ch_3, ch_2, ch_1]
        if patch_width1 != None:
            assert image_height % patch_height1 == 0 and image_width % patch_width1 == 0, 'Image dimensions must be divisible by the patch size.'
        if patch_width2 != None:
            assert image_height % patch_height2 == 0 and image_width % patch_width2 == 0, 'Image dimensions must be divisible by the patch size.'
        if patch_width3 != None:
            assert image_height % patch_height3 == 0 and image_width % patch_width3 == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height1) * (image_width // patch_width1)
        if patch_width2 != None:
            num_patches = (image_height // (patch_height1 * patch_height2)) * (image_width // (patch_width1 * patch_width2))
        if patch_width3 != None:
            num_patches = (image_height // (patch_height1 * patch_height2 * patch_height3)) * (image_width // (patch_width1 * patch_width2 * patch_width3))
        patch_dim = ch_1 * patch_height1 * patch_width1
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        layerlist = []
        ch_previous = channels
        for i in np.arange(3):
            if patch_width[i] != None:
                if tcn_layers == 2:
                    layerlist.append(nn.Conv2d(ch_previous, ch[i], kernel_size = (1,3), padding = (0,1))) 
                    layerlist.append(nn.ReLU(inplace=True))  # Apply activation function - ReLU
                    layerlist.append(nn.BatchNorm2d(ch[i]))  # Apply batch normalization
                if i != 2:
                    layerlist.append(nn.Conv2d(ch[i], ch[i], kernel_size = (1,patch_width[i]), stride = (1,patch_width[i]))) 
                    layerlist.append(nn.ReLU(inplace=True))  # Apply activation function - ReLU
                    layerlist.append(nn.BatchNorm2d(ch[i]))  # Apply batch normalization
                ch_previous = ch[i]


        layerlist.append(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height1, p2 = patch_width1))
        layerlist.append(nn.Linear(patch_dim, dim_patch))
        self.to_patch_embedding = nn.Sequential(*layerlist)

        bound = 1 / (patch_dim ** .5)
        nn.init.uniform_(self.to_patch_embedding[-1].weight, -bound, bound)
        nn.init.uniform_(self.to_patch_embedding[-1].bias, -bound, bound)
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim_patch))
        else:
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, dim_patch))
        nn.init.normal_(self.pos_embedding, mean=0, std=.02)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim_patch))
        nn.init.zeros_(self.cls_token)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim_patch, depth, heads, dim_head, dim_patch * 2, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim_patch),
            nn.Linear(dim_patch, num_classes)
        )
        bound = 1 / (dim_patch ** .5)
        nn.init.uniform_(self.mlp_head[1].weight, -bound, bound)
        nn.init.uniform_(self.mlp_head[1].bias, -bound, bound)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else :
            x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


# Formula: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
def get_conv_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1, **ignore):
    tuple_to_int = lambda x: int(x[0]) if isinstance(x, tuple) else int(x)
    kernel_size, stride, padding, dilation = tuple_to_int(kernel_size), tuple_to_int(stride), tuple_to_int(padding), tuple_to_int(dilation)
    return int( ( (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1 )

class TEMPONet(nn.Module):
    
    def __init__(self, input_size=300, input_channels=14):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = (1,3), dilation=(1,2), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = (1,3), dilation=(1,2), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = (1,5), stride = (1,2), padding=(0,2)),
            torch.nn.AvgPool2d((1,2), stride=(1,2), padding=(0,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = (1,3), dilation=(1,4), padding=(0,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = (1,3), dilation=(1,4), padding=(0,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = (1,5), stride = (1,2), padding=(0,2)),
            torch.nn.AvgPool2d((1,2), stride=(1,2), padding=(0,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = (1,3), dilation=(1,8), padding=(0,8)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = (1,3), dilation=(1,8), padding=(0,8)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = (1,5), stride = (1,4), padding=(0,2)),
            torch.nn.AvgPool2d((1,2), stride=(1,2), padding=(0,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        def get_fc_input_size():
            is_layer_conv = lambda x: isinstance(x, nn.Conv2d) or isinstance(x, nn.AvgPool2d)
            layers = list(filter(is_layer_conv, [*self.conv1, *self.conv2, *self.conv3]))
            
            output_size = input_size
            last_layer_output_planes = 1
            for layer in layers:
                output_size = get_conv_output_size(output_size, **vars(layer))
                last_layer_output_planes = layer.out_channels if hasattr(layer, "out_channels") else last_layer_output_planes
            
            return output_size * last_layer_output_planes

        self.fc = nn.Sequential(
            nn.Linear(256, 256), # input=640
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 8),
        )
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.flatten(1)

        x = self.fc(x)
        
        return x