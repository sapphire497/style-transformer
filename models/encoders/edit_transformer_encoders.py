import torch
import torch.nn.init as init
from torch import nn
from models.transformer import TransformerDecoderLayer

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class EditTransformerEncoder(nn.Module):
    def __init__(self):
        super(EditTransformerEncoder, self).__init__()

        self.transformerlayer = TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=1024)

    def forward(self, src, ref):
        ref = ref.permute(1, 0, 2) # N, B, C
        src = src.permute(1, 0, 2)

        # Cross attention for reference and class token.
        mix = self.transformerlayer(src, ref) # query: src. key, value: ref
        codes = mix.permute(1, 0, 2)

        return codes

class LCNet_40(nn.Module):
    def __init__(self, fmaps=[9216, 2048, 512], n_classes=40, activ='relu'):
        super().__init__()
        self.n_classes = n_classes
        # Linear layers
        self.fcs = nn.ModuleList()
        for i in range(len(fmaps)-1):
            in_channel = fmaps[i]
            out_channel = fmaps[i+1]
            self.fcs.append(nn.Linear(in_channel, out_channel, bias=True))
        self.indep_fcs = nn.ModuleList()
        self.output_fcs = nn.ModuleList()
        # Independent linear layers for embeddings
        for _ in range(n_classes):
            self.indep_fcs.append(nn.Linear(512, 32, bias=True))
            self.output_fcs.append(nn.Linear(32, 1, bias=True))

        # Activation
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        for layer in self.fcs:
            x = self.relu(layer(x))
        pre_list = []
        out_list = []
        for i in range(self.n_classes):
            out = self.indep_fcs[i](x)
            out_list.append(out)
            pre_list.append(self.output_fcs[i](out))
        pre = torch.cat(pre_list, dim=1) # Scores (B, n_classes)
        feat = torch.stack(out_list, dim=1) # Embeddings (B, n_classes, 32)
        return pre, feat
