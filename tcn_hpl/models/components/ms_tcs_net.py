import torch
from torch import nn
import torch.nn.functional as F
import copy
import einops

class MultiStageModel(nn.Module):
    def __init__(self, 
                 num_stages, 
                 num_layers, 
                 num_f_maps, 
                 dim, 
                 num_classes,
                 window_size,):
        """Initialize a `MultiStageModel` module.

        :param num_stages: Nubmer of State Model Layers.
        :param num_layers: Number of Layers within each State Model.
        :param num_f_maps: Feature size within the state model
        :param dim: Feature size between state models.
        :param num_classes: Number of output classes.
        """
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        print(f"num classes: {num_classes}")
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)
                )
                for s in range(num_stages - 1)
            ]
        )
        
        
        self.fc = nn.Sequential(
            
            nn.Linear(dim*window_size, 4096),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(4096, 8192),
            nn.Dropout(0.25),
            nn.Linear(8192, 16384),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(16384, 8192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(8192, 4096),
            nn.Dropout(0.25),
            nn.Linear(4096, dim*window_size),
            )
        # self.fc1 = nn.Linear(dim*30, 4096)
        # self.act = nn.GELU()
        # self.drop1 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(4096, 8192)
        # self.drop2 = nn.Dropout(0.1)

        # self.fc3 = nn.Linear(8192, 16384)
        # self.act3 = nn.GELU()
        # self.drop3 = nn.Dropout(0.1)
        # self.fc4 = nn.Linear(16384, dim*30)
        
        # self.fc = nn.Linear(1280, 2048)

    def forward(self, x, mask):
        b, d, c = x.shape
        # print(f"x: {x.shape}")
        # print(f"mask: {mask.shape}")
        
        re_x = einops.rearrange(x, 'b d c -> b (d c)')
        re_x = self.fc(re_x)
        x = einops.rearrange(re_x, 'b (d c) -> b d c', d=d, c=c)
        # print(f"re_x: {re_x.shape}")
        # print(f"x: {x.shape}")
        
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, None, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        # print(f"outputs: {outputs.shape}")
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, None, :]
        
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.BatchNorm1d(out_channels)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride=1)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.activation(out)
        # out = self.pool(out)
        out = self.norm(out)
        out = self.dropout(out)
        return (x + out) * mask[:, None, :]


if __name__ == "__main__":
    _ = MultiStageModel()
