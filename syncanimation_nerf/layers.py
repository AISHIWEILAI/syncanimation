from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch



class ConvNormRelu(nn.Module):
    def __init__(self, conv_type='1d', in_channels=3, out_channels=64, downsample=False,
                 kernel_size=None, stride=None, padding=None, norm='BN', leaky=False):
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == '2d':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError
        elif conv_type == '1d':
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'IN':
                self.norm = nn.InstanceNorm1d(out_channels)
            else:
                raise NotImplementedError
        nn.init.kaiming_normal_(self.conv.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))  # normalize on [C]
        else:
            x = self.norm(x)
        x = self.act(x)
        return x

class FCNormRelu(nn.Module):
    def __init__(self, in_features=256, out_features=256, norm='BN', leaky=False, drop_out=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if norm == 'BN':
            self.norm = nn.BatchNorm1d(out_features, eps=1e-5)
        elif norm == 'IN':
            self.norm = nn.InstanceNorm1d(out_features, eps=1e-5)
        nn.init.kaiming_normal_(self.fc.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        self.dp = nn.Dropout(p=drop_out)

    def forward(self, x):

        x = self.fc(x)
        if isinstance(self.norm, nn.InstanceNorm1d):
            x = self.norm(x)
        else:
            x = self.norm(x)

        x = self.act(x)
        x = self.dp(x)
        return x


class PoseDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        """Per-frame dataset from {dir_name: (label, N x 6 tensor)} dict."""
        self.data_dict = data_dict
        self.transform = transform
        self.data_keys = list(data_dict.keys())
        self.data = []

        for dir_name, (c, input_data) in data_dict.items():
            for i in range(input_data.shape[0]):
                self.data.append((dir_name, c, input_data[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dir_name, c, frame_input = self.data[idx]

        if self.transform:
            frame_input = self.transform(frame_input)

        return dir_name, c, frame_input

    def get_video(self, dir_name):
        """Return full video data and label by directory name."""
        if dir_name in self.data_dict:
            return self.data_dict[dir_name]
        else:
            raise ValueError(f"Video '{dir_name}' not found in the dataset.")

