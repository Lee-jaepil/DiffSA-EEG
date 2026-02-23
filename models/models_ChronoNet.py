import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels, 32, kernel_size=(1, 8), stride=(1, 2), padding=(0, 3))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class ChronoNet(nn.Module):
    def __init__(self, input_channels=21, sequence_length=250):
        super(ChronoNet, self).__init__()
        self.inception1 = Inception(input_channels)
        self.inception2 = Inception(96)
        self.inception3 = Inception(96)

        # Calculate the size after Inception layers (3 layers with stride 2)
        self.inception_output_size = sequence_length // 8

        self.gru1 = nn.GRU(96 * self.inception_output_size, 32, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.gru4 = nn.GRU(96, 32, batch_first=True)

        # LayerNorm for each GRU output
        self.ln1 = nn.LayerNorm(32)
        self.ln2 = nn.LayerNorm(32)
        self.ln3 = nn.LayerNorm(32)
        self.ln4 = nn.LayerNorm(32)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # x shape: (B, T, C, L) -> (B * T, C, L)
        B, T, C, L = x.size()
        x = x.view(B * T, C, L)

        # Add extra dimension for 2D convolutions
        x = x.unsqueeze(2)  # (B * T, C, 1, L)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        # Reshape for GRU layers
        x = x.view(B, T, -1)  # (B, T, 96 * inception_output_size)

        x, _ = self.gru1(x)
        x = self.ln1(x)
        x = self.dropout(x)
        x_res = x

        x, _ = self.gru2(x)
        x = self.ln2(x)
        x = self.dropout(x)
        x_res2 = x

        x_cat1 = torch.cat([x_res, x], dim=2)
        x, _ = self.gru3(x_cat1)
        x = self.ln3(x)
        x = self.dropout(x)

        x = torch.cat([x_res, x_res2, x], dim=2)
        x, _ = self.gru4(x)
        x = self.ln4(x)
        x = self.dropout(x)

        # Use the last output of the GRU
        x = x[:, -1, :]  # (B, 32)

        x = self.fc(x)  # (B, 2)

        return x
