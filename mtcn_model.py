import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNSingleBlock(nn.Module):
    """
    A single block of the Temporal Convolutional Network (TCN), consisting of
    two dilated causal convolutions with residual connections.
    """
    def __init__(self, input_dim, output_dim, kernel_size, stride, dilation, padding, dropout):
        super(TCNSingleBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride,
                               dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride=stride,
                               dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the first convolution block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Apply the second convolution block
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Add residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN Model with multiple temporal blocks.
    """
    def __init__(self, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.initialized = False

    def initialize_layers(self, input_dim):
        """
        Dynamically initializes the layers based on input dimensions.
        """
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]
            self.layers.append(
                TCNSingleBlock(in_channels, out_channels, self.kernel_size,
                               stride=1, dilation=dilation_size,
                               padding=(self.kernel_size - 1) * dilation_size,
                               dropout=self.dropout)
            )
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize_layers(x.shape[1])  # Dynamically initialize based on input_dim
        for layer in self.layers:
            x = layer(x)
        return x


class TCNForecaster(nn.Module):
    """
    TCN model for multivariate time-series forecasting.
    """
    def __init__(self, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNForecaster, self).__init__()
        # Temporal Convolutional Network (TCN)
        self.tcn = TemporalConvNet(num_channels, kernel_size, dropout)
        # Fully connected layer to map the last time step to the output size
        self.linear = None  # Initialize dynamically

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, input_dim, sequence_length)
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], sequence_length)
        if self.linear is None:
            self.linear = nn.Linear(tcn_out.shape[1], x.shape[1]).to(x.device)  # Initialize linear layer dynamically
        tcn_out = tcn_out[:, :, -1]  # (batch_size, num_channels[-1])
        output = self.linear(tcn_out)  # (batch_size, output_size)
        return output
