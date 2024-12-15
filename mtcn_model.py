import torch
import torch.nn as nn


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
        print("Input shape to conv1:", x.shape)  # Print shape before applying conv1
        # Apply the first convolution block
        out = self.conv1(x)
        print("Shape after conv1:", out.shape)  # Print shape after applying conv1
        out = self.relu1(out)
        print("Shape after relu1:", out.shape)
        out = self.dropout1(out)
        print("Shape after dropout1:", out.shape)

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
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Initialize layers
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim
            out_channels = self.num_channels[i]
            self.layers.append(
                TCNSingleBlock(in_channels, out_channels, self.kernel_size,
                               stride=1, dilation=dilation_size,
                               padding=(self.kernel_size - 1) * dilation_size,
                               dropout=self.dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TCNForecaster(nn.Module):
    """
    TCN model for multivariate time-series forecasting.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNForecaster, self).__init__()
        # Temporal Convolutional Network (TCN)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # Fully connected layer to map the last time step to the output size
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, input_size, sequence_length)
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        # Ensure x is in the right shape (batch_size, input_size, sequence_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], sequence_length)
        tcn_out = tcn_out[:, :, -1]  # Take the last time step (batch_size, num_channels[-1])
        output = self.linear(tcn_out)  # (batch_size, output_size)
        return output
