import torch
import torch.nn as nn


class TCNSingleBlock(nn.Module):
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

        # Residual connection (downsample if input_dim != output_dim)
        self.downsample = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None

    def forward(self, x):
        # Apply the first convolution block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Apply the second convolution block
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection (ensure shapes match)
        res = x if self.downsample is None else self.downsample(x)

        # Ensure the sequence length of `out` matches `res`
        if out.size(2) != res.size(2):
            if out.size(2) > res.size(2):
                out = out[:, :, :res.size(2)]  # Slice the larger tensor
            else:
                padding = (0, res.size(2) - out.size(2))  # Calculate padding
                out = torch.nn.functional.pad(out, padding)  # Pad the smaller tensor

        return self.relu1(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Initialize layers (first layer should take input_dim as 3, as your data has 3 features)
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else self.num_channels[i - 1]  # First layer takes input_dim as input
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
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNForecaster, self).__init__()
        # Temporal Convolutional Network (TCN)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        # Fully connected layer to output 1 value (scalar)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, input_size, sequence_length)
        Returns:
            Tensor of shape (batch_size, output_size)
        """
        print(f"Input shape before preprocessing: {x.shape}")  # Debugging the input shape

        # Ensure x has 3 dimensions: (batch_size, input_size, sequence_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        x = x.permute(0, 1, 2)

        # Pass through the Temporal Convolutional Network
        tcn_out = self.tcn(x)

        # Extract the last time step (last feature across the sequence)
        tcn_out = tcn_out[:, :, -1]  # We can use the last time step for prediction

        # Pass through the fully connected layer to get the final output
        output = self.linear(tcn_out)
        print(f"Output shape: {output.shape}")

        return output
