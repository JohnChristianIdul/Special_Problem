import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TCNSingleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, dilation, padding, dropout):
        super(TCNSingleBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        # Match sequence lengths
        if out.size(2) != res.size(2):
            diff = abs(out.size(2) - res.size(2))
            if out.size(2) > res.size(2):
                out = out[:, :, :-diff]
            else:
                res = res[:, :, :-diff]

        return self.relu1(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.layers.append(
                TCNSingleBlock(in_channels, out_channels, kernel_size,
                               stride=1, dilation=dilation_size,
                               padding=(kernel_size - 1) * dilation_size,
                               dropout=dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TCNForecaster(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNForecaster, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: [batch, sequence_length, channels]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # Permute to [batch, channels, sequence_length]
        x = x.permute(0, 2, 1)
        # print(f"Shape after permute: {x.shape}")
        tcn_out = self.tcn(x)
        tcn_out = tcn_out[:, :, -1]
        output = self.linear(tcn_out)
        return output


def create_forecaster(input_size, output_size, num_channels):
    return TCNForecaster(input_size, output_size, num_channels, kernel_size=3, dropout=0.2)


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, sequence_length):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length + 1)

    def __getitem__(self, idx):
        if idx + self.sequence_length > len(self.features):
            raise IndexError("Index exceeds dataset length.")
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        return x, y


class TCNTrainer:
    def __init__(self, model, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            try:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Input shape: {batch_x.shape}")
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            output = self.model(x)
            return output.cpu().numpy()


def train_and_predict(features, targets, sequence_length, num_epochs, batch_size):
    dataset = TimeSeriesDataset(features, targets, sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_size = features.shape[1]
    output_size = 1
    num_channels = [16, 32, 64]

    # print(f"Input size: {input_size}")

    model = create_forecaster(input_size, output_size, num_channels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TCNTrainer(model, criterion, optimizer)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return trainer, history
