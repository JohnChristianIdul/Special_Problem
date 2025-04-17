import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.5):
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
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
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
    return TCNForecaster(input_size, output_size, num_channels, kernel_size=3, dropout=0.5)


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

    def train_step(self, batch_x, batch_y):
        """Single training step for one batch"""
        self.model.train()
        try:
            # Outputs and batch_y would now be [batch_size, sequence_length, features]
            outputs = self.model(batch_x)
            outputs = outputs.view(-1)
            batch_y = batch_y.view(-1)

            loss = self.criterion(outputs, batch_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except Exception as e:
            print(f"Error during training step: {e}")
            print(f"Input shape: {batch_x.shape}")
            return 0

    def validate_step(self, batch_x, batch_y):
        """Single validation step for one batch"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            return loss.item()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            try:
                if isinstance(batch, dict):
                    batch_x = batch['inputs'].to(self.device)
                    batch_y = batch['labels'].to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                # Check dimensions
                assert batch_x.size(0) == batch_y.size(
                    0), f"Mismatch in batch sizes - Inputs: {batch_x.size(0)}, Targets: {batch_y.size(0)}"

                loss = self.train_step(batch_x, batch_y)
                total_loss += loss
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"Input shape: {batch['inputs'].shape}, Target shape: {batch['labels'].shape}")
                else:
                    print(f"Input shape: {batch.shape}")
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        nonzero_mask = all_targets != 0
        mape = np.mean(
            np.abs((all_targets[nonzero_mask] - all_preds[nonzero_mask]) / all_targets[nonzero_mask])) * 100 if np.any(
            nonzero_mask) else float('inf')

        print(f"Validation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        return avg_loss, mse, rmse, mae, mape

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
    # num_channels = [16, 32, 64]
    num_channels = [64, 128, 256]

    model = create_forecaster(input_size, output_size, num_channels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
    trainer = TCNTrainer(model, criterion, optimizer)

    history = {'train_loss': [], 'val_loss': []}

    print("Start training...")

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss[0]:.4f}, MSE: {val_loss[1]:.4f}, RMSE: {val_loss[2]:.4f}, '
                  f'MAE: {val_loss[3]:.4f}, MAPE: {val_loss[4]:.2f}%')

    return trainer, history
