from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from filterpy.kalman import KalmanFilter
import torch


class TimeSeriesLoader(Dataset):
    def __init__(self, df, selected_features, timestamp_col, kalman_col, sequence_length):
        self.sequence_length = sequence_length
        self.selected_features = selected_features
        self.timestamp_col = timestamp_col
        self.kalman_col = kalman_col
        self.categorical_features = None
        self.numerical_features = None
        self.norm_train_data = None
        self.log_train_data = None

        # Add datetime and pm2.5 columns to selected features
        self.selected_features += ['datetime', 'pm2.5']

        # Load and preprocess data
        self.train_data = df
        self.train_data[self.timestamp_col] = pd.to_datetime(self.train_data[self.timestamp_col])
        self.train_data.set_index(self.timestamp_col, inplace=True)
        self.train_data.drop_duplicates(inplace=True)

        # Optionally, reset index to keep datetime as a column again
        self.train_data.reset_index(inplace=True)

        # Prepare the data
        self.clean_data()
        self.preprocess_data()
        self.feature_engineering()
        self.add_temporal_features()

        # Filter to relevant features and ensure no missing values
        self.train_data = self.train_data[self.selected_features]
        print(self.train_data.head(5))
        self.train_data.dropna(inplace=True)

    def clean_data(self):
        # Check if the datetime column is of correct dtype
        print("Datetime column dtype before conversion:", self.train_data['datetime'].dtype)

        # Ensure 'datetime' column is converted to datetime type
        self.train_data['datetime'] = pd.to_datetime(self.train_data['datetime'])

        # Check dtype again after conversion
        print("Datetime column dtype after conversion:", self.train_data['datetime'].dtype)

        # Set 'datetime' column as index
        self.train_data.set_index('datetime', inplace=True)

        # Verify the index
        print("Index after setting datetime:", self.train_data.index)

        # Handle missing values with time-based interpolation for numerical columns only
        numeric_cols = self.train_data.select_dtypes(include=['float64', 'int64']).columns
        self.train_data[numeric_cols] = self.train_data[numeric_cols].interpolate(method='time', axis=0)

        # Identify categorical and numerical features
        self.categorical_features = []
        self.numerical_features = []

        for col, dtype in self.train_data.dtypes.items():
            if dtype == 'object' or self.train_data[col].nunique() <= 10:
                self.categorical_features.append(col)
            elif dtype in ['int64', 'float64']:
                self.numerical_features.append(col)

        # Drop any duplicate rows (including datetime-based duplicates)
        self.train_data.drop_duplicates(inplace=True)

        # Check for duplicate timestamps
        duplicate_timestamps = self.train_data[self.train_data.index.duplicated()]
        if not duplicate_timestamps.empty:
            print("columns header: ", duplicate_timestamps.columns)
            print(f"Duplicate timestamps found: \n{duplicate_timestamps}")
            # You can handle this by dropping duplicates or aggregating them
            self.train_data = self.train_data.loc[~self.train_data.index.duplicated(keep='first')]

        # Reset index to include 'datetime' as a column again (optional)
        self.train_data.reset_index(inplace=True)

    def preprocess_data(self):
        # Normalize numerical features
        scaler = StandardScaler()
        numeric_cols = self.train_data.select_dtypes(include=['float64', 'int64']).columns

        self.norm_train_data = pd.DataFrame(
            scaler.fit_transform(self.train_data[numeric_cols]),
            columns=numeric_cols,
            index=self.train_data.index
        )

        # Log-transform normalized data, handling non-positive values
        self.norm_train_data[self.norm_train_data <= 0] = np.nan
        self.log_train_data = np.log(self.norm_train_data)
        self.log_train_data.fillna(method='ffill', inplace=True)

        self.train_data = self.train_data.loc[~self.train_data.index.duplicated(keep='first')]

    def feature_engineering(self, process_noise_covariance=None, measurement_noise_covariance=None):
        # Apply Kalman filter to smooth the target feature
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x = np.array([[0], [0]])
        kf.P *= 1000
        kf.Q = process_noise_covariance or np.array([[1, 0], [0, 1]])
        kf.R = measurement_noise_covariance or np.array([[1]])

        # Apply Kalman filtering
        data_to_filter = self.train_data[self.kalman_col].values
        estimated_positions = []
        for z in data_to_filter:
            kf.predict()
            kf.update(z)
            estimated_positions.append(kf.x[0, 0])

        self.train_data['kalman_estimate'] = estimated_positions
        self.train_data = self.train_data.loc[~self.train_data.index.duplicated(keep='first')]

    def add_temporal_features(self):
        # Add temporal features based on the 'datetime' column
        print("INDEX: ", self.train_data.index)

        # Ensure the 'datetime' column is being used to extract date-related features
        self.train_data['day_of_month'] = self.train_data['datetime'].dt.day
        self.train_data['week'] = self.train_data['datetime'].dt.isocalendar().week
        self.train_data['month'] = self.train_data['datetime'].dt.month

        # Create cyclical features for periodic data
        self.train_data['sin_day_of_month'] = np.sin(2 * np.pi * self.train_data['day_of_month'] / 31)
        self.train_data['cos_day_of_month'] = np.cos(2 * np.pi * self.train_data['day_of_month'] / 31)
        self.train_data['sin_week'] = np.sin(2 * np.pi * self.train_data['week'] / 52)
        self.train_data['cos_week'] = np.cos(2 * np.pi * self.train_data['week'] / 52)
        self.train_data['sin_month'] = np.sin(2 * np.pi * self.train_data['month'] / 12)
        self.train_data['cos_month'] = np.cos(2 * np.pi * self.train_data['month'] / 12)

    def __len__(self):
        return len(self.train_data) - self.sequence_length

    def __getitem__(self, idx):
        # Ensure that the sequence length is fixed
        sequence_start = max(0, idx - self.sequence_length + 1)
        sequence_end = idx + 1

        # Extract the sequence of data
        sequence_data = self.log_train_data.iloc[sequence_start:sequence_end].copy()

        # Convert datetime to numeric (Unix timestamp)
        if 'datetime' in sequence_data.columns:
            sequence_data['datetime'] = sequence_data['datetime'].astype(int) // 10 ** 9  # Convert to Unix timestamp

        # Prepare inputs (exclude target column)
        inputs_columns = [col for col in sequence_data.columns if col != self.kalman_col]
        inputs = sequence_data[inputs_columns]

        # Convert inputs to float32
        inputs = inputs.astype(np.float32)

        # Prepare labels (target column)
        labels = sequence_data[self.kalman_col].values.astype(np.float32)

        # Reshape inputs for sequence model
        inputs = inputs.values.T  # Transpose to get (features, sequence_length)

        # Ensure inputs are of consistent shape
        if inputs.shape[1] < self.sequence_length:
            # Pad the inputs if they are shorter than the expected length
            padding = np.zeros((inputs.shape[0], self.sequence_length - inputs.shape[1]), dtype=np.float32)
            inputs = np.concatenate((inputs, padding), axis=1)

        return {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
