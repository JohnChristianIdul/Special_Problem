from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class TimeSeriesLoader(Dataset):
    def __init__(self, df, selected_features, timestamp_col, kalman_col, sequence_length):
        """
        Initialize the TimeSeriesLoader with dataset preparation steps.
        """
        self.sequence_length = sequence_length
        self.selected_features = selected_features
        self.timestamp_col = timestamp_col
        self.kalman_col = kalman_col
        self.log_train_data = None
        self.norm_train_data = None

        # Ensure that the dataframe contains necessary columns
        for feature in self.selected_features:
            if feature not in df.columns:
                print(f"Feature {feature} is missing. Adding it with default value 0.")
                df[feature] = 0

        self.train_data = df
        self.clean_data()
        self.preprocess_data()
        self.feature_engineering()
        self.add_temporal_features()

        # Filter to relevant features and ensure no missing values
        self.train_data = self.train_data[self.selected_features].dropna()

    def clean_data(self):
        """
        Clean the dataset by handling missing values, setting the datetime index, and removing duplicates.
        """
        # Ensure 'datetime' is in the correct format
        self.train_data[self.timestamp_col] = pd.to_datetime(self.train_data[self.timestamp_col], errors='coerce')

        # Set the datetime column as the index before interpolation
        self.train_data.set_index(self.timestamp_col, inplace=True, drop=False)

        # Print to verify index and dtypes
        print("index: \n", self.train_data.index, "\ndtypes: \n", self.train_data.dtypes)

        # Handle missing values across the dataset before doing anything
        # result = seasonal_decompose(self.train_data['pm2.5'], model='additive', period=12, extrapolate_trend='freq')
        # self.train_data['pm2.5'] = self.train_data['pm2.5'].fillna(result)
        # self.train_data.ffill(inplace=True)
        # self.train_data.dropna(inplace=True)

        # Check for duplicate datetime entries
        self.train_data.drop_duplicates(subset=[self.timestamp_col], inplace=True)

        # Separate numeric and datetime columns
        datetime_cols = [col for col in self.train_data.columns if
                         pd.api.types.is_datetime64_any_dtype(self.train_data[col])]
        numeric_cols = [col for col in self.train_data.columns if pd.api.types.is_numeric_dtype(self.train_data[col])]

        # Interpolate only numeric columns
        self.train_data[numeric_cols] = self.train_data[numeric_cols].interpolate(method='time', axis=0)

        # Handle any missing values in datetime columns (if necessary)
        self.train_data[datetime_cols] = self.train_data[datetime_cols].ffill()

        # Remove any rows with NaN values (after interpolation)
        self.train_data.dropna(inplace=True)

    def preprocess_data(self):
        """
        Normalize numerical features and apply log transformation.
        """
        # Normalize the numerical columns
        scaler = StandardScaler()
        numeric_cols = self.train_data.select_dtypes(include=['float64', 'int64']).columns

        # Include the kalman_col (pm2.5) in the selected numeric columns
        selected_numeric_cols = [col for col in numeric_cols if col in self.selected_features] + [self.kalman_col]

        # Ensure that the kalman_col exists in the dataset
        if self.kalman_col not in self.train_data.columns:
            raise KeyError(f"Column {self.kalman_col} not found in the training data!")

        # Normalize the selected numeric columns including the kalman_col
        self.norm_train_data = pd.DataFrame(
            scaler.fit_transform(self.train_data[selected_numeric_cols]),
            columns=selected_numeric_cols,
            index=self.train_data.index
        )

        # Log-transform normalized data (handling non-positive values)
        self.log_train_data = np.log(self.norm_train_data.clip(lower=1e-5))

        # Add datetime in ordinal form to log-transformed data
        self.log_train_data['Date Time'] = self.train_data['Date Time'].map(pd.Timestamp.toordinal)

    def feature_engineering(self):
        """
        Apply Kalman Filter for smoothing the target feature.
        """
        # Apply Kalman filter on the target column (kalman_col)
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.x = np.array([[0], [0]])
        kf.P *= 1000
        kf.Q = np.array([[1, 0], [0, 1]])
        kf.R = np.array([[1]])

        data_to_filter = self.train_data[self.kalman_col].values
        estimated_positions = []
        for z in data_to_filter:
            kf.predict()
            kf.update(z)
            estimated_positions.append(kf.x[0, 0])

        self.train_data['kalman_estimate'] = estimated_positions

    def add_temporal_features(self):
        """
        Generate temporal features such as day of month, week, and month.
        """
        # Add cyclical features for day of month, week, and month
        self.train_data['day_of_month'] = self.train_data['Date Time'].dt.day
        self.train_data['week'] = self.train_data['Date Time'].dt.isocalendar().week
        self.train_data['month'] = self.train_data['Date Time'].dt.month

        # Generate sin/cos transformations for cyclical features
        self.train_data['sin_day_of_month'] = np.sin(2 * np.pi * self.train_data['day_of_month'] / 31)
        self.train_data['cos_day_of_month'] = np.cos(2 * np.pi * self.train_data['day_of_month'] / 31)
        self.train_data['sin_week'] = np.sin(2 * np.pi * self.train_data['week'] / 52)
        self.train_data['cos_week'] = np.cos(2 * np.pi * self.train_data['week'] / 52)
        self.train_data['sin_month'] = np.sin(2 * np.pi * self.train_data['month'] / 12)
        self.train_data['cos_month'] = np.cos(2 * np.pi * self.train_data['month'] / 12)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.train_data) - self.sequence_length

    def __getitem__(self, idx):
        """
        Fetch a sequence of data for a given index.
        """
        sequence_start = max(0, idx - self.sequence_length + 1)
        sequence_end = idx + 1

        # Extract the sequence data
        sequence_data = self.log_train_data.iloc[sequence_start:sequence_end].copy()

        # Ensure pm2.5 is included in the inputs
        inputs_columns = [col for col in sequence_data.columns]
        inputs = sequence_data[inputs_columns].values.T.astype(np.float32)

        # Check if the kalman_col (pm2.5) exists in sequence_data
        if self.kalman_col not in sequence_data.columns:
            raise KeyError(f"Column {self.kalman_col} not found in the sequence data!")

        # Now, include pm2.5 as part of the labels (this is the column we use for labels)
        labels = sequence_data[self.kalman_col].values.astype(np.float32)

        # Pad the inputs if they are shorter than the expected sequence length
        if inputs.shape[1] < self.sequence_length:
            padding = np.zeros((inputs.shape[0], self.sequence_length - inputs.shape[1]), dtype=np.float32)
            inputs = np.concatenate((inputs, padding), axis=1)

        return {
            'inputs': torch.tensor(inputs, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
