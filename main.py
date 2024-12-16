import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
from darts import TimeSeries
import matplotlib.pyplot as plt

from data_loader import TimeSeriesLoader
from informer_model import Informer
from mtcn_model import TCNForecaster


def preprocess_data_feature(file_path, target_column, date_columns):
    """
    Preprocess the data:
    - Load data
    - Clean and process features
    - Scale numerical and categorical features
    - Return processed data and scaled data
    """
    df = pd.read_csv(file_path)

    # Ensure column names are unique
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns found: {df.columns[df.columns.duplicated()]}")

    # Fill missing values and drop rows with NA
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['No'], inplace=True)  # Drop unwanted columns

    # Combine year, month, day, and hour into a single datetime column
    df['datetime'] = pd.to_datetime(df[date_columns])

    # Separate target column and combine datetime components
    target_series = df[target_column]

    # Drop individual datetime components and keep only 'datetime' in features
    df.drop(columns=date_columns, inplace=True)
    feature_data = df.drop(columns=[target_column, 'datetime'])

    # Identify numerical and categorical columns
    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply scaling and one-hot encoding
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    transformed_features = preprocessor.fit_transform(feature_data)

    feature_names = numerical_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    )

    # Add 'datetime' column back as part of the selected features (feature engineering)
    datetime_feature = (df['datetime'].
                        apply(lambda x: x.hour + x.day * 24 + x.month * 30 * 24 + x.year * 365 * 24).
                        values.reshape(-1, 1))

    # Concatenate the transformed features with the datetime feature
    features_with_datetime = np.hstack((transformed_features, datetime_feature))

    # Include datetime and target in the final dataframe
    final_df = pd.concat([
        df['datetime'],
        pd.DataFrame(features_with_datetime, columns=feature_names + ['datetime']),
        target_series
    ], axis=1)

    return {
        'features': features_with_datetime,
        'feature_names': feature_names + ['datetime'],
        'date': df['datetime'],
        'target': target_series,
        'scalers': {
            'numeric': preprocessor.named_transformers_['num'],
            'categorical': preprocessor.named_transformers_['cat']
        }
    }


def perform_feature_selection(file_path, selection_method, selection_threshold):
    """
    Perform feature selection using Informer and return the selected features.
    """
    data_config = preprocess_data_feature(
        file_path,
        target_column='pm2.5',
        date_columns=['year', 'month', 'day', 'hour']
    )

    features = data_config['features']
    feature_names = data_config['feature_names']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(features, dtype=torch.float32).to(device)

    # Initialize Informer model for feature selection
    model = Informer(
        enc_in=x.shape[1],
        selection_method=selection_method,
        selection_threshold=selection_threshold
    ).to(device)

    # Get feature importance
    with torch.no_grad():
        importance_scores = model.compute_feature_importance(x)

    # Select features based on importance scores
    max_importance = importance_scores.max().item()
    selected_indices = [i for i, score in enumerate(importance_scores) if score.item() == max_importance]
    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_feature_data = features[:, selected_indices]

    final_df = pd.concat([
        data_config['date'],
        pd.DataFrame(selected_feature_data, columns=selected_feature_names),
        data_config['target']
    ], axis=1)

    return {
        'selected_features': selected_feature_names,
        'full_dataframe': final_df,
        'date': data_config['date'],
        'target': data_config['target']
    }


def collate_fn(batch):
    """
    Custom collate function for padding and batching input data.
    """
    inputs = [item['inputs'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad labels to the same length
    max_length = max(label.size(0) for label in labels)
    labels_padded = [torch.nn.functional.pad(label, (0, max_length - label.size(0))) for label in labels]

    inputs_padded = torch.stack(inputs)
    labels_padded = torch.stack(labels_padded)

    return {'inputs': inputs_padded, 'labels': labels_padded}


def main():
    """
    Main function to load data, perform feature selection, and train the model.
    """
    file_path = "beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv"

    # Perform feature selection using the Informer model
    selection_results = perform_feature_selection(file_path, 'importance', 0.3)

    # Prepare the dataset based on the selected features
    training_df = selection_results['full_dataframe']
    training_df.ffill(inplace=True)

    # Ensure 'datetime' column is in the correct format (before conversion to datetime type)
    training_df.drop_duplicates(subset=['datetime'], inplace=True)

    # Now, ensure 'datetime' column is properly parsed as datetime type
    training_df['datetime'] = pd.to_datetime(training_df['datetime'], errors='coerce')

    # Handle any NaT (Not a Time) values resulting from coercion
    training_df.dropna(subset=['datetime'], inplace=True)
    training_df.drop_duplicates(subset=['datetime'], inplace=True)

    # Check the 'datetime' column and its dimensions
    # print("Datetime column shape:", training_df['datetime'].shape)
    # print("Datetime column head:", training_df['datetime'].head())
    # print("Datetime type:", training_df['datetime'].dtypes)

    # Reset index if 'datetime' is in the index
    if isinstance(training_df.index, pd.DatetimeIndex):
        training_df.reset_index(inplace=True)

    print("Check columns: ", training_df.columns)

    # Create 'unique_id' to ensure there are no further duplicates
    if 'datetime' in training_df.columns:
        training_df['unique_id'] = training_df.groupby('datetime').cumcount() + 1

    # Drop any remaining duplicates just in case
    training_df.drop_duplicates(subset=['datetime'], inplace=True)

    # Create TimeSeriesLoader with the selected features
    train_dataset = TimeSeriesLoader(
        df=training_df,
        selected_features=selection_results['selected_features'],
        timestamp_col='datetime',
        kalman_col='pm2.5',
        sequence_length=30
    )

    # Create DataLoader with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize the TCN model
    input_s = len(selection_results['selected_features'])
    input_s += 2
    model = TCNForecaster(
        input_size=input_s,
        output_size=1,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2
    )

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(10):
        for batch in train_loader:
            inputs = batch['inputs']
            targets = batch['labels']

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()  # Set the model to evaluation mode
    predictions = []
    actual_value = []

    # Loop over the training data for predictions
    with torch.no_grad():
        for batch in train_loader:
            inputs = batch['inputs']
            targets = batch['labels']

            # Make predictions
            outputs = model(inputs)
            predictions.append(outputs.numpy().flatten())  # Flatten to make it 1D for plotting
            actual_value.append(targets.numpy().flatten())

    # Convert predictions and actuals to numpy arrays
    predictions = np.concatenate(predictions)
    actual_value = np.concatenate(actual_value)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_value, label='Actual PM2.5')
    plt.plot(predictions, label='Predicted PM2.5', linestyle='--')
    plt.title('Actual vs Predicted PM2.5')
    plt.xlabel('Time')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.show()

    # Print a few sample predictions and actual values
    for i in range(10):
        print(f"Actual: {actual_value[i]:.4f}, Predicted: {predictions[i]:.4f}")


if __name__ == "__main__":
    main()
