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


def preprocess_data_feature(file_path, target_column):
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

    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    # Fill missing values and drop rows with NA
    # df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Combine year, month, day, and hour into a single datetime column
    # df['Date Time'] = pd.to_datetime(df[date_columns])

    # Separate target column and combine datetime components
    target_series = df[target_column]

    # Drop individual datetime components and keep only 'Date Time' in features
    # df.drop(columns=date_columns, inplace=True)
    feature_data = df.drop(columns=[target_column, 'Date Time'])

    print("Checkpoint 1")
    # Identify numerical and categorical columns
    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()

    print("Checkpoint 2")
    # Apply scaling and one-hot encoding
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    print("Checkpoint 3")
    transformed_features = preprocessor.fit_transform(feature_data)
    #
    # feature_names = numerical_cols + list(
    #     preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    # )
    feature_names = numerical_cols

    print("Checkpoint 4")
    # Define a reference start date (e.g., the earliest date in your dataset)
    start_date = df['Date Time'].min()

    # Calculate the difference in hours from the start date (vectorized operation)
    df['datetime_feature'] = (df['Date Time'] - start_date).dt.total_seconds() / 3600

    # Now, the 'datetime_feature' is a column with numeric values representing the number of hours
    datetime_feature = df['datetime_feature'].values.reshape(-1, 1)

    # Checkpoint: print out the first few values to ensure it's correct
    print("Checkpoint 5")
    print("head of datetime_feature: \n", df['datetime_feature'].head(5))

    # Assuming transformed_features is your processed feature matrix
    # Ensure the shape of transformed_features matches the required number of rows (420,551)
    print("Checkpoint 5")
    print("tfeatures shape: ", transformed_features.shape)  # Should be (420551, n_features)
    print("dfeatures shape: ", datetime_feature.shape)  # Should be (420551, 1)

    # Concatenate the transformed features with the datetime feature
    features_with_datetime = np.hstack((transformed_features, datetime_feature))

    # Final shape after concatenation
    print("Shape after concatenation: ", features_with_datetime.shape)

    print("Checkpoint 6")
    # Include datetime and target in the final dataframe
    final_df = pd.concat([
        df['Date Time'],
        pd.DataFrame(features_with_datetime, columns=feature_names + ['Date Time']),
        target_series
    ], axis=1)

    # Print the dimensions of the final dataframe
    print(f"Dimensions of the final processed data: {final_df.shape}")

    return {
        'features': features_with_datetime,
        'feature_names': feature_names + ['Date Time'],
        'date': df['Date Time'],
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
        target_column='T (degC)'
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
    file_path = "jena_climate_2009_2016.csv/jena_climate_2009_2016.csv"

    # Perform feature selection using the Informer model
    selection_results = perform_feature_selection(file_path, 'importance', 0.5)

    # Prepare the dataset based on the selected features
    training_df = selection_results['full_dataframe']
    training_df.ffill(inplace=True)

    # Ensure 'Date Time' column is in the correct format (before conversion to datetime type)
    training_df.drop_duplicates(subset=['Date Time'], inplace=True)

    # Now, ensure 'Date Time' column is properly parsed as datetime type
    training_df['Date Time'] = pd.to_datetime(training_df['Date Time'], errors='coerce')

    # Handle any NaT (Not a Time) values resulting from coercion
    # training_df.dropna(subset=['Date Time'], inplace=True)
    training_df.drop_duplicates(subset=['Date Time'], inplace=True)

    # Check the 'Date Time' column and its dimensions
    # print("Datetime column shape:", training_df['Date Time'].shape)
    # print("Datetime column head:", training_df['Date Time'].head())
    # print("Datetime type:", training_df['Date Time'].dtypes)

    # Reset index if 'Date Time' is in the index
    if isinstance(training_df.index, pd.DatetimeIndex):
        training_df.reset_index(inplace=True)

    print("Check columns: ", training_df.columns)

    # Create 'unique_id' to ensure there are no further duplicates
    if 'Date Time' in training_df.columns:
        training_df['unique_id'] = training_df.groupby('Date Time').cumcount() + 1

    print("Features selected: \n", selection_results['selected_features'])
    # Drop any remaining duplicates just in case
    training_df.drop_duplicates(subset=['Date Time'], inplace=True)

    # Create TimeSeriesLoader with the selected features
    train_dataset = TimeSeriesLoader(
        df=training_df,
        selected_features=selection_results['selected_features'],
        timestamp_col='Date Time',
        kalman_col='T (degC)',
        sequence_length=30
    )

    # Create DataLoader with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=250, shuffle=False, collate_fn=collate_fn)

    # Initialize the TCN model
    input_s = len(selection_results['selected_features'])
    input_s += 2
    model = TCNForecaster(
        input_size=input_s,
        output_size=30,
        num_channels=[8, 16, 32],
        kernel_size=3,
        dropout=0.2
    )

    print("Check for Nan values: \n", training_df.values == 0)

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
            predictions.append(outputs.numpy().flatten())
            actual_value.append(targets.numpy().flatten())

    # Convert predictions and actuals to numpy arrays
    predictions = np.concatenate(predictions)
    actual_value = np.concatenate(actual_value)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_value, label='Actual')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted T (degC)')
    plt.xlabel('Date Time')
    plt.ylabel('T (degC)')
    plt.legend()
    plt.show()

    # Print a few sample predictions and actual values
    for i in range(10):
        print(f"Actual: {actual_value[i]:.4f}, Predicted: {predictions[i]:.4f}")


if __name__ == "__main__":
    main()
