import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader
from darts import TimeSeries
import matplotlib.pyplot as plt

from data_loader import TimeSeriesLoader
from informer_model import Informer
from mtcn_model import TCNForecaster


def preprocess_data_feature(file_path, target_column='pm2.5', date_columns=['year', 'month', 'day', 'hour']):
    """
    Preprocess the data, performing the following steps:
    - Load data
    - Drop NA values
    - Drop the column 'No'
    - Separate the target and date columns
    - Combine year, month, day, and hour into an aggregate datetime column
    - Identify numerical and categorical features
    - One-hot encode categorical data
    - Scale numerical features
    - Combine features
    - Return features, feature names, date, target, and scalers (numerical and categorical).

    Parameters:
        file_path (str): Path to the CSV file containing the data.
        target_column (str): Name of the target column (default: 'PM2.5').
        date_columns (list): List of column names representing the date components (default: ['year', 'month', 'day', 'hour']).

    Returns:
        dict: A dictionary containing processed features, feature names, date, target, and scalers.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    import numpy as np

    # Load the data
    df = pd.read_csv(file_path)

    # Ensure column names are unique
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns found: {df.columns[df.columns.duplicated()].tolist()}")

    # Forward fill missing values, drop any remaining NA values
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['No'], inplace=True)

    # Separate target column
    target_series = df[target_column]

    # Create an aggregate datetime column
    df['datetime'] = pd.to_datetime(df[date_columns].rename(columns={
        date_columns[0]: 'year', date_columns[1]: 'month', date_columns[2]: 'day', date_columns[3]: 'hour'
    }))

    # Drop the original date columns
    df.drop(columns=date_columns, inplace=True)

    # Separate features (excluding target and datetime columns)
    feature_data = df.drop(columns=[target_column, 'datetime'])

    # Identify numerical and categorical columns
    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encode categorical variables and scale numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Apply transformations
    transformed_features = preprocessor.fit_transform(feature_data)

    # Get the names of the transformed features
    feature_names = numerical_cols + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    )

    # Return the processed data
    return {
        'features': transformed_features,
        'feature_names': feature_names,
        'date': df['datetime'],
        'target': target_series,
        'scalers': {
            'numeric': preprocessor.named_transformers_['num'],
            'categorical': preprocessor.named_transformers_['cat']
        }
    }


def perform_feature_selection(file_path, selection_method, selection_threshold):
    """
    Perform feature selection using Informer
    """
    # Preprocess data
    data_config = preprocess_data_feature(file_path)
    features = data_config['features']
    feature_names = data_config['feature_names']
    date_series = data_config['date']
    target_series = data_config['target']

    # Add date column as a feature
    date_feature = pd.to_numeric(pd.to_datetime(date_series)).values.reshape(-1, 1)
    features = np.hstack((features, date_feature))
    feature_names.append('datetime')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensor and move to device
    x = torch.tensor(features, dtype=torch.float32).to(device)

    # Initialize Informer for feature selection
    model = Informer(
        enc_in=x.shape[1],
        selection_method=selection_method,
        selection_threshold=selection_threshold
    ).to(device)

    # Compute feature importance
    with torch.no_grad():
        importance_scores = model.compute_feature_importance(x)

    # Select features based on importance
    max_importance = importance_scores.max().item()
    selected_indices = [i for i, score in enumerate(importance_scores) if score.item() == max_importance]

    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_feature_data = features[:, selected_indices]

    # Combine date and target back into dataframe
    final_df = pd.concat([
        date_series,
        pd.DataFrame(selected_feature_data, columns=selected_feature_names),
        target_series
    ], axis=1)

    return {
        'selected_features': selected_feature_names,
        'full_dataframe': final_df,
        'date': date_series,
        'target': target_series
    }


def main():
    # File path to your dataset
    file_path = "beijing+pm2+5+data/PRSA_data_2010.1.1-2014.12.31.csv"

    # Perform feature selection
    selection_results = perform_feature_selection(file_path, 'importance', 0.3)

    training_df = selection_results['full_dataframe']
    training_df.ffill(inplace=True)

    # Drop duplicate datetime columns if they exist
    training_df = training_df.loc[:, ~training_df.columns.duplicated()]

    # Drop any duplicate rows (based on all columns or specific columns)
    training_df.drop_duplicates(inplace=True)

    # Create a unique identifier for each row with the same timestamp
    training_df['unique_id'] = training_df.groupby('datetime').cumcount() + 1

    print(training_df.columns.tolist())
    print("train df \n", training_df.head(5))

    print("first checkpoint")
    train_dataset = TimeSeriesLoader(
        df=training_df,
        selected_features=selection_results['selected_features'],
        timestamp_col='datetime',
        kalman_col='pm2.5',
        sequence_length=30
    )

    print("second checkpoint")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    print("third checkpoint")
    # Initialize TCN model
    model = TCNForecaster(
        input_size=len(selection_results['selected_features']),
        output_size=3,
        num_channels=[32, 64, 128],
        kernel_size=3,
        dropout=0.2
    )

    print("fourth checkpoint")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(10):
        for batch in train_loader:
            inputs = batch['inputs']  # Correctly access 'inputs' from the dictionary
            targets = batch['labels']  # Correctly access 'labels' from the dictionary

            # Debugging prints to check the type and shape of inputs and targets
            print(f"Data type of inputs: {type(inputs)}, Data type of targets: {type(targets)}")
            print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Predictions
    # test_df = pd.read_csv(test_file_path)
    # test_loader = DataLoader(TimeSeriesLoader(test_df, selected_features=selection_results['selected_features'],
    #                                           timestamp_col='Date', kalman_col='Sales', sequence_length=30), batch_size=32)
    # predictions = []
    # with torch.no_grad():
    #     for batch in test_loader:
    #         inputs, _ = batch
    #         predictions.append(model(inputs))

    # Visualization
    # plt.plot(training_df['Sales'], label="Actual")
    # plt.plot(predictions, label="Predicted")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
